"""
Junior Aladdin — AngelOne WebSocket Client
=============================================
WebSocket connection manager for AngelOne SmartAPI feed.

Responsibilities:
  - WebSocket connection & authentication
  - Auto-reconnect with exponential backoff
  - Heartbeat / ping-pong management
  - Subscription management (spot, options, depth)
  - Message routing to receivers

Data Center Architecture compliant.
"""

import asyncio
import json
import msgpack
from typing import Callable, Dict, List, Optional, Set

import websockets
from websockets.exceptions import ConnectionClosed

from loguru import logger

from configs.websocket_config import (
    WS_URL,
    WS_CONNECT_TIMEOUT,
    WS_RECONNECT_DELAY,
    WS_MAX_RECONNECT_DELAY,
    WS_MAX_RECONNECT_ATTEMPTS,
    WS_HEARTBEAT_INTERVAL,
    WS_HEARTBEAT_TIMEOUT,
    WS_PING_INTERVAL,
    WS_PING_TIMEOUT,
    WS_RECV_BUFFER_SIZE,
    WS_SEND_BUFFER_SIZE,
    NIFTY_SPOT_TOKEN,
    OPTIONS_TOKENS,
    DEPTH_TOKENS,
    SUBSCRIPTION_MODE,
)


class AngelWSClient:
    """
    AngelOne WebSocket client with auto-reconnect, heartbeat,
    subscription management, and async message routing.
    """

    def __init__(
        self,
        jwt_token: str,
        client_code: str,
        feed_token: str,
        on_tick: Optional[Callable] = None,
        on_option: Optional[Callable] = None,
        on_depth: Optional[Callable] = None,
        on_error: Optional[Callable] = None,
        on_connection_change: Optional[Callable] = None,
    ):
        """
        Args:
            jwt_token: JWT auth token
            client_code: AngelOne client code
            feed_token: Feed token for subscription
            on_tick: Callback(raw_tick_dict) for NIFTY spot ticks
            on_option: Callback(raw_option_dict) for options ticks
            on_depth: Callback(raw_depth_dict) for market depth
            on_error: Callback(error_message)
            on_connection_change: Callback(is_connected: bool)
        """
        self.jwt_token = jwt_token
        self.client_code = client_code
        self.feed_token = feed_token

        self._on_tick = on_tick
        self._on_option = on_option
        self._on_depth = on_depth
        self._on_error = on_error
        self._on_connection_change = on_connection_change

        # Connection state
        self._ws: Optional[websockets.WebSocketClientProtocol] = None
        self._connected: bool = False
        self._reconnect_attempts: int = 0
        self._should_run: bool = True
        self._subscribed_tokens: Set[str] = set()

        # Internal locks / tasks
        self._heartbeat_task: Optional[asyncio.Task] = None
        self._listener_task: Optional[asyncio.Task] = None
        self._reconnect_task: Optional[asyncio.Task] = None

        # Connection ready event
        self._ready = asyncio.Event()

    # ──────────────────────────────────────────
    # PUBLIC API
    # ──────────────────────────────────────────

    async def connect(self) -> None:
        """
        Start the WebSocket connection and listener loop.
        Will auto-reconnect on disconnect.
        """
        self._should_run = True
        self._listener_task = asyncio.create_task(self._run_connection_loop())

    async def disconnect(self) -> None:
        """Gracefully disconnect the WebSocket."""
        self._should_run = False
        self._ready.clear()

        if self._heartbeat_task:
            self._heartbeat_task.cancel()
            self._heartbeat_task = None

        if self._listener_task:
            self._listener_task.cancel()
            self._listener_task = None

        if self._reconnect_task:
            self._reconnect_task.cancel()
            self._reconnect_task = None

        if self._ws:
            try:
                await self._ws.close()
            except Exception:
                pass
            self._ws = None

        self._set_connected(False)

    async def subscribe_spot(self) -> None:
        """Subscribe to NIFTY spot ticks."""
        await self._subscribe_tokens([NIFTY_SPOT_TOKEN])

    async def subscribe_options(self) -> None:
        """Subscribe to options ticks (if tokens configured)."""
        if OPTIONS_TOKENS:
            await self._subscribe_tokens(OPTIONS_TOKENS)

    async def subscribe_depth(self) -> None:
        """Subscribe to market depth (if tokens configured)."""
        if DEPTH_TOKENS:
            await self._subscribe_tokens(DEPTH_TOKENS)

    async def subscribe_all(self) -> None:
        """Subscribe based on configured SUBSCRIPTION_MODE."""
        if SUBSCRIPTION_MODE == "spot_only":
            await self.subscribe_spot()
        elif SUBSCRIPTION_MODE == "options_only":
            await self.subscribe_options()
        else:
            # "full" — subscribe to everything
            tokens = [NIFTY_SPOT_TOKEN]
            if OPTIONS_TOKENS:
                tokens.extend(OPTIONS_TOKENS)
            if DEPTH_TOKENS:
                tokens.extend(DEPTH_TOKENS)
            await self._subscribe_tokens(tokens)

    @property
    def is_connected(self) -> bool:
        return self._connected

    async def wait_until_ready(self, timeout: Optional[float] = None) -> bool:
        """
        Wait until the WebSocket connection is established and ready.
        Returns True if ready, False if timeout.
        """
        try:
            await asyncio.wait_for(self._ready.wait(), timeout=timeout)
            return self._ready.is_set()
        except asyncio.TimeoutError:
            return False

    # ──────────────────────────────────────────
    # INTERNAL — CONNECTION LOOP
    # ──────────────────────────────────────────

    async def _run_connection_loop(self) -> None:
        """Main loop — connects, listens, handles reconnects."""
        while self._should_run:
            try:
                await self._connect_and_listen()
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Connection loop error: {e}", exc_info=True)
                if self._on_error:
                    self._on_error(f"Connection loop error: {e}")

            if not self._should_run:
                break

            # Reconnect with exponential backoff
            await self._backoff_reconnect()

    async def _connect_and_listen(self) -> None:
        """Establish connection and listen for messages."""
        try:
            self._ws = await asyncio.wait_for(
                websockets.connect(
                    WS_URL,
                    ping_interval=WS_PING_INTERVAL,
                    ping_timeout=WS_PING_TIMEOUT,
                    max_size=WS_RECV_BUFFER_SIZE,
                    write_limit=WS_SEND_BUFFER_SIZE,
                ),
                timeout=WS_CONNECT_TIMEOUT,
            )
        except asyncio.TimeoutError:
            logger.warning("WebSocket connection timed out")
            return
        except Exception as e:
            logger.warning(f"WebSocket connection failed: {e}")
            return

        # Connected
        self._reconnect_attempts = 0
        self._set_connected(True)

        # Authenticate
        if not await self._authenticate():
            logger.error("Authentication failed, disconnecting")
            await self._safe_close()
            self._set_connected(False)
            return

        # Resubscribe to previous tokens
        if self._subscribed_tokens:
            await self._resubscribe()

        # Start heartbeat
        self._heartbeat_task = asyncio.create_task(self._heartbeat_loop())

        # Signal ready
        self._ready.set()

        # Listen for messages
        await self._listen_loop()

    async def _authenticate(self) -> bool:
        """
        Send authentication request after connecting.
        Returns True if auth acknowledged.
        """
        if not self._ws:
            return False

        auth_payload = json.dumps({
            "action": "authenticate",
            "data": {
                "jwtToken": self.jwt_token,
                "clientcode": self.client_code,
                "feedToken": self.feed_token,
            },
        })

        try:
            await self._ws.send(auth_payload)
            # Wait for auth response
            response = await asyncio.wait_for(self._ws.recv(), timeout=5)
            resp_data = json.loads(response) if isinstance(response, str) else response
            if isinstance(resp_data, dict) and resp_data.get("status") == "success":
                logger.info("WebSocket authentication successful")
                return True
            else:
                logger.error(f"Auth response unexpected: {resp_data}")
                return False
        except asyncio.TimeoutError:
            logger.error("Auth response timeout")
            return False
        except Exception as e:
            logger.error(f"Auth error: {e}")
            return False

    async def _listen_loop(self) -> None:
        """Continuously receive messages from WebSocket."""
        try:
            async for message in self._ws:
                await self._route_message(message)
        except ConnectionClosed:
            logger.warning("WebSocket connection closed")
        except asyncio.CancelledError:
            pass
        except Exception as e:
            logger.error(f"Listen loop error: {e}")

        # Connection lost
        self._set_connected(False)
        self._ready.clear()

        if self._heartbeat_task:
            self._heartbeat_task.cancel()
            self._heartbeat_task = None

    # ──────────────────────────────────────────
    # MESSAGE ROUTING
    # ──────────────────────────────────────────

    async def _route_message(self, message: str | bytes) -> None:
        """
        Parse incoming message and route to appropriate callback.
        
        AngelOne SmartAPI WebSocket sends:
          - Auth/subscription responses as TEXT (JSON)
          - Tick/option/depth data as BINARY (msgpack)
        
        This method handles both formats transparently.
        """
        data: Optional[dict] = None

        # ── Try msgpack (binary) first — this is the tick data format ──
        if isinstance(message, bytes):
            try:
                data = msgpack.unpackb(message)
            except Exception:
                logger.warning(f"Failed to msgpack-decode binary message ({len(message)} bytes)")
                return

        # ── Try JSON (text) — auth responses, subscription acks, heartbeats ──
        elif isinstance(message, str):
            try:
                data = json.loads(message)
            except json.JSONDecodeError:
                logger.warning(f"Failed to JSON-parse text message: {message[:120]}")
                return

        if not isinstance(data, dict):
            return

        msg_type = data.get("type", "")

        # ── High-frequency market data (binary msgpack format) ──
        if msg_type == "tick" and self._on_tick:
            await self._safe_callback(self._on_tick, data)

        elif msg_type == "option" and self._on_option:
            await self._safe_callback(self._on_option, data)

        elif msg_type == "depth" and self._on_depth:
            await self._safe_callback(self._on_depth, data)

        # ── Control messages (JSON text format) ──
        elif msg_type == "heartbeat":
            # Heartbeat acknowledged — nothing to do
            logger.debug("Heartbeat ack received")
            pass

        elif msg_type == "subscription":
            logger.info(f"Subscription confirmed: {data.get('tokens', [])}")

        elif msg_type == "error":
            error_msg = data.get("message", "Unknown error")
            logger.error(f"Server error: {error_msg}")
            if self._on_error:
                await self._safe_callback(self._on_error, error_msg)

        else:
            logger.debug(f"Unhandled message type='{msg_type}', keys={list(data.keys())}")

    # ──────────────────────────────────────────
    # SUBSCRIPTION MANAGEMENT
    # ──────────────────────────────────────────

    async def _subscribe_tokens(self, tokens: List[str]) -> None:
        """Send subscription request for list of tokens."""
        if not self._ws or not self._connected:
            # Queue for later when connected
            self._subscribed_tokens.update(tokens)
            return

        payload = json.dumps({
            "action": "subscribe",
            "data": {
                "tokens": tokens,
                "feedToken": self.feed_token,
            },
        })

        try:
            await self._ws.send(payload)
            self._subscribed_tokens.update(tokens)
            logger.info(f"Subscribed to {len(tokens)} tokens")
        except Exception as e:
            logger.error(f"Subscribe failed: {e}")

    async def _resubscribe(self) -> None:
        """Resubscribe to all previously subscribed tokens after reconnect."""
        tokens = list(self._subscribed_tokens)
        if tokens:
            await self._subscribe_tokens(tokens)

    # ──────────────────────────────────────────
    # HEARTBEAT
    # ──────────────────────────────────────────

    async def _heartbeat_loop(self) -> None:
        """Send periodic heartbeat pings to keep connection alive."""
        try:
            while self._should_run and self._ws and self._connected:
                await asyncio.sleep(WS_HEARTBEAT_INTERVAL)

                try:
                    heartbeat_payload = json.dumps({
                        "action": "heartbeat",
                        "data": {"feedToken": self.feed_token},
                    })
                    await asyncio.wait_for(
                        self._ws.send(heartbeat_payload),
                        timeout=WS_HEARTBEAT_TIMEOUT,
                    )
                except asyncio.TimeoutError:
                    logger.warning("Heartbeat send timed out")
                except Exception as e:
                    logger.warning(f"Heartbeat error: {e}")
                    break

        except asyncio.CancelledError:
            pass

    # ──────────────────────────────────────────
    # RECONNECT LOGIC
    # ──────────────────────────────────────────

    async def _backoff_reconnect(self) -> None:
        """Exponential backoff before reconnecting."""
        self._reconnect_attempts += 1

        if self._reconnect_attempts > WS_MAX_RECONNECT_ATTEMPTS:
            logger.error(f"Max reconnect attempts ({WS_MAX_RECONNECT_ATTEMPTS}) reached")
            if self._on_error:
                self._on_error("Max reconnect attempts reached")
            self._should_run = False
            return

        delay = min(
            WS_RECONNECT_DELAY * (2 ** (self._reconnect_attempts - 1)),
            WS_MAX_RECONNECT_DELAY,
        )
        logger.info(
            f"Reconnecting in {delay}s "
            f"(attempt {self._reconnect_attempts}/{WS_MAX_RECONNECT_ATTEMPTS})"
        )

        try:
            await asyncio.sleep(delay)
        except asyncio.CancelledError:
            pass

    # ──────────────────────────────────────────
    # HELPERS
    # ──────────────────────────────────────────

    def _set_connected(self, value: bool) -> None:
        """Update connection state and notify observer."""
        if self._connected != value:
            self._connected = value
            if self._on_connection_change:
                asyncio.create_task(
                    self._safe_callback(self._on_connection_change, value)
                )

    async def _safe_callback(self, callback: Callable, *args, **kwargs) -> None:
        """Safely invoke a callback, catching exceptions."""
        try:
            if asyncio.iscoroutinefunction(callback):
                await callback(*args, **kwargs)
            else:
                callback(*args, **kwargs)
        except Exception as e:
            logger.error(f"Callback error: {e}")

    async def _safe_close(self) -> None:
        """Safely close the WebSocket connection."""
        if self._ws:
            try:
                await self._ws.close()
            except Exception:
                pass
            self._ws = None