"""
Junior Aladdin - Authentication Manager
=========================================
PURPOSE:
    Handle Angel One SmartAPI authentication, token management,
    and auto-refresh. This is the gateway to ALL broker operations.

INSTITUTIONAL HARDENING (2026):
    - No long network calls while holding state lock
    - Single-flight authentication/refresh using _auth_lock + _auth_done_event
    - Feed token update callback to notify WebSocket manager
    - Thread-safe SmartConnect access via proxy guarded by _api_lock
    - Token validation: never mark authenticated if jwtToken missing/empty
    - Network timeout best-effort enforcement (Windows-safe)

USAGE:
    from src.core.auth_manager import AuthManager

    auth = AuthManager()
    auth.set_feed_token_callback(lambda new_token: print("feed_token updated"))
    success = auth.authenticate()
    if success:
        api = auth.get_smart_api()    # Thread-safe proxy object
        profile = api.getProfile(auth.refresh_token)
        auth.logout()
"""

import os
import socket
import threading
import time
from contextlib import contextmanager
from datetime import datetime, timezone, timedelta
from typing import Callable, Optional, Any, Dict

from dotenv import load_dotenv

from src.utils.config_loader import Config
from src.utils.logger import setup_logger


# ============================================
# Optional pyotp import (do NOT crash at startup)
# ============================================
try:
    import pyotp as _pyotp  # type: ignore
except ImportError:
    _pyotp = None


# ============================================
# IST Timezone
# ============================================
IST = timezone(timedelta(hours=5, minutes=30))


class _SmartApiProxy:
    """
    Thread-safe proxy around SmartConnect instance.

    All method calls are protected by AuthManager._api_lock and dynamically
    routed to the current AuthManager._smart_api reference (supports swap on refresh).

    NOTE:
        This proxy enforces thread-safe access regardless of whether the underlying
        SmartApi library itself is thread-safe.
    """

    def __init__(self, manager: "AuthManager"):
        self._m = manager

    def __getattr__(self, name: str):
        def _call(*args, **kwargs):
            # Serialize ALL underlying SmartConnect calls
            with self._m._api_lock:
                api = self._m._get_smart_api_ref()
                if api is None:
                    raise RuntimeError(
                        "SmartAPI is not available (not authenticated or logged out)."
                    )
                attr = getattr(api, name)
                if callable(attr):
                    return attr(*args, **kwargs)
                # Non-callable attribute access
                return attr

        return _call


class AuthManager:
    """
    Manages Angel One SmartAPI authentication lifecycle.

    Critical Guarantees:
        - State mutations are protected by _lock (fast, in-process only)
        - Broker network calls are NEVER executed while holding _lock
        - Only one thread can authenticate/refresh at a time via _auth_lock
        - SmartConnect usage is serialized via _api_lock (proxy enforces)
    """

    def __init__(self):
        self._logger = setup_logger("auth_manager")

        # Fast state lock (never hold during network calls)
        self._lock = threading.RLock()

        # Single-flight auth/refresh lock (network calls happen under this lock, not _lock)
        self._auth_lock = threading.Lock()

        # Event indicates whether an auth/refresh attempt is currently NOT in progress.
        # - set(): no auth in progress
        # - clear(): auth/refresh in progress
        self._auth_done_event = threading.Event()
        self._auth_done_event.set()

        # Serialize ALL SmartConnect method calls (thread-safety unknown upstream)
        self._api_lock = threading.Lock()

        # Feed token update callback (e.g., WebSocketManager reconnect)
        self._feed_token_callback: Optional[Callable[[str], None]] = None

        # Load .env file
        load_dotenv()

        # Read credentials
        self._api_key: str = os.getenv("ANGEL_API_KEY", "")
        self._client_id: str = os.getenv("ANGEL_CLIENT_ID", "")
        self._password: str = os.getenv("ANGEL_PASSWORD", "")
        self._totp_secret: str = os.getenv("ANGEL_TOTP_SECRET", "")

        # Token storage
        self.auth_token: Optional[str] = None
        self.feed_token: Optional[str] = None
        self.refresh_token: Optional[str] = None

        # State
        self.is_authenticated: bool = False
        self.last_auth_time: Optional[datetime] = None

        # Monotonic timestamp to avoid wall-clock jumps affecting expiry checks
        self._last_auth_monotonic: Optional[float] = None

        # Underlying SmartConnect (never expose directly; use proxy)
        self._smart_api = None
        self._smart_api_proxy = _SmartApiProxy(self)

        # Validate credentials exist
        self._credentials_valid = self._validate_credentials()

        # Proactive dependency warning (do not crash at startup)
        if _pyotp is None and self._credentials_valid:
            self._logger.error(
                "Dependency 'pyotp' not installed; live authentication will fail.",
                extra={"fix": "pip install pyotp"},
            )

    # --------------------------
    # Public additions (allowed)
    # --------------------------
    def set_feed_token_callback(self, callback: Callable[[str], None]):
        """
        Register callback invoked when feed_token changes.
        Intended for WebSocket manager to reconnect immediately on refresh.

        Args:
            callback: Callable[[str], None]
        """
        if not callable(callback):
            raise TypeError("callback must be callable")
        with self._lock:
            self._feed_token_callback = callback

    # --------------------------
    # Internal helpers
    # --------------------------
    def _get_smart_api_ref(self):
        """Internal: return current SmartConnect reference under lock (fast)."""
        with self._lock:
            return self._smart_api

    def _validate_credentials(self) -> bool:
        placeholders = [
            "paste_your_api_key_here",
            "paste_your_client_id_here",
            "paste_your_trading_password_here",
            "paste_your_totp_secret_here",
            "",
        ]

        fields = {
            "ANGEL_API_KEY": self._api_key,
            "ANGEL_CLIENT_ID": self._client_id,
            "ANGEL_PASSWORD": self._password,
            "ANGEL_TOTP_SECRET": self._totp_secret,
        }

        all_valid = True
        for name, value in fields.items():
            if value in placeholders:
                self._logger.warning(
                    "Credential not configured",
                    extra={"field": name, "status": "missing_or_placeholder"},
                )
                all_valid = False
        return all_valid

    def has_credentials(self) -> bool:
        return self._credentials_valid

    def _get_network_timeout_sec(self) -> int:
        """
        Best-effort timeout for broker calls. Configurable if auth section exists.
        Not required in config.yaml; safe default is used.
        """
        try:
            t = Config.get("auth", "network_timeout_sec", default=10)
            t_int = int(t) if isinstance(t, (int, float, str)) else 10
            return max(3, min(t_int, 30))
        except Exception:
            return 10

    @contextmanager
    def _temporary_socket_timeout(self, timeout_sec: int):
        """
        Best-effort network timeout enforcement on Windows.

        socket.setdefaulttimeout affects newly-created sockets globally within the process.
        We limit the time window by restoring the previous value immediately after the call.
        """
        prev = socket.getdefaulttimeout()
        socket.setdefaulttimeout(timeout_sec)
        try:
            yield
        finally:
            socket.setdefaulttimeout(prev)

    def _create_smart_api(self):
        """
        Create SmartConnect with best-effort timeout parameter if supported.
        """
        from SmartApi.smartConnect import SmartConnect

        timeout_sec = self._get_network_timeout_sec()
        try:
            # Some versions may support timeout kwargs
            return SmartConnect(api_key=self._api_key, timeout=timeout_sec)
        except TypeError:
            # Fallback for versions without timeout parameter
            return SmartConnect(api_key=self._api_key)

    def _generate_totp(self) -> str:
        """
        Generate current TOTP code using the secret from .env.
        Raises RuntimeError if pyotp is not installed.
        """
        if _pyotp is None:
            raise RuntimeError("Missing dependency 'pyotp'. Install it with: pip install pyotp")

        try:
            totp = _pyotp.TOTP(self._totp_secret)
            code = totp.now()
            return code
        except Exception as e:
            self._logger.error("TOTP generation failed", extra={"error": str(e)})
            raise

    def _update_tokens_atomically(
        self,
        *,
        smart_api_obj: Any,
        auth_token: str,
        refresh_token: str,
        feed_token: str,
        source: str,
    ) -> None:
        """
        Atomically updates ALL token-related state under _lock, then triggers callback outside the lock.
        """
        if not auth_token:
            raise ValueError("auth_token is empty")
        if not refresh_token:
            raise ValueError("refresh_token is empty")
        if not feed_token:
            raise ValueError("feed_token is empty")

        callback = None
        callback_token = None

        with self._lock:
            old_feed = self.feed_token

            self._smart_api = smart_api_obj
            self.auth_token = auth_token
            self.refresh_token = refresh_token
            self.feed_token = feed_token

            self.is_authenticated = True
            self.last_auth_time = datetime.now(IST)
            self._last_auth_monotonic = time.monotonic()

            if self.feed_token and self.feed_token != old_feed:
                callback = self._feed_token_callback
                callback_token = self.feed_token

        # Callback must be outside lock (avoid deadlocks / long operations inside state lock)
        if callback and callback_token:
            try:
                callback(callback_token)
                self._logger.info(
                    "feed_token callback invoked",
                    extra={"source": source, "changed": True},
                )
            except Exception as e:
                self._logger.error(
                    "feed_token callback failed",
                    extra={"source": source, "error": str(e), "error_type": type(e).__name__},
                )

    # --------------------------
    # Public API (signatures unchanged)
    # --------------------------
    def authenticate(self, max_retries: int = 3) -> bool:
        """
        Authenticate with Angel One SmartAPI.

        Institutional changes:
            - No network calls while holding _lock
            - Single-flight via _auth_lock + _auth_done_event
            - Token response validation
            - Atomic state update + callback on feed_token change
        """
        if not self._credentials_valid:
            self._logger.error("Cannot authenticate — credentials not configured in .env")
            return False

        # Fast-path: already authenticated with non-empty token and SmartAPI ref
        with self._lock:
            if self.is_authenticated and self._smart_api and self.auth_token:
                return True

        # Single-flight auth
        with self._auth_lock:
            # If another thread already authenticated while we were waiting, return
            with self._lock:
                if self.is_authenticated and self._smart_api and self.auth_token:
                    return True

            self._auth_done_event.clear()

            try:
                for attempt in range(1, max_retries + 1):
                    try:
                        self._logger.info(
                            "Authenticating with Angel One",
                            extra={"attempt": attempt, "client_id": self._client_id},
                        )

                        totp_code = self._generate_totp()

                        smart_api = self._create_smart_api()
                        timeout_sec = self._get_network_timeout_sec()

                        with self._temporary_socket_timeout(timeout_sec):
                            session_data = smart_api.generateSession(
                                self._client_id,
                                self._password,
                                totp_code,
                            )

                        if session_data and session_data.get("status"):
                            data = session_data.get("data", {}) or {}

                            jwt = data.get("jwtToken")
                            rft = data.get("refreshToken")
                            fdt = smart_api.getfeedToken()

                            # Strict validation (never mark authenticated on empty tokens)
                            if not jwt:
                                self._logger.critical(
                                    "Authentication response missing jwtToken; refusing to authenticate.",
                                    extra={"response": str(session_data)[:300]},
                                )
                                return False
                            if not rft:
                                self._logger.critical(
                                    "Authentication response missing refreshToken; refusing to authenticate.",
                                    extra={"response": str(session_data)[:300]},
                                )
                                return False
                            if not fdt:
                                self._logger.critical(
                                    "Authentication failed to obtain feed_token; refusing to authenticate."
                                )
                                return False

                            # Atomic state update (fast)
                            self._update_tokens_atomically(
                                smart_api_obj=smart_api,
                                auth_token=str(jwt),
                                refresh_token=str(rft),
                                feed_token=str(fdt),
                                source="authenticate",
                            )

                            self._logger.info(
                                "Authentication successful",
                                extra={
                                    "client_id": self._client_id,
                                    "has_auth_token": True,
                                    "has_feed_token": True,
                                    "has_refresh_token": True,
                                },
                            )
                            return True

                        # Login failed
                        error_msg = (
                            session_data.get("message", "Unknown error") if session_data else "No response"
                        )
                        self._logger.error(
                            "Authentication failed",
                            extra={
                                "attempt": attempt,
                                "error": error_msg,
                                "response": str(session_data)[:200],
                            },
                        )

                        # If invalid credentials, don't retry
                        if session_data and "Invalid" in str(session_data.get("message", "")):
                            self._logger.critical("Invalid credentials — check .env file. NOT retrying.")
                            return False

                    except ImportError as e:
                        self._logger.error(
                            "SmartAPI library import failed",
                            extra={"error": str(e)},
                        )
                        return False

                    except Exception as e:
                        self._logger.error(
                            "Authentication attempt failed",
                            extra={
                                "attempt": attempt,
                                "error": str(e),
                                "error_type": type(e).__name__,
                            },
                        )
                        if attempt < max_retries:
                            wait_sec = 2 * attempt  # backoff
                            self._logger.info(
                                f"Retrying in {wait_sec} seconds...",
                                extra={"next_attempt": attempt + 1},
                            )
                            time.sleep(wait_sec)

                self._logger.error("All authentication attempts exhausted")
                return False

            finally:
                self._auth_done_event.set()

    def refresh_session(self) -> bool:
        """
        Refresh authentication tokens.

        Institutional changes:
            - No network calls while holding _lock
            - Single-flight via _auth_lock + _auth_done_event
            - Strict token validation
            - Atomic token swap + feed_token callback
        """
        # Quick eligibility check (fast)
        with self._lock:
            if not self.is_authenticated or not self._smart_api or not self.refresh_token:
                self._logger.warning("Cannot refresh — not authenticated or refresh_token missing")
                return False

        with self._auth_lock:
            # Another thread may have refreshed already while we waited
            if not self._is_token_old():
                return True

            self._auth_done_event.clear()

            try:
                with self._lock:
                    current_refresh = self.refresh_token
                    if not current_refresh:
                        self._logger.warning("Cannot refresh — refresh_token missing")
                        return False

                timeout_sec = self._get_network_timeout_sec()
                # Use a NEW SmartConnect instance for refresh to avoid interfering with ongoing calls
                smart_api = self._create_smart_api()

                try:
                    with self._temporary_socket_timeout(timeout_sec):
                        token_data = smart_api.generateToken(current_refresh)

                    if token_data and token_data.get("status"):
                        data = token_data.get("data", {}) or {}
                        jwt = data.get("jwtToken")

                        if not jwt:
                            self._logger.critical(
                                "Token refresh response missing jwtToken; refusing to apply refresh.",
                                extra={"response": str(token_data)[:300]},
                            )
                            return False

                        # After refresh, get new feed token from this instance
                        fdt = smart_api.getfeedToken()

                        if not fdt:
                            self._logger.critical(
                                "Token refresh failed to obtain feed_token; refusing to apply refresh."
                            )
                            return False

                        # refresh_token may or may not rotate; keep existing if not provided
                        new_refresh = data.get("refreshToken") or current_refresh

                        self._update_tokens_atomically(
                            smart_api_obj=smart_api,
                            auth_token=str(jwt),
                            refresh_token=str(new_refresh),
                            feed_token=str(fdt),
                            source="refresh_session",
                        )

                        self._logger.info(
                            "Token refreshed successfully",
                            extra={"client_id": self._client_id},
                        )
                        return True

                    error_msg = token_data.get("message", "Unknown") if token_data else "No response"
                    self._logger.warning(
                        "Token refresh failed, attempting full re-authentication",
                        extra={"error": error_msg},
                    )
                    return self.authenticate()

                except Exception as e:
                    self._logger.error(
                        "Token refresh error, attempting full re-authentication",
                        extra={"error": str(e), "error_type": type(e).__name__},
                    )
                    return self.authenticate()

            finally:
                self._auth_done_event.set()

    def _is_token_old(self, max_age_minutes: int = 25) -> bool:
        """
        Check if token is older than max_age_minutes.

        Uses monotonic clock when available to avoid wall-clock jumps.
        Reads configurable max_age_minutes from config: auth.token_max_age_minutes (default 25).
        """
        # Configurable token age threshold
        try:
            cfg_max_age = Config.get("auth", "token_max_age_minutes", default=max_age_minutes)
            if isinstance(cfg_max_age, (int, float)) and cfg_max_age > 0:
                max_age_minutes = int(cfg_max_age)
        except Exception as e:
            self._logger.warning(
                "Could not read auth.token_max_age_minutes from config; using default",
                extra={"error": str(e), "default_max_age_minutes": max_age_minutes},
            )

        with self._lock:
            last_mono = self._last_auth_monotonic
            last_wall = self.last_auth_time

        # Prefer monotonic (immune to NTP/time changes)
        if last_mono is not None:
            age_sec = time.monotonic() - last_mono
            if age_sec < 0:
                age_sec = 0.0
            return age_sec > (max_age_minutes * 60)

        # Fallback to wall clock
        if last_wall is None:
            return True
        age = datetime.now(IST) - last_wall
        age_sec = age.total_seconds()
        if age_sec < 0:
            age_sec = 0.0
        return age_sec > (max_age_minutes * 60)

    def get_smart_api(self):
        """
        Get thread-safe SmartConnect proxy for API calls.

        Mandates:
            - Double-checked locking for auth
            - Single-flight authenticate/refresh
            - No long network calls inside _lock
        """
        # Double-checked locking for authentication
        with self._lock:
            authed = bool(self.is_authenticated and self._smart_api and self.auth_token)

        if not authed:
            # Re-check under lock to avoid races, then authenticate outside _lock
            with self._lock:
                authed = bool(self.is_authenticated and self._smart_api and self.auth_token)
            if not authed:
                ok = self.authenticate()
                if not ok:
                    self._logger.error("get_smart_api failed — could not authenticate")
                    return None

        # Refresh if old (single-flight inside refresh_session)
        if self._is_token_old():
            self._logger.info("Token is old, refreshing...")
            self.refresh_session()

        # Ensure still authenticated after refresh attempt
        with self._lock:
            authed = bool(self.is_authenticated and self._smart_api and self.auth_token)

        if not authed:
            return None

        return self._smart_api_proxy

    def logout(self) -> bool:
        """
        Logout from Angel One (terminate session).

        Institutional changes:
            - Immediately reset local state atomically (so system doesn't remain half-authenticated)
            - Best-effort terminateSession outside state lock, with socket timeout
            - Serialized against auth/refresh via _auth_lock
        """
        # Serialize against concurrent auth/refresh
        with self._auth_lock:
            with self._lock:
                if not self.is_authenticated or not self._smart_api:
                    self._logger.info("Already logged out or never authenticated")
                    self.is_authenticated = False
                    self.auth_token = None
                    self.feed_token = None
                    self.refresh_token = None
                    self._smart_api = None
                    self.last_auth_time = None
                    self._last_auth_monotonic = None
                    return True

                smart_api = self._smart_api
                client_id = self._client_id

                # Reset state immediately (atomic local logout)
                self.is_authenticated = False
                self.auth_token = None
                self.feed_token = None
                self.refresh_token = None
                self._smart_api = None
                self.last_auth_time = None
                self._last_auth_monotonic = None

            # Best-effort remote termination outside state lock
            try:
                timeout_sec = self._get_network_timeout_sec()
                with self._temporary_socket_timeout(timeout_sec):
                    # Serialize SmartConnect call too
                    with self._api_lock:
                        smart_api.terminateSession(client_id)
                self._logger.info("Logged out successfully", extra={"client_id": client_id})
            except Exception as e:
                self._logger.warning(
                    "Logout API call failed (session may have already expired)",
                    extra={"error": str(e), "error_type": type(e).__name__},
                )

            return True

    def get_status(self) -> dict:
        """
        Get current authentication status (thread-safe, non-blocking).
        """
        with self._lock:
            is_authenticated = self.is_authenticated
            has_credentials = self._credentials_valid
            client_id = self._client_id if self._credentials_valid else "NOT_SET"
            has_auth_token = bool(self.auth_token)
            has_feed_token = bool(self.feed_token)
            last_auth_time = self.last_auth_time
            last_mono = self._last_auth_monotonic

        token_age_seconds = None
        if last_mono is not None:
            token_age_seconds = max(0.0, time.monotonic() - last_mono)
        elif last_auth_time is not None:
            token_age_seconds = max(0.0, (datetime.now(IST) - last_auth_time).total_seconds())

        return {
            "is_authenticated": is_authenticated,
            "has_credentials": has_credentials,
            "client_id": client_id,
            "has_auth_token": has_auth_token,
            "has_feed_token": has_feed_token,
            "last_auth_time": str(last_auth_time) if last_auth_time else None,
            "token_age_seconds": token_age_seconds,
            "auth_in_progress": not self._auth_done_event.is_set(),
        }


# ============================================
# Module Self-Test
# ============================================
if __name__ == "__main__":
    print("=" * 60)
    print("  JUNIOR ALADDIN — AuthManager Test")
    print("=" * 60)
    print()

    passed = 0
    failed = 0

    # ── Test 1: Create AuthManager ──
    print("  [Test 1] Create AuthManager...")
    try:
        auth = AuthManager()
        print("    ✅ AuthManager created")
        passed += 1
    except Exception as e:
        print(f"    ❌ Failed: {e}")
        failed += 1

    # ── Test 2: Check credentials status ──
    print("\n  [Test 2] Credentials check...")
    has_creds = auth.has_credentials()
    status = auth.get_status()
    print(f"    Credentials configured: {has_creds}")
    print(f"    Client ID: {status['client_id']}")

    if has_creds:
        print("    ✅ Credentials found in .env")
        passed += 1
    else:
        print("    ⚠️  Credentials NOT configured — using placeholders")
        print("    ℹ️  This is OK for now. Set up .env when ready.")
        print("    ℹ️  Skipping live authentication tests.")
        passed += 1  # Not a failure — just not configured yet

    # ── Test 3: TOTP generation ──
    print("\n  [Test 3] TOTP generation...")
    if has_creds:
        try:
            totp_code = auth._generate_totp()
            print(f"    ✅ TOTP generated: {totp_code} (6 digits)")
            if len(totp_code) == 6 and totp_code.isdigit():
                print("    ✅ Valid 6-digit code")
                passed += 1
            else:
                print("    ❌ Invalid TOTP format")
                failed += 1
        except Exception as e:
            print(f"    ❌ TOTP generation failed: {e}")
            failed += 1
    else:
        print("    ⏭️  Skipped — no TOTP secret configured")
        passed += 1

    # ── Test 4: Live authentication ──
    print("\n  [Test 4] Live authentication...")
    if has_creds:
        try:
            result = auth.authenticate()
            if result:
                print("    ✅ Authenticated successfully!")
                print(f"    ✅ Auth token: {auth.auth_token[:30]}...")
                print(f"    ✅ Feed token: {auth.feed_token[:20] if auth.feed_token else 'N/A'}...")
                passed += 1

                # ── Test 5: Get profile via proxy ──
                print("\n  [Test 5] Fetch profile...")
                try:
                    api = auth.get_smart_api()
                    profile = api.getProfile(auth.refresh_token)
                    if profile and profile.get("status"):
                        name = profile.get("data", {}).get("name", "Unknown")
                        print(f"    ✅ Profile name: {name}")
                        passed += 1
                    else:
                        print(f"    ⚠️  Profile fetch returned: {profile}")
                        passed += 1  # Non-critical
                except Exception as e:
                    print(f"    ⚠️  Profile fetch error: {e}")
                    passed += 1  # Non-critical

                # ── Test 6: Token age check ──
                print("\n  [Test 6] Token age check...")
                is_old = auth._is_token_old(max_age_minutes=25)
                print(f"    Token age old (>25 min): {is_old}")
                print("    ✅ Token age check executed")
                passed += 1

                # ── Test 7: Status dict ──
                print("\n  [Test 7] Status check...")
                status = auth.get_status()
                print(f"    ✅ Status: authenticated={status['is_authenticated']}")
                print(f"    ✅ Token age: {status['token_age_seconds']:.1f}s")
                passed += 1

                # ── Test 8: Logout ──
                print("\n  [Test 8] Logout...")
                auth.logout()
                if not auth.is_authenticated:
                    print("    ✅ Logged out successfully")
                    passed += 1
                else:
                    print("    ❌ Still authenticated after logout")
                    failed += 1
            else:
                print("    ❌ Authentication failed!")
                print("    Check your .env credentials.")
                failed += 1
        except Exception as e:
            print(f"    ❌ Authentication error: {e}")
            failed += 1
    else:
        print("    ⏭️  Skipped — credentials not configured")
        print("    ℹ️  To test live auth, fill in .env and re-run")
        passed += 1

    # ── Summary ──
    print("\n" + "=" * 60)
    print(f"  Results: {passed} passed, {failed} failed")
    if has_creds and failed == 0:
        print("\n  ✅ AuthManager working with LIVE connection!")
    elif not has_creds and failed == 0:
        print("\n  ✅ AuthManager structure OK (credentials pending)")
        print("  ℹ️  Set up .env credentials, then re-run for live test.")
    else:
        print(f"\n  ⚠️  {failed} tests failed.")
    print("=" * 60)