"""Binary frame packing and unpacking for dashboard-backend transport.

Frame layout (36-byte header, network byte order):
    magic        : 4s   (b"JADB")
    version      : B    (uint8)
    kind         : B    (uint8)
    flags        : H    (uint16)
    seq          : Q    (uint64)
    timestamp_ns : Q    (uint64)
    payload_len  : I    (uint32)  # compressed length when compression is used
    crc32        : I    (uint32)  # CRC of UNCOMPRESSED payload bytes
    reserved     : I    (uint32)

This is the frozen dashboard/backend IPC contract for the current roadmap stage.
If the backend publisher changes this layout, it must bump FRAME_VERSION and the
contract tests must be updated in the same commit.
"""

from __future__ import annotations

import struct
import time
import zlib
from typing import Any, Dict

import msgpack

FRAME_MAGIC = b"JADB"
FRAME_VERSION = 1

KIND_HOT = 1
KIND_WARM = 2
KIND_COLD = 3

FLAG_COMPRESSED = 1 << 0

DEFAULT_COMPRESS_THRESHOLD = 65536

HEADER_STRUCT = struct.Struct("!4sBBHQQIII")
HEADER_SIZE = HEADER_STRUCT.size  # 36 bytes

# Pre-Week-9 stabilization: freeze the exact binary frame contract before
# backend publishers are added.  The roadmap prose had an earlier high-level
# field-width sketch; this dictionary is the executable source of truth for
# both dashboard decoder and future backend encoder.
BINARY_FRAME_CONTRACT_VERSION = "JADB/1"
BINARY_FRAME_CONTRACT: Dict[str, Any] = {
    "contract_version": BINARY_FRAME_CONTRACT_VERSION,
    "endianness": "network_big_endian",
    "header_struct": "!4sBBHQQIII",
    "header_size": HEADER_SIZE,
    "fields": [
        {"name": "magic", "format": "4s", "bytes": 4, "value": FRAME_MAGIC},
        {"name": "version", "format": "B", "bytes": 1, "value": FRAME_VERSION},
        {"name": "kind", "format": "B", "bytes": 1, "values": {"hot": KIND_HOT, "warm": KIND_WARM, "cold": KIND_COLD}},
        {"name": "flags", "format": "H", "bytes": 2, "values": {"compressed": FLAG_COMPRESSED}},
        {"name": "seq", "format": "Q", "bytes": 8, "semantics": "monotonic_per_kind"},
        {"name": "timestamp_ns", "format": "Q", "bytes": 8, "semantics": "producer_timestamp_ns"},
        {"name": "payload_len", "format": "I", "bytes": 4, "semantics": "wire_payload_bytes"},
        {"name": "crc32", "format": "I", "bytes": 4, "semantics": "uncompressed_payload_crc32"},
        {"name": "reserved", "format": "I", "bytes": 4, "value": 0},
    ],
    "payload_codec": "msgpack_use_bin_type_raw_false",
    "compression": "optional_zstd_when_available",
}


def get_binary_frame_contract() -> Dict[str, Any]:
    """Return a copy of the frozen IPC contract for tests/backend publishers."""
    fields = []
    for field in BINARY_FRAME_CONTRACT["fields"]:
        copied = dict(field)
        if isinstance(copied.get("values"), dict):
            copied["values"] = dict(copied["values"])
        fields.append(copied)
    out = dict(BINARY_FRAME_CONTRACT)
    out["fields"] = fields
    return out


def _load_zstd_adapter() -> Dict[str, Any]:
    """Load a zstd-compatible adapter.

    Supports either:
    - `zstd` module (direct APIs)
    - `zstandard` module (common package name)
    """
    try:
        import zstd as _zstd  # type: ignore

        def _compress(data: bytes) -> bytes:
            if hasattr(_zstd, "ZSTD_compress"):
                return _zstd.ZSTD_compress(data)
            if hasattr(_zstd, "compress"):
                return _zstd.compress(data)
            raise RuntimeError("zstd_compress_api_missing")

        def _decompress(data: bytes) -> bytes:
            if hasattr(_zstd, "ZSTD_decompress"):
                return _zstd.ZSTD_decompress(data)
            if hasattr(_zstd, "decompress"):
                return _zstd.decompress(data)
            raise RuntimeError("zstd_decompress_api_missing")

        return {"available": True, "compress": _compress, "decompress": _decompress}
    except Exception:
        pass

    try:
        import zstandard as _zstd_std  # type: ignore

        compressor = _zstd_std.ZstdCompressor()
        decompressor = _zstd_std.ZstdDecompressor()

        def _compress(data: bytes) -> bytes:
            return compressor.compress(data)

        def _decompress(data: bytes) -> bytes:
            return decompressor.decompress(data)

        return {"available": True, "compress": _compress, "decompress": _decompress}
    except Exception:
        return {"available": False, "compress": None, "decompress": None}


_ZSTD = _load_zstd_adapter()


def pack_frame(
    payload: dict,
    kind: int,
    seq: int,
    timestamp_ns: int | None = None,
    compress_threshold: int = DEFAULT_COMPRESS_THRESHOLD,
    max_payload_bytes: int = 10 * 1024 * 1024,
) -> bytes:
    """Pack a dictionary into a binary frame.

    Thread-safe: pure functions with no shared state.

    Args:
        payload: Dictionary payload to serialize.
        kind: Frame kind (`KIND_HOT`, `KIND_WARM`, `KIND_COLD`).
        seq: Monotonic sequence number.
        timestamp_ns: Optional nanosecond timestamp; defaults to `time.time_ns()`.
        compress_threshold: Minimum payload size for optional zstd compression.

    Returns:
        Encoded frame bytes.
    """
    if timestamp_ns is None:
        timestamp_ns = time.time_ns()

    if kind not in (KIND_HOT, KIND_WARM, KIND_COLD):
        raise ValueError(f"invalid_kind: {kind}")

    raw_payload = msgpack.packb(payload, use_bin_type=True)
    if len(raw_payload) > max_payload_bytes:
        raise ValueError(f"payload_too_large: {len(raw_payload)} bytes")

    crc32 = zlib.crc32(raw_payload) & 0xFFFFFFFF

    flags = 0
    wire_payload = raw_payload

    if _ZSTD["available"] and len(raw_payload) >= compress_threshold:
        wire_payload = _ZSTD["compress"](raw_payload)
        flags |= FLAG_COMPRESSED

    header = HEADER_STRUCT.pack(
        FRAME_MAGIC,
        FRAME_VERSION,
        int(kind),
        flags,
        int(seq),
        int(timestamp_ns),
        len(wire_payload),
        crc32,
        0,
    )
    return header + wire_payload


def unpack_frame(
    data: bytes,
    last_seq: int = -1,
    max_payload_bytes: int = 10 * 1024 * 1024,
) -> dict:
    """Unpack a binary frame.

    Thread-safe: pure functions with no shared state.

    Returns a dict with keys:
        valid, error, kind, seq, timestamp_ns, payload
    """
    invalid = {
        "valid": False,
        "error": "",
        "kind": None,
        "seq": None,
        "timestamp_ns": None,
        "payload": None,
    }

    if not isinstance(data, (bytes, bytearray)) or len(data) < HEADER_SIZE:
        invalid["error"] = "invalid_frame"
        return invalid

    try:
        magic, version, kind, flags, seq, timestamp_ns, payload_len, checksum, _reserved = HEADER_STRUCT.unpack(
            data[:HEADER_SIZE]
        )
    except struct.error:
        invalid["error"] = "invalid_frame"
        return invalid

    invalid["kind"] = kind
    invalid["seq"] = seq
    invalid["timestamp_ns"] = timestamp_ns

    if magic != FRAME_MAGIC:
        invalid["error"] = "invalid_magic"
        return invalid

    if version > FRAME_VERSION:
        invalid["error"] = "unsupported_version"
        return invalid

    if seq <= last_seq:
        invalid["error"] = "stale_sequence"
        return invalid

    if payload_len > max_payload_bytes:
        invalid["error"] = "payload_too_large"
        return invalid

    if len(data) != HEADER_SIZE + payload_len:
        invalid["error"] = "invalid_frame"
        return invalid

    payload_bytes = bytes(data[HEADER_SIZE : HEADER_SIZE + payload_len])

    if flags & FLAG_COMPRESSED:
        if not _ZSTD["available"]:
            invalid["error"] = "decompress_failed"
            return invalid
        try:
            payload_bytes = _ZSTD["decompress"](payload_bytes)
        except Exception:
            invalid["error"] = "decompress_failed"
            return invalid
        if len(payload_bytes) > max_payload_bytes:
            invalid["error"] = "payload_too_large"
            return invalid

    computed_crc = zlib.crc32(payload_bytes) & 0xFFFFFFFF
    if computed_crc != checksum:
        invalid["error"] = "checksum_mismatch"
        return invalid

    try:
        payload = msgpack.unpackb(payload_bytes, raw=False, strict_map_key=False) if payload_bytes else {}
    except Exception:
        invalid["error"] = "invalid_payload"
        return invalid

    if not isinstance(payload, dict):
        invalid["error"] = "invalid_payload"
        return invalid

    return {
        "valid": True,
        "error": "",
        "kind": kind,
        "seq": seq,
        "timestamp_ns": timestamp_ns,
        "payload": payload,
    }


if __name__ == "__main__":
    # Round-trip test
    sample = {"a": 1, "b": "ok", "nested": {"x": True}}
    blob = pack_frame(sample, kind=KIND_HOT, seq=1)
    out = unpack_frame(blob)
    assert out["valid"] is True
    assert out["payload"] == sample

    # Empty payload round-trip (heartbeat frame)
    empty_blob = pack_frame({}, kind=KIND_HOT, seq=99)
    out_empty = unpack_frame(empty_blob)
    assert out_empty["valid"] is True, f"Empty payload failed: {out_empty['error']}"
    assert out_empty["payload"] == {}

    # CRC corruption test
    tampered = bytearray(blob)
    tampered[HEADER_SIZE] ^= 0x01
    out_crc = unpack_frame(bytes(tampered))
    assert out_crc["valid"] is False
    assert out_crc["error"] == "checksum_mismatch"

    # Version mismatch test
    bad_version = bytearray(blob)
    bad_version[4] = FRAME_VERSION + 1
    out_ver = unpack_frame(bytes(bad_version))
    assert out_ver["valid"] is False
    assert out_ver["error"] == "unsupported_version"

    # Sequence stale test
    stale_blob = pack_frame({"ok": True}, kind=KIND_WARM, seq=5)
    out_stale = unpack_frame(stale_blob, last_seq=5)
    assert out_stale["valid"] is False
    assert out_stale["error"] == "stale_sequence"

    # Compression test (if zstd available)
    if _ZSTD["available"]:
        large_payload = {"data": "x" * 100_000}
        large_blob = pack_frame(large_payload, kind=KIND_COLD, seq=9, compress_threshold=1024)
        header = HEADER_STRUCT.unpack(large_blob[:HEADER_SIZE])
        _flags = header[3]
        assert (_flags & FLAG_COMPRESSED) == FLAG_COMPRESSED
        out_large = unpack_frame(large_blob)
        assert out_large["valid"] is True
        assert out_large["payload"] == large_payload

    print("binary_frame self-test passed")