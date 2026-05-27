# Junior Aladdin Dashboard Binary Frame Contract — JADB/1

This document freezes the dashboard/backend IPC frame header used by
`dashboard/core/binary_frame.py` before the backend snapshot publisher is added.

## Contract

- Magic: `b"JADB"`
- Frame version: `1`
- Endianness: network big-endian
- Python struct: `!4sBBHQQIII`
- Header size: `36` bytes

| Field | Struct | Bytes | Semantics |
|---|---:|---:|---|
| `magic` | `4s` | 4 | Constant `b"JADB"` |
| `version` | `B` | 1 | Frame version, currently `1` |
| `kind` | `B` | 1 | `1=HOT`, `2=WARM`, `3=COLD` |
| `flags` | `H` | 2 | Bit flags, bit 0 = zstd-compressed payload |
| `seq` | `Q` | 8 | Monotonic sequence per frame kind |
| `timestamp_ns` | `Q` | 8 | Backend producer timestamp in nanoseconds |
| `payload_len` | `I` | 4 | Wire payload length in bytes |
| `crc32` | `I` | 4 | CRC32 of the uncompressed msgpack payload |
| `reserved` | `I` | 4 | Reserved, must be `0` |

## Payload

- Codec: msgpack
- Packing: `use_bin_type=True`
- Unpacking: `raw=False`, `strict_map_key=False`
- Payload must decode to a dictionary.
- Optional zstd compression may be used when available and the payload exceeds
  `DEFAULT_COMPRESS_THRESHOLD`.

## Change rule

Any backend/dashboard publisher or decoder that changes this layout must:

1. bump `FRAME_VERSION`,
2. update `BINARY_FRAME_CONTRACT` in `binary_frame.py`,
3. update this document,
4. update the contract tests in the same commit.

Do not implement backend snapshot transport against a different header shape.
