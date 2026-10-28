"""Small application protocol for splitting JPEG frames across UDP packets.

Each UDP datagram has this layout::

    0                 4       8      10     12       16
    +-----------------+-------+------+------+--------+------------------+
    | magic (4 bytes) | frame | index| count| total  | JPEG slice       |
    |                 | id    |      |      | size   |                  |
    +-----------------+-------+------+------+--------+------------------+

The fixed header is encoded with ``!4sIHHI`` (network byte order) and is
therefore 16 bytes:

* ``magic`` identifies this protocol and rejects unrelated UDP traffic.
* ``frame_id`` identifies one camera frame and wraps at 2**32.
* ``index`` is the zero-based fragment number within that frame.
* ``count`` is the total number of fragments in that frame.
* ``total_size`` is the complete JPEG byte length, used after reassembly.

The rest of the datagram contains a consecutive slice of the JPEG bytes.
UDP can reorder or drop datagrams, so the receiver stores slices by ``index``
and displays a frame only after every index is present and the final length
matches ``total_size``. There is intentionally no retransmission here: an
incomplete frame expires at the server and the next camera frame is used.

The default client chunk size is 1200 bytes. Adding the 16-byte header keeps
the usual packet below common Ethernet MTUs and avoids relying on IP-level
fragmentation. Larger chunks can improve efficiency on a controlled network,
but increase the chance that one lost UDP packet invalidates a whole frame.
"""

from __future__ import annotations

import struct
from collections.abc import Iterator
from dataclasses import dataclass


# Header fields: magic, frame_id, fragment_index, fragment_count, total JPEG
# size. ``!`` means network byte order with standard (non-native) sizes.
MAGIC = b"JPG1"
HEADER = struct.Struct("!4sIHHI")
HEADER_SIZE = HEADER.size
MAX_UDP_PAYLOAD = 65507
MAX_CHUNKS = 65535


@dataclass(frozen=True)
class Fragment:
    frame_id: int
    index: int
    count: int
    total_size: int
    payload: bytes


def packetize(jpeg: bytes, frame_id: int, chunk_size: int) -> Iterator[bytes]:
    """Yield UDP datagrams for one JPEG frame."""

    if not jpeg:
        raise ValueError("JPEG payload cannot be empty")
    if not 1 <= chunk_size <= MAX_UDP_PAYLOAD - HEADER_SIZE:
        raise ValueError(
            f"chunk_size must be between 1 and {MAX_UDP_PAYLOAD - HEADER_SIZE}"
        )

    count = (len(jpeg) + chunk_size - 1) // chunk_size
    if count > MAX_CHUNKS:
        raise ValueError("JPEG payload requires too many UDP fragments")

    for index in range(count):
        start = index * chunk_size
        payload = jpeg[start : start + chunk_size]
        yield HEADER.pack(MAGIC, frame_id & 0xFFFFFFFF, index, count, len(jpeg)) + payload


def parse_packet(datagram: bytes) -> Fragment | None:
    """Parse and validate a UDP fragment header."""

    if len(datagram) < HEADER_SIZE:
        return None

    magic, frame_id, index, count, total_size = HEADER.unpack_from(datagram)
    if magic != MAGIC or count == 0 or index >= count or total_size == 0:
        return None

    return Fragment(frame_id, index, count, total_size, datagram[HEADER_SIZE:])
