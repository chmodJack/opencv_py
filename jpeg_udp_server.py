"""Receive fragmented JPEG frames over UDP and display them with OpenCV."""

from __future__ import annotations

import argparse
import logging
import socket
import time
from dataclasses import dataclass, field

import cv2
import numpy as np

from jpeg_udp_protocol import Fragment, parse_packet


LOGGER = logging.getLogger("jpeg-udp-server")


@dataclass
class FrameAssembly:
    total_size: int
    count: int
    created_at: float
    fragments: dict[int, bytes] = field(default_factory=dict)

    def add(self, fragment: Fragment) -> bytes | None:
        if fragment.total_size != self.total_size or fragment.count != self.count:
            return None
        self.fragments.setdefault(fragment.index, fragment.payload)
        if len(self.fragments) != self.count:
            return None

        frame = b"".join(self.fragments[index] for index in range(self.count))
        if len(frame) != self.total_size:
            return None
        return frame


class UdpJpegServer:
    def __init__(
        self,
        host: str,
        port: int,
        window_name: str,
        frame_timeout: float,
        max_frame_size: int,
        max_inflight: int,
    ) -> None:
        self.host = host
        self.port = port
        self.window_name = window_name
        self.frame_timeout = frame_timeout
        self.max_frame_size = max_frame_size
        self.max_inflight = max_inflight
        self.frames: dict[tuple[tuple[str, int], int], FrameAssembly] = {}
        self.received_frames = 0
        self.completed_frames = 0
        self.dropped_frames = 0

    def _expire_frames(self, now: float) -> None:
        expired = [
            key
            for key, assembly in self.frames.items()
            if now - assembly.created_at > self.frame_timeout
        ]
        for key in expired:
            self.frames.pop(key, None)
            self.dropped_frames += 1

    def _make_room(self) -> None:
        if len(self.frames) < self.max_inflight:
            return
        oldest_key = min(self.frames, key=lambda key: self.frames[key].created_at)
        self.frames.pop(oldest_key, None)
        self.dropped_frames += 1

    def _handle_datagram(self, datagram: bytes, address: tuple[str, int], now: float) -> bytes | None:
        fragment = parse_packet(datagram)
        if fragment is None or fragment.total_size > self.max_frame_size:
            self.dropped_frames += 1
            return None

        key = (address, fragment.frame_id)
        assembly = self.frames.get(key)
        if assembly is None:
            self._make_room()
            assembly = FrameAssembly(fragment.total_size, fragment.count, now)
            self.frames[key] = assembly

        frame = assembly.add(fragment)
        if frame is not None:
            self.frames.pop(key, None)
            return frame
        return None

    @staticmethod
    def _decode_jpeg(payload: bytes) -> np.ndarray | None:
        try:
            return cv2.imdecode(np.frombuffer(payload, dtype=np.uint8), cv2.IMREAD_COLOR)
        except cv2.error:
            return None

    def run(self) -> None:
        sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, 4 * 1024 * 1024)
        sock.bind((self.host, self.port))
        sock.settimeout(0.02)

        LOGGER.info("listening on udp://%s:%d", self.host, self.port)
        try:
            while True:
                now = time.monotonic()
                self._expire_frames(now)
                try:
                    datagram, address = sock.recvfrom(65535)
                except socket.timeout:
                    key = cv2.waitKey(1) & 0xFF
                    if key == ord("q"):
                        break
                    continue

                self.received_frames += 1
                frame_bytes = self._handle_datagram(datagram, address, now)
                if frame_bytes is None:
                    continue

                frame = self._decode_jpeg(frame_bytes)
                if frame is None:
                    LOGGER.warning("invalid JPEG frame from %s", address)
                    continue

                self.completed_frames += 1
                cv2.imshow(self.window_name, frame)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break
                if self.completed_frames % 30 == 0:
                    LOGGER.info(
                        "displayed %d frames; UDP packets=%d, expired=%d",
                        self.completed_frames,
                        self.received_frames,
                        self.dropped_frames,
                    )
        finally:
            sock.close()
            cv2.destroyAllWindows()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default="0.0.0.0", help="UDP bind address")
    parser.add_argument("--port", type=int, default=8765, help="UDP port (default: 8765)")
    parser.add_argument("--window-name", default="UDP JPEG", help="OpenCV window title")
    parser.add_argument(
        "--frame-timeout",
        type=float,
        default=0.5,
        help="seconds before an incomplete frame is discarded (default: 0.5)",
    )
    parser.add_argument(
        "--max-frame-size",
        type=int,
        default=4 * 1024 * 1024,
        help="maximum reassembled JPEG size (default: 4194304)",
    )
    parser.add_argument(
        "--max-inflight",
        type=int,
        default=256,
        help="maximum incomplete frames held in memory (default: 256)",
    )
    return parser.parse_args()


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    args = parse_args()
    try:
        UdpJpegServer(
            args.host,
            args.port,
            args.window_name,
            args.frame_timeout,
            args.max_frame_size,
            args.max_inflight,
        ).run()
    except KeyboardInterrupt:
        LOGGER.info("server stopped")


if __name__ == "__main__":
    main()
