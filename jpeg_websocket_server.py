"""Receive JPEG frames over WebSocket and display them with OpenCV.

The client must send each JPEG image as one binary WebSocket message. Text
messages are ignored and answered with a short error message.
"""

from __future__ import annotations

import argparse
import asyncio
import contextlib
import logging
from typing import Any

import cv2
import numpy as np
from websockets.exceptions import ConnectionClosed

try:
    # websockets >= 14
    from websockets.asyncio.server import serve
except ImportError:  # pragma: no cover - compatibility with older releases
    from websockets.server import serve


LOGGER = logging.getLogger("jpeg-websocket-server")


def decode_jpeg(payload: bytes | bytearray | memoryview) -> np.ndarray | None:
    """Decode a binary WebSocket payload into a BGR OpenCV image.

    ``None`` is returned for an empty or invalid JPEG payload.
    """

    if not isinstance(payload, (bytes, bytearray, memoryview)):
        return None

    encoded = np.frombuffer(payload, dtype=np.uint8)
    if encoded.size == 0:
        return None

    try:
        return cv2.imdecode(encoded, cv2.IMREAD_COLOR)
    except cv2.error:
        return None


class FrameViewer:
    """Display decoded frames and provide a signal for a clean shutdown."""

    def __init__(self, window_name: str, stop_event: asyncio.Event) -> None:
        self.window_name = window_name
        self.stop_event = stop_event
        self.frame_count = 0

    def show(self, frame: np.ndarray) -> None:
        self.frame_count += 1
        cv2.imshow(self.window_name, frame)

        # A one millisecond wait both refreshes the HighGUI window and lets
        # OpenCV receive keyboard events. Press q in the window to stop.
        if cv2.waitKey(1) & 0xFF == ord("q"):
            self.stop_event.set()

    def close(self) -> None:
        cv2.destroyAllWindows()


def enqueue_latest(frame_queue: asyncio.Queue[bytes], payload: bytes) -> None:
    """Keep only the newest frame so a slow display cannot build a backlog."""

    if frame_queue.full():
        try:
            frame_queue.get_nowait()
            frame_queue.task_done()
        except asyncio.QueueEmpty:
            pass
    try:
        frame_queue.put_nowait(payload)
    except asyncio.QueueFull:
        # Another handler may have filled the queue between get_nowait and
        # put_nowait. Dropping this frame is preferable to blocking receive.
        pass


async def handle_client(
    websocket: Any,
    frame_queue: asyncio.Queue[bytes],
    stop_event: asyncio.Event,
) -> None:
    """Receive complete WebSocket messages without blocking on display work."""

    peer = getattr(websocket, "remote_address", "unknown")
    LOGGER.info("client connected: %s", peer)

    received_count = 0
    try:
        async for message in websocket:
            if isinstance(message, str):
                await websocket.send("Send each JPEG as a binary WebSocket message.")
                continue

            enqueue_latest(frame_queue, message)
            received_count += 1
            if stop_event.is_set():
                break

            if received_count % 30 == 0:
                LOGGER.info("received %d frames from %s", received_count, peer)

            # recv() may return immediately while frames are already queued.
            # Yield explicitly so WebSocket ping/pong and other clients run.
            await asyncio.sleep(0)
    except ConnectionClosed as error:
        LOGGER.info(
            "client disconnected: %s (code=%s, reason=%s)",
            peer,
            error.code,
            error.reason,
        )
    finally:
        LOGGER.info("client connection closed: %s", peer)


async def display_frames(frame_queue: asyncio.Queue[bytes], viewer: FrameViewer) -> None:
    """Decode and display the newest queued frame."""

    try:
        while not viewer.stop_event.is_set():
            payload = await frame_queue.get()
            try:
                frame = decode_jpeg(payload)
                if frame is None:
                    LOGGER.warning("invalid JPEG payload")
                else:
                    viewer.show(frame)
            finally:
                frame_queue.task_done()

            # get() may also return immediately when the producer is faster
            # than display. Keep the asyncio event loop responsive.
            await asyncio.sleep(0)
    except asyncio.CancelledError:
        raise
    except Exception:
        # Keep display failures visible and let run_server close the client
        # cleanly instead of leaving a live socket with no frame consumer.
        LOGGER.exception("OpenCV display task failed")
        viewer.stop_event.set()


async def run_server(host: str, port: int, window_name: str, max_frame_size: int) -> None:
    """Run the WebSocket server until q is pressed or Ctrl+C is used."""

    stop_event = asyncio.Event()
    viewer = FrameViewer(window_name, stop_event)
    frame_queue: asyncio.Queue[bytes] = asyncio.Queue(maxsize=1)
    display_task = asyncio.create_task(display_frames(frame_queue, viewer))

    async def client_handler(websocket: Any, *_path: Any) -> None:
        await handle_client(websocket, frame_queue, stop_event)

    try:
        async with serve(
            client_handler,
            host,
            port,
            max_size=max_frame_size,
        ):
            LOGGER.info("listening on ws://%s:%d", host, port)
            await stop_event.wait()
    finally:
        display_task.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await display_task
        viewer.close()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default="0.0.0.0", help="address to bind (default: 0.0.0.0)")
    parser.add_argument("--port", type=int, default=10000, help="WebSocket port (default: 10000)")
    parser.add_argument("--window-name", default="WebSocket JPEG", help="OpenCV window title")
    parser.add_argument(
        "--max-frame-size",
        type=int,
        default=10 * 1024 * 1024,
        metavar="BYTES",
        help="maximum WebSocket message size (default: 10485760)",
    )
    return parser.parse_args()


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    args = parse_args()

    try:
        asyncio.run(run_server(args.host, args.port, args.window_name, args.max_frame_size))
    except KeyboardInterrupt:
        LOGGER.info("server stopped")


if __name__ == "__main__":
    main()
