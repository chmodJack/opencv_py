"""Capture a USB camera and send JPEG frames to the WebSocket server."""

from __future__ import annotations

import argparse
import asyncio
import inspect
import logging
import os
from typing import Any

import cv2
from websockets.asyncio.client import connect


LOGGER = logging.getLogger("jpeg-websocket-client")


def open_camera(index: int, backend_name: str) -> cv2.VideoCapture:
    """Open a camera and use V4L2 when explicitly requested."""

    if backend_name == "v4l2":
        backend = getattr(cv2, "CAP_V4L2", cv2.CAP_ANY)
        camera = cv2.VideoCapture(index, backend)
    else:
        camera = cv2.VideoCapture(index, cv2.CAP_ANY)

    if not camera.isOpened():
        camera.release()
        raise RuntimeError(
            f"cannot open camera index {index}; check the USB camera and permissions"
        )
    return camera


def configure_camera(
    camera: cv2.VideoCapture,
    width: int,
    height: int,
    fps: float,
) -> tuple[int, int, float, str]:
    """Request MJPEG and the target capture mode, then return actual values."""

    # Setting MJPG before the dimensions lets V4L2 choose the compressed
    # 1920x1080 mode instead of falling back to an uncompressed pixel format.
    camera.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
    camera.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    camera.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    camera.set(cv2.CAP_PROP_FPS, fps)

    actual_width = int(camera.get(cv2.CAP_PROP_FRAME_WIDTH))
    actual_height = int(camera.get(cv2.CAP_PROP_FRAME_HEIGHT))
    actual_fps = camera.get(cv2.CAP_PROP_FPS)
    actual_fourcc = int(camera.get(cv2.CAP_PROP_FOURCC))
    fourcc = "".join(chr((actual_fourcc >> (8 * byte)) & 0xFF) for byte in range(4))
    return actual_width, actual_height, actual_fps, fourcc


def connect_options() -> dict[str, Any]:
    """Return options that keep local camera examples away from HTTP proxies."""

    options: dict[str, Any] = {}
    if "proxy" in inspect.signature(connect).parameters:
        options["proxy"] = None
    return options


async def send_camera(
    uri: str,
    camera_index: int,
    backend: str,
    width: int,
    height: int,
    fps: float,
    quality: int,
) -> None:
    camera = open_camera(camera_index, backend)
    try:
        actual_width, actual_height, actual_fps, fourcc = configure_camera(
            camera, width, height, fps
        )
        LOGGER.info(
            "camera %d opened: %dx%d at %.2f FPS, FOURCC=%s",
            camera_index,
            actual_width,
            actual_height,
            actual_fps,
            fourcc or "unknown",
        )
        if (actual_width, actual_height) != (width, height):
            LOGGER.warning(
                "camera did not accept requested resolution %dx%d; using %dx%d",
                width,
                height,
                actual_width,
                actual_height,
            )

        async with connect(uri, **connect_options()) as websocket:
            LOGGER.info("connected to %s; press Ctrl+C to stop", uri)
            frame_count = 0
            while True:
                ok, frame = camera.read()
                if not ok:
                    raise RuntimeError("camera frame capture failed")

                encoded_ok, encoded = cv2.imencode(
                    ".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, quality]
                )
                if not encoded_ok:
                    LOGGER.warning("could not encode camera frame as JPEG")
                    continue

                await websocket.send(encoded.tobytes())
                frame_count += 1
                if frame_count % 30 == 0:
                    LOGGER.info(
                        "sent %d frames, latest JPEG size=%d bytes",
                        frame_count,
                        encoded.nbytes,
                    )

                # send() may complete immediately while the TCP write buffer
                # has room. Yield explicitly so WebSocket keepalive tasks can
                # run alongside synchronous camera capture and JPEG encoding.
                await asyncio.sleep(0)
    finally:
        camera.release()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--camera", type=int, default=0, help="USB camera index (default: 0)")
    parser.add_argument("--uri", default="ws://127.0.0.1:10000", help="server WebSocket URI")
    parser.add_argument(
        "--backend",
        choices=("auto", "v4l2"),
        default="auto" if os.name != "posix" else "v4l2",
        help="OpenCV capture backend (default: v4l2 on POSIX, auto elsewhere)",
    )
    parser.add_argument("--width", type=int, default=1920, help="requested width (default: 1920)")
    parser.add_argument("--height", type=int, default=1080, help="requested height (default: 1080)")
    parser.add_argument("--fps", type=float, default=30.0, help="requested camera FPS (default: 30)")
    parser.add_argument(
        "--quality",
        type=int,
        choices=range(1, 101),
        metavar="1-100",
        default=90,
        help="JPEG quality (default: 90)",
    )
    return parser.parse_args()


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    args = parse_args()
    try:
        asyncio.run(
            send_camera(
                args.uri,
                args.camera,
                args.backend,
                args.width,
                args.height,
                args.fps,
                args.quality,
            )
        )
    except KeyboardInterrupt:
        LOGGER.info("camera client stopped")


if __name__ == "__main__":
    main()
