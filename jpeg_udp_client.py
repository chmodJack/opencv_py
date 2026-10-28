"""Capture a USB camera and send JPEG frames over UDP."""

from __future__ import annotations

import argparse
import logging
import os
import socket

import cv2

from jpeg_udp_protocol import packetize


LOGGER = logging.getLogger("jpeg-udp-client")


def open_camera(index: int, backend_name: str) -> cv2.VideoCapture:
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


def send_camera(
    host: str,
    port: int,
    camera_index: int,
    backend: str,
    width: int,
    height: int,
    fps: float,
    quality: int,
    chunk_size: int,
) -> None:
    camera = open_camera(camera_index, backend)
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_SNDBUF, 4 * 1024 * 1024)
    destination = (host, port)
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

        frame_id = 0
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

            jpeg = encoded.tobytes()
            packet_count = 0
            for packet in packetize(jpeg, frame_id, chunk_size):
                sock.sendto(packet, destination)
                packet_count += 1

            frame_id = (frame_id + 1) & 0xFFFFFFFF
            frame_count += 1
            if frame_count % 30 == 0:
                LOGGER.info(
                    "sent %d frames (%d UDP packets this frame, JPEG size=%d)",
                    frame_count,
                    packet_count,
                    len(jpeg),
                )
    finally:
        camera.release()
        sock.close()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default="192.168.31.7", help="UDP server address")
    parser.add_argument("--port", type=int, default=8765, help="UDP port (default: 8765)")
    parser.add_argument("--camera", type=int, default=0, help="USB camera index (default: 0)")
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
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=1200,
        help="JPEG bytes per UDP packet (default: 1200)",
    )
    return parser.parse_args()


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    args = parse_args()
    try:
        send_camera(
            args.host,
            args.port,
            args.camera,
            args.backend,
            args.width,
            args.height,
            args.fps,
            args.quality,
            args.chunk_size,
        )
    except KeyboardInterrupt:
        LOGGER.info("camera client stopped")


if __name__ == "__main__":
    main()
