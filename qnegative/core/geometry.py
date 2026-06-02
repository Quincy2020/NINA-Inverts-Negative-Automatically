from __future__ import annotations

import math

import cv2
import numpy as np

from qnegative.core.models import ImagePoint, ImageRect, ImageSize


def scale_point(point: ImagePoint, from_size: ImageSize, to_size: ImageSize) -> ImagePoint:
    return ImagePoint(
        x=round(point.x * to_size.width / from_size.width),
        y=round(point.y * to_size.height / from_size.height),
    )


def scale_rect(rect: ImageRect, from_size: ImageSize, to_size: ImageSize) -> ImageRect:
    return ImageRect(
        x=round(rect.x * to_size.width / from_size.width),
        y=round(rect.y * to_size.height / from_size.height),
        width=max(1, round(rect.width * to_size.width / from_size.width)),
        height=max(1, round(rect.height * to_size.height / from_size.height)),
        angle=rect.angle,
    )


def rotated_rect_corners(rect: ImageRect) -> np.ndarray:
    half_w = rect.width / 2.0
    half_h = rect.height / 2.0
    center_x = rect.x + half_w
    center_y = rect.y + half_h
    radians = math.radians(rect.angle)
    cos_a = math.cos(radians)
    sin_a = math.sin(radians)

    local_points = np.array(
        [
            [-half_w, -half_h],
            [half_w, -half_h],
            [half_w, half_h],
            [-half_w, half_h],
        ],
        dtype=np.float32,
    )

    rotation = np.array(
        [
            [cos_a, -sin_a],
            [sin_a, cos_a],
        ],
        dtype=np.float32,
    )
    return local_points @ rotation.T + np.array([center_x, center_y], dtype=np.float32)


def warp_rotated_rect(image: np.ndarray, rect: ImageRect) -> np.ndarray:
    width = max(1, int(round(rect.width)))
    height = max(1, int(round(rect.height)))
    source = rotated_rect_corners(rect).astype(np.float32)
    target = np.array(
        [
            [0, 0],
            [width - 1, 0],
            [width - 1, height - 1],
            [0, height - 1],
        ],
        dtype=np.float32,
    )
    matrix = cv2.getPerspectiveTransform(source, target)
    return cv2.warpPerspective(
        image,
        matrix,
        (width, height),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_REPLICATE,
    )


def clamp_rect_to_image(rect: ImageRect, size: ImageSize) -> ImageRect:
    width = min(max(1, rect.width), size.width)
    height = min(max(1, rect.height), size.height)
    x = min(max(0, rect.x), max(0, size.width - width))
    y = min(max(0, rect.y), max(0, size.height - height))
    return ImageRect(x=x, y=y, width=width, height=height, angle=rect.angle)
