"""Prediction-side mask consistency for choosing between pose candidates."""
import numpy as np
from PIL import Image, ImageDraw


def _cross(origin, a, b):
    return ((a[0] - origin[0]) * (b[1] - origin[1])
            - (a[1] - origin[1]) * (b[0] - origin[0]))


def convex_hull(points):
    points = sorted(set(map(tuple, np.rint(points).astype(np.int64))))
    if len(points) <= 1:
        return points
    lower = []
    for point in points:
        while len(lower) >= 2 and _cross(lower[-2], lower[-1], point) <= 0:
            lower.pop()
        lower.append(point)
    upper = []
    for point in reversed(points):
        while len(upper) >= 2 and _cross(upper[-2], upper[-1], point) <= 0:
            upper.pop()
        upper.append(point)
    return lower[:-1] + upper[:-1]


def projected_hull_mask(vertices, pose, intrinsic, shape):
    camera = vertices @ pose[:3, :3].T + pose[:3, 3]
    valid = np.isfinite(camera).all(axis=1) & (camera[:, 2] > 0.001)
    camera = camera[valid]
    if len(camera) < 3:
        return np.zeros(shape, dtype=bool)
    pixels = camera @ intrinsic.T
    pixels = pixels[:, :2] / pixels[:, 2:3]
    hull = convex_hull(pixels)
    if len(hull) < 3:
        return np.zeros(shape, dtype=bool)
    canvas = Image.new('1', (shape[1], shape[0]), 0)
    ImageDraw.Draw(canvas).polygon(hull, fill=1)
    return np.asarray(canvas, dtype=bool)


def projected_mask_iou(vertices, pose, intrinsic, observed_mask):
    projected = projected_hull_mask(vertices, pose, intrinsic, observed_mask.shape)
    intersection = np.logical_and(projected, observed_mask).sum()
    union = np.logical_or(projected, observed_mask).sum()
    return 0.0 if union == 0 else float(intersection / union)


def select_source(track_iou, register_iou):
    """Prefer temporal continuity whenever current-mask support is not worse."""
    return 'track' if track_iou >= register_iou else 'register'
