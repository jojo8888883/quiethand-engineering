"""Same-pixel RGB-D translation refinement inside an observed object mask."""
import numpy as np


def refine_translation_once(vertices, pose, intrinsic, depth, mask):
    camera = vertices @ pose[:3, :3].T + pose[:3, 3]
    front = np.isfinite(camera).all(axis=1) & (camera[:, 2] > 0.001)
    camera = camera[front]
    pixels = camera @ intrinsic.T
    pixels = np.rint(pixels[:, :2] / pixels[:, 2:3]).astype(np.int64)
    height, width = mask.shape
    inside = ((pixels[:, 0] >= 0) & (pixels[:, 0] < width)
              & (pixels[:, 1] >= 0) & (pixels[:, 1] < height))
    camera, pixels = camera[inside], pixels[inside]
    if len(camera) == 0:
        raise ValueError('projected CAD has no image pixels')

    flat = pixels[:, 1] * width + pixels[:, 0]
    order = np.lexsort((camera[:, 2], flat))
    flat, camera, pixels = flat[order], camera[order], pixels[order]
    first = np.r_[True, flat[1:] != flat[:-1]]
    camera, pixels = camera[first], pixels[first]
    u, v = pixels[:, 0], pixels[:, 1]
    observed_depth = depth[v, u]
    valid = mask[v, u] & np.isfinite(observed_depth) & (observed_depth >= 0.001)
    camera, pixels, observed_depth = camera[valid], pixels[valid], observed_depth[valid]
    if len(camera) < 20:
        raise ValueError('fewer than 20 same-mask visible-surface correspondences')

    rays = np.c_[pixels, np.ones(len(pixels))] @ np.linalg.inv(intrinsic).T
    observed = rays * observed_depth[:, None]
    residual = observed - camera
    delta = np.median(residual, axis=0)
    refined = pose.copy()
    refined[:3, 3] += delta
    return refined, {
        'correspondence_count': int(len(camera)),
        'translation_delta_m': delta.tolist(),
        'median_abs_depth_residual_before_m': float(np.median(np.abs(residual[:, 2]))),
    }
