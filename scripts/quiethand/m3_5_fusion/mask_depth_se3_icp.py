"""Occlusion-aware local SE(3) refinement from a mask and metric depth."""
import numpy as np
from scipy.spatial import cKDTree


def _mask_depth_cloud(depth, mask, intrinsic):
    valid = mask & np.isfinite(depth) & (depth >= 0.001)
    v, u = np.nonzero(valid)
    if len(u) == 0:
        raise ValueError('mask has no valid metric depth')
    pixels = np.c_[u, v, np.ones(len(u), dtype=np.float64)]
    rays = pixels @ np.linalg.inv(intrinsic).T
    return rays * depth[v, u, None]


def _visible_masked_vertices(vertices, pose, intrinsic, mask):
    camera = vertices @ pose[:3, :3].T + pose[:3, 3]
    front = np.isfinite(camera).all(axis=1) & (camera[:, 2] > 0.001)
    camera = camera[front]
    pixels_h = camera @ intrinsic.T
    pixels = np.rint(pixels_h[:, :2] / pixels_h[:, 2:3]).astype(np.int64)
    height, width = mask.shape
    inside = ((pixels[:, 0] >= 0) & (pixels[:, 0] < width)
              & (pixels[:, 1] >= 0) & (pixels[:, 1] < height))
    camera, pixels = camera[inside], pixels[inside]
    if len(camera) == 0:
        raise ValueError('projected CAD has no image pixels')

    flat = pixels[:, 1] * width + pixels[:, 0]
    order = np.lexsort((camera[:, 2], flat))
    flat, camera, pixels = flat[order], camera[order], pixels[order]
    frontmost = np.r_[True, flat[1:] != flat[:-1]]
    camera, pixels = camera[frontmost], pixels[frontmost]
    in_mask = mask[pixels[:, 1], pixels[:, 0]]
    camera = camera[in_mask]
    if len(camera) == 0:
        raise ValueError('visible CAD has no projection inside the object mask')
    return camera


def _proper_rigid_alignment(source, target):
    if source.shape != target.shape or source.ndim != 2 or source.shape[1] != 3:
        raise ValueError('source and target must be matching Nx3 arrays')
    source_center = source.mean(axis=0)
    target_center = target.mean(axis=0)
    centered_source = source - source_center
    centered_target = target - target_center
    u, singular_values, vt = np.linalg.svd(centered_source.T @ centered_target)
    if singular_values[1] <= np.finfo(np.float64).eps * singular_values[0]:
        raise ValueError('rigid correspondences are rank deficient')
    rotation = vt.T @ u.T
    if np.linalg.det(rotation) < 0:
        vt[-1] *= -1
        rotation = vt.T @ u.T
    translation = target_center - rotation @ source_center
    return rotation, translation, singular_values


def _rotation_angle_deg(rotation):
    cosine = np.clip((np.trace(rotation) - 1.0) / 2.0, -1.0, 1.0)
    return float(np.degrees(np.arccos(cosine)))


def refine_se3_projective_icp(vertices, pose, intrinsic, depth, mask,
                              max_iterations=8, trim_fraction=0.70,
                              max_distance_fraction=0.20, min_inliers=100):
    """Run one frozen local ICP schedule and return the refined object pose."""
    vertices = np.asarray(vertices, dtype=np.float64)
    current = np.asarray(pose, dtype=np.float64).copy()
    intrinsic = np.asarray(intrinsic, dtype=np.float64)
    depth = np.asarray(depth, dtype=np.float64)
    mask = np.asarray(mask, dtype=bool)
    if vertices.ndim != 2 or vertices.shape[1] != 3:
        raise ValueError('vertices must be Nx3')
    if current.shape != (4, 4) or intrinsic.shape != (3, 3):
        raise ValueError('pose and intrinsic shapes are invalid')
    if depth.shape != mask.shape:
        raise ValueError('depth and mask shapes differ')
    if not (0.0 < trim_fraction <= 1.0):
        raise ValueError('trim_fraction must be in (0, 1]')

    mesh_diagonal = float(np.linalg.norm(np.ptp(vertices, axis=0)))
    if not np.isfinite(mesh_diagonal) or mesh_diagonal <= 0:
        raise ValueError('mesh diagonal is invalid')
    maximum_distance = max_distance_fraction * mesh_diagonal
    observed = _mask_depth_cloud(depth, mask, intrinsic)
    observed_tree = cKDTree(observed)
    records = []
    for iteration in range(max_iterations):
        predicted = _visible_masked_vertices(vertices, current, intrinsic, mask)
        distances, indices = observed_tree.query(predicted, k=1, workers=1)
        within_cap = np.isfinite(distances) & (distances <= maximum_distance)
        capped_distances = distances[within_cap]
        if len(capped_distances) < min_inliers:
            raise ValueError('fewer than minimum capped RGB-D correspondences')
        trim_distance = float(np.quantile(capped_distances, trim_fraction))
        inliers = within_cap & (distances <= trim_distance)
        if int(inliers.sum()) < min_inliers:
            raise ValueError('fewer than minimum trimmed RGB-D correspondences')

        source = predicted[inliers]
        target = observed[indices[inliers]]
        rotation, translation, singular_values = _proper_rigid_alignment(source, target)
        next_pose = np.eye(4, dtype=np.float64)
        next_pose[:3, :3] = rotation @ current[:3, :3]
        next_pose[:3, 3] = rotation @ current[:3, 3] + translation
        rotation_step_deg = _rotation_angle_deg(rotation)
        translation_step_m = float(np.linalg.norm(translation))
        records.append({
            'iteration': iteration + 1,
            'visible_prediction_count': int(len(predicted)),
            'observed_point_count': int(len(observed)),
            'capped_correspondence_count': int(within_cap.sum()),
            'inlier_count': int(inliers.sum()),
            'maximum_distance_m': float(maximum_distance),
            'trim_distance_m': trim_distance,
            'median_inlier_distance_before_m': float(np.median(distances[inliers])),
            'rotation_step_deg': rotation_step_deg,
            'translation_step_m': translation_step_m,
            'alignment_singular_values': singular_values.tolist(),
        })
        current = next_pose
        if rotation_step_deg <= 0.05 and translation_step_m <= 0.00005:
            break

    rotation = current[:3, :3]
    if (not np.isfinite(current).all()
            or not np.allclose(rotation.T @ rotation, np.eye(3), atol=1e-6)
            or not np.isclose(np.linalg.det(rotation), 1.0, atol=1e-6)
            or not np.allclose(current[3], [0.0, 0.0, 0.0, 1.0])):
        raise ValueError('refined pose is not a finite proper SE3 transform')
    return current, {
        'iteration_count': len(records),
        'mesh_diagonal_m': mesh_diagonal,
        'iteration_records': records,
    }
