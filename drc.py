import numpy as np


def verify_drc(
    centers: list[tuple[float, float]],
    scatter_radius: float = 0.1,
    min_wall_clearance: float = 0.05,
    tip_exclusion_radius: float = 0.20,
    max_section_radius: float = 1.25
) -> tuple[bool, str]:
    """Validates scatterer coordinates against FIB milling and PiFM tip clearance constraints."""
    try:
        min_center_dist: float = 2.0 * scatter_radius + min_wall_clearance
        for idx, (x_coord, y_coord) in enumerate(centers):
            radial_dist: float = float(np.hypot(x_coord, y_coord))
            if radial_dist < tip_exclusion_radius:
                return False, f"Scatterer {idx} encroaches on central PiFM tip zone (r={radial_dist:.3f} um)."
            if radial_dist > max_section_radius:
                return False, f"Scatterer {idx} exceeds maximum section radius (r={radial_dist:.3f} um)."
        num_pts: int = len(centers)
        for i in range(num_pts):
            for j in range(i + 1, num_pts):
                sep: float = float(np.hypot(centers[i][0] - centers[j][0], centers[i][1] - centers[j][1]))
                if sep < min_center_dist:
                    return False, f"Scatterers {i} and {j} violate minimum FIB wall spacing (d={sep:.3f} um)."
        return True, "DRC Passed"
    except Exception as exc:
        return False, f"DRC execution exception: {exc}"