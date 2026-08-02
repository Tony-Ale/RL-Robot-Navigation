import numpy as np


def is_boundary_wall(wall, map_x, map_y, atol=1e-6):
    """Return whether an axis-aligned wall lies along the room perimeter."""
    orientation = float(wall.orientation)
    horizontal = abs(np.sin(orientation)) <= atol
    vertical = abs(np.cos(orientation)) <= atol
    if horizontal:
        return np.isclose(
            abs(float(wall.y)) + float(wall.thickness) / 2.0,
            map_y / 2.0,
            atol=atol,
            rtol=0.0,
        )
    if vertical:
        return np.isclose(
            abs(float(wall.x)) + float(wall.thickness) / 2.0,
            map_x / 2.0,
            atol=atol,
            rtol=0.0,
        )
    return False
