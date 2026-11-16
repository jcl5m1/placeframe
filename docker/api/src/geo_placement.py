from __future__ import annotations

from typing import Any

from numpy import argmin, array, cos, cross, dtype, float64, ndarray, radians, sin, sqrt, zeros
from scipy.spatial.transform import Rotation

# A capture may carry the capturing device's location (location.json at the root of the capture tar):
#   {"latitude": deg, "longitude": deg, "altitude": metres above the WGS84 ellipsoid (optional),
#    "horizontal_accuracy": metres (optional), "map_up_axis": "x" | "-x" | "y" | "-y" | "z" | "-z" (optional)}
# A map built from it can then be placed on the globe: the centroid of its camera positions sits at that location
# and its vertical axis points along local up. Heading is unknown from a location alone, so the map's horizontal
# axes are left in an arbitrary orientation; that does not affect selecting maps by distance and can be refined by
# hand in the MapRegistrationTool.
WGS84_A = 6378137.0
WGS84_E2 = 6.69437999014e-3
MAX_HORIZONTAL_ACCURACY_M = 200.0
AXES = "xyz"

type Positions = ndarray[tuple[int, int], dtype[float64]]
type Matrix3 = ndarray[tuple[int, int], dtype[float64]]
type Vector3 = ndarray[tuple[int], dtype[float64]]


def placement_from_location(
    location: dict[str, Any], camera_positions: Positions
) -> tuple[list[float], list[float]] | None:
    latitude, longitude = location.get("latitude"), location.get("longitude")
    if latitude is None or longitude is None:
        return None
    accuracy = location.get("horizontal_accuracy")
    if accuracy is not None and float(accuracy) > MAX_HORIZONTAL_ACCURACY_M:
        return None
    enu_from_map = _enu_from_map(camera_positions, location.get("map_up_axis"))
    ecef_from_map: Matrix3 = _ecef_from_enu(float(latitude), float(longitude)) @ enu_from_map
    anchor: Vector3 = camera_positions.mean(axis=0) if len(camera_positions) else zeros(3)
    origin = _ecef_from_geodetic(float(latitude), float(longitude), float(location.get("altitude") or 0.0))
    translation: Vector3 = origin - ecef_from_map @ anchor
    rotation = Rotation.from_matrix(ecef_from_map).as_quat()
    return [float(v) for v in translation], [float(v) for v in rotation]


def _enu_from_map(camera_positions: Positions, up_axis: str | None) -> Matrix3:
    # Without a hint, the vertical axis is the one along which the camera positions vary least: a walk-through
    # spreads horizontally, not vertically. Its sign cannot be told from positions and is taken as positive.
    if up_axis is None:
        up_axis = AXES[int(argmin(camera_positions.std(axis=0)))] if len(camera_positions) > 2 else "z"
    index = AXES.index(up_axis[-1])
    up = zeros(3)
    up[index] = -1.0 if up_axis.startswith("-") else 1.0
    east = zeros(3)
    east[(index + 1) % 3] = 1.0
    return array([east, cross(up, east), up])


def _ecef_from_enu(latitude: float, longitude: float) -> Matrix3:
    lat, lon = radians(latitude), radians(longitude)
    east = [-sin(lon), cos(lon), 0.0]
    north = [-sin(lat) * cos(lon), -sin(lat) * sin(lon), cos(lat)]
    up = [cos(lat) * cos(lon), cos(lat) * sin(lon), sin(lat)]
    return array([east, north, up]).T


def _ecef_from_geodetic(latitude: float, longitude: float, altitude: float) -> Vector3:
    lat, lon = radians(latitude), radians(longitude)
    n = WGS84_A / sqrt(1 - WGS84_E2 * sin(lat) ** 2)
    return array([
        (n + altitude) * cos(lat) * cos(lon),
        (n + altitude) * cos(lat) * sin(lon),
        (n * (1 - WGS84_E2) + altitude) * sin(lat),
    ])
