from __future__ import annotations

import math
import random
from dataclasses import dataclass
from typing import Iterable, Sequence

import numpy as np

from .config import Q3Config
from .models import ChannelState, ClearPlan, DirectionEvidence, Point, RegionUpdate


@dataclass(frozen=True)
class Circle:
    center: Point
    radius: float


class ConservativeGrid:
    """Shared square grid and conservative region operations.

    A retained cell means only that the continuous feasible set may intersect it.
    Every deletion test is sufficient for the entire cell to be impossible.
    """

    def __init__(self, config: Q3Config):
        self.config = config
        h = config.grid_step_m
        radius = config.target_radius_m
        count = int(math.ceil(2.0 * radius / h))
        self.x_centers = -radius + h / 2.0 + np.arange(count, dtype=float) * h
        self.y_centers = self.x_centers.copy()
        self.xx, self.yy = np.meshgrid(self.x_centers, self.y_centers)
        # A square intersects the target disk iff its nearest point to O is in it.
        half = h / 2.0
        nearest_x = np.maximum(np.abs(self.xx) - half, 0.0)
        nearest_y = np.maximum(np.abs(self.yy) - half, 0.0)
        self.target_mask = nearest_x**2 + nearest_y**2 <= radius**2 + 1.0e-9

    @property
    def shape(self) -> tuple[int, int]:
        return self.target_mask.shape

    def fresh_region(self) -> np.ndarray:
        return self.target_mask.copy()

    def center_points(self, mask: np.ndarray) -> np.ndarray:
        return np.column_stack((self.xx[mask], self.yy[mask]))

    def _distance_bounds(self, station: Point) -> tuple[np.ndarray, np.ndarray]:
        half = self.config.grid_step_m / 2.0
        dx = np.abs(self.xx - station[0])
        dy = np.abs(self.yy - station[1])
        nearest_dx = np.maximum(dx - half, 0.0)
        nearest_dy = np.maximum(dy - half, 0.0)
        min_dist = np.hypot(nearest_dx, nearest_dy)
        max_dist = np.hypot(dx + half, dy + half)
        return min_dist, max_dist

    def _halfplane_may_intersect(
        self, a: float, b: float, c: float, relation: str
    ) -> np.ndarray:
        half = self.config.grid_step_m / 2.0
        center_value = a * self.xx + b * self.yy
        variation = (abs(a) + abs(b)) * half
        if relation == "le":
            return center_value - variation <= c + 1.0e-10
        if relation == "ge":
            return center_value + variation >= c - 1.0e-10
        raise ValueError(f"unknown relation: {relation}")

    def direction_compatible_mask(self, evidence: DirectionEvidence) -> np.ndarray:
        theta = math.radians(evidence.bearing_deg)
        alpha = math.radians(self.config.bearing_error_deg)
        lower = (math.cos(theta - alpha), math.sin(theta - alpha))
        upper = (math.cos(theta + alpha), math.sin(theta + alpha))
        sx, sy = evidence.position

        # cross(lower, X-S) >= 0
        a1, b1 = -lower[1], lower[0]
        c1 = a1 * sx + b1 * sy
        left_ok = self._halfplane_may_intersect(a1, b1, c1, "ge")

        # cross(upper, X-S) <= 0
        a2, b2 = -upper[1], upper[0]
        c2 = a2 * sx + b2 * sy
        right_ok = self._halfplane_may_intersect(a2, b2, c2, "le")

        min_dist, max_dist = self._distance_bounds(evidence.position)
        range_ok = min_dist <= self.config.maximum_receive_radius_m + 1.0e-9
        not_entirely_near = max_dist > self.config.near_radius_m + 1.0e-9
        return self.target_mask & left_ok & right_ok & range_ok & not_entirely_near

    def intersect_direction(self, mask: np.ndarray, evidence: DirectionEvidence) -> np.ndarray:
        """Intersect only retained cells; logically identical to mask & full compatibility."""
        rows, cols = np.nonzero(mask)
        if len(rows) == 0:
            return mask.copy()
        x = self.x_centers[cols]
        y = self.y_centers[rows]
        theta = math.radians(evidence.bearing_deg)
        alpha = math.radians(self.config.bearing_error_deg)
        lower = (math.cos(theta - alpha), math.sin(theta - alpha))
        upper = (math.cos(theta + alpha), math.sin(theta + alpha))
        sx, sy = evidence.position
        half = self.config.grid_step_m / 2.0

        a1, b1 = -lower[1], lower[0]
        c1 = a1 * sx + b1 * sy
        left_ok = a1 * x + b1 * y + (abs(a1) + abs(b1)) * half >= c1 - 1.0e-10
        a2, b2 = -upper[1], upper[0]
        c2 = a2 * sx + b2 * sy
        right_ok = a2 * x + b2 * y - (abs(a2) + abs(b2)) * half <= c2 + 1.0e-10

        dx = np.abs(x - sx)
        dy = np.abs(y - sy)
        min_dist = np.hypot(np.maximum(dx - half, 0.0), np.maximum(dy - half, 0.0))
        max_dist = np.hypot(dx + half, dy + half)
        compatible = (
            left_ok
            & right_ok
            & (min_dist <= self.config.maximum_receive_radius_m + 1.0e-9)
            & (max_dist > self.config.near_radius_m + 1.0e-9)
        )
        result = np.zeros_like(mask)
        result[rows[compatible], cols[compatible]] = True
        return result

    def update_direction(
        self, channel: ChannelState, evidence: DirectionEvidence
    ) -> RegionUpdate:
        old = self.fresh_region() if channel.region_mask is None else channel.region_mask
        before = int(np.count_nonzero(old))
        proposed = self.intersect_direction(old, evidence)
        return self._commit(channel, old, proposed, before, "direction")

    def update_no_signal(self, channel: ChannelState, station: Point) -> RegionUpdate:
        old = self.fresh_region() if channel.region_mask is None else channel.region_mask
        before = int(np.count_nonzero(old))
        _, max_dist = self._distance_bounds(station)
        definitely_received = max_dist <= self.config.guaranteed_receive_radius_m + 1.0e-9
        proposed = old & ~definitely_received
        return self._commit(channel, old, proposed, before, "no_signal")

    def exclude_no_signal(self, mask: np.ndarray, station: Point) -> np.ndarray:
        """Pure conservative no-signal update for branch evaluation."""
        _, max_dist = self._distance_bounds(station)
        definitely_received = max_dist <= self.config.guaranteed_receive_radius_m + 1.0e-9
        return mask & ~definitely_received

    def no_signal_reduction_count(self, mask: np.ndarray, station: Point) -> int:
        return int(np.count_nonzero(mask) - np.count_nonzero(self.exclude_no_signal(mask, station)))

    def update_clear_failure(self, channel: ChannelState, point: Point) -> RegionUpdate:
        if channel.region_mask is None:
            old = self.fresh_region()
        else:
            old = channel.region_mask
        before = int(np.count_nonzero(old))
        _, max_dist = self._distance_bounds(point)
        definitely_cleared = max_dist <= self.config.clear_radius_m + 1.0e-9
        proposed = old & ~definitely_cleared
        return self._commit(channel, old, proposed, before, "clear_failure")

    @staticmethod
    def _commit(
        channel: ChannelState,
        old: np.ndarray,
        proposed: np.ndarray,
        before: int,
        reason: str,
    ) -> RegionUpdate:
        after = int(np.count_nonzero(proposed))
        if after == 0:
            message = f"{reason} update rejected because it would empty the conservative region"
            channel.anomaly_log.append(message)
            return RegionUpdate(False, before, before, message)
        channel.region_mask = proposed
        channel.region_version += 1
        return RegionUpdate(True, before, after, reason)

    def point_guarantees_reception(self, mask: np.ndarray, point: Point) -> bool:
        if not np.any(mask):
            return False
        dx = self.xx[mask] - point[0]
        dy = self.yy[mask] - point[1]
        farthest_center = float(np.sqrt(dx * dx + dy * dy).max())
        return farthest_center + self.config.cell_radius_m <= (
            self.config.guaranteed_receive_radius_m + 1.0e-9
        )

    def point_guarantees_reception_with_history(
        self, channel: ChannelState, point: Point
    ) -> bool:
        """Sufficient variable-radius guarantee derived from successful stations.

        This helper is candidate-generation only.  It never changes the base
        feasible region or a completion certificate.
        """
        if channel.region_mask is None or not np.any(channel.region_mask):
            return False
        centers = self.center_points(channel.region_mask)
        rho = self.config.cell_radius_m
        candidate_upper = np.linalg.norm(centers - np.asarray(point), axis=1) + rho
        lower = np.full(len(centers), self.config.guaranteed_receive_radius_m)
        for evidence in channel.direction_observations:
            station = np.asarray(evidence.position)
            lower = np.maximum(lower, np.maximum(0.0, np.linalg.norm(centers - station, axis=1) - rho))
        return bool(np.all(candidate_upper <= lower + 1.0e-9))

    def region_extends_outside_fallback_hexagon(self, mask: np.ndarray) -> bool:
        """Return true when any retained square reaches outside the 1150 m hexagon."""
        if not np.any(mask):
            return False
        centers = self.center_points(mask)
        half = self.config.grid_step_m / 2.0
        offsets = np.asarray(
            [(-half, -half), (-half, half), (half, -half), (half, half)],
            dtype=float,
        )
        vertices = (centers[:, None, :] + offsets[None, :, :]).reshape(-1, 2)
        polygon = np.asarray(search_points(self.config)[1:], dtype=float)
        signs = []
        for index in range(len(polygon)):
            a = polygon[index]
            b = polygon[(index + 1) % len(polygon)]
            edge = b - a
            relative = vertices - a
            signs.append(edge[0] * relative[:, 1] - edge[1] * relative[:, 0])
        return bool(np.any(np.min(np.vstack(signs), axis=0) < -1.0e-9))

    def nearest_outer_point_index(self, mask: np.ndarray) -> int:
        """Return the 1-based fallback vertex nearest to the retained region."""
        outer = search_points(self.config)[1:]
        return min(
            range(1, 7),
            key=lambda index: (self.distance_to_region(mask, outer[index - 1]), index),
        )

    def distance_to_region(self, mask: np.ndarray, point: Point) -> float:
        """Exact distance from a point to the union of retained square cells."""
        rows, cols = np.nonzero(mask)
        if len(rows) == 0:
            return math.inf
        half = self.config.grid_step_m / 2.0
        dx = np.maximum(np.abs(self.x_centers[cols] - point[0]) - half, 0.0)
        dy = np.maximum(np.abs(self.y_centers[rows] - point[1]) - half, 0.0)
        return float(np.hypot(dx, dy).min())

    def distance_to_region_segment(self, mask: np.ndarray, start: Point, end: Point) -> float:
        """Conservative distance proxy from retained cells to a route segment.

        Cell centers are projected exactly onto the segment and the square-cell
        circumradius is subtracted.  The result is a lower bound, used only for
        opportunity ordering and never for a completion certificate.
        """
        points = self.center_points(mask)
        if len(points) == 0:
            return math.inf
        a = np.asarray(start, dtype=float)
        vector = np.asarray(end, dtype=float) - a
        length_sq = float(np.dot(vector, vector))
        if length_sq <= 1.0e-18:
            return self.distance_to_region(mask, start)
        t = np.clip(((points - a) @ vector) / length_sq, 0.0, 1.0)
        projection = a + t[:, None] * vector
        center_distance = np.linalg.norm(points - projection, axis=1)
        return max(0.0, float(center_distance.min()) - self.config.cell_radius_m)

    def nearest_region_center_to_segment(self, mask: np.ndarray, start: Point, end: Point) -> Point:
        points = self.center_points(mask)
        if len(points) == 0:
            raise ValueError("empty region has no nearest center")
        a = np.asarray(start, dtype=float)
        vector = np.asarray(end, dtype=float) - a
        length_sq = float(np.dot(vector, vector))
        if length_sq <= 1.0e-18:
            distances = np.linalg.norm(points - a, axis=1)
        else:
            t = np.clip(((points - a) @ vector) / length_sq, 0.0, 1.0)
            projection = a + t[:, None] * vector
            distances = np.linalg.norm(points - projection, axis=1)
        return tuple(map(float, points[int(np.argmin(distances))]))

    def representative_points(self, mask: np.ndarray, limit: int) -> list[Point]:
        points = self.center_points(mask)
        if len(points) <= limit:
            return [tuple(map(float, point)) for point in points]
        selected: list[int] = []
        for values in (points[:, 0], points[:, 1], points[:, 0] + points[:, 1], points[:, 0] - points[:, 1]):
            selected.extend((int(np.argmin(values)), int(np.argmax(values))))
        selected = list(dict.fromkeys(selected))
        min_distance_sq = np.full(len(points), np.inf)
        for index in selected:
            delta = points - points[index]
            min_distance_sq = np.minimum(min_distance_sq, np.einsum("ij,ij->i", delta, delta))
        while len(selected) < limit:
            index = int(np.argmax(min_distance_sq))
            selected.append(index)
            delta = points - points[index]
            min_distance_sq = np.minimum(min_distance_sq, np.einsum("ij,ij->i", delta, delta))
        return [tuple(map(float, points[index])) for index in selected[:limit]]

    def build_clear_plan(
        self, mask: np.ndarray, current: Point, reconnect: Point | None = None
    ) -> ClearPlan:
        if not np.any(mask):
            raise ValueError("cannot build a clear plan for an empty region")
        points = self.center_points(mask)
        circle = minimum_enclosing_circle(points)
        rho = self.config.cell_radius_m
        if circle.radius + rho <= self.config.clear_radius_m + 1.0e-9:
            clear_points = [circle.center]
        else:
            clear_points = self._greedy_cover(mask, current)
        ordered = choose_clear_order(clear_points, current, reconnect, self.config)
        route_length = route_length_from(current, ordered)
        m = len(ordered)
        upper = worst_clear_completion_time(ordered, current, reconnect, self.config)
        return ClearPlan(
            points=ordered,
            route_length_m=route_length,
            upper_time_s=upper,
            certificate={
                "grid_step_m": self.config.grid_step_m,
                "cell_radius_m": rho,
                "retained_cell_count": int(np.count_nonzero(mask)),
                "clear_point_count": m,
                "reconnect_point": list(reconnect) if reconnect is not None else None,
                "worst_completion_time_s": upper,
                "all_retained_cells_fully_covered": self.verify_clear_cover(mask, ordered),
                "minimum_enclosing_circle": {
                    "center": list(circle.center),
                    "radius_of_cell_centers_m": circle.radius,
                },
            },
        )

    def _cover_offsets(self) -> list[tuple[int, int]]:
        h = self.config.grid_step_m
        usable = self.config.clear_radius_m - self.config.cell_radius_m
        reach = int(math.floor(usable / h + 1.0e-12))
        return [
            (dr, dc)
            for dr in range(-reach, reach + 1)
            for dc in range(-reach, reach + 1)
            if math.hypot(dr * h, dc * h) <= usable + 1.0e-9
        ]

    def _greedy_cover(self, mask: np.ndarray, current: Point) -> list[Point]:
        import heapq

        original = mask.copy()
        uncovered = mask.copy()
        offsets = self._cover_offsets()
        counts = np.zeros(mask.shape, dtype=np.int32)
        rows, cols = mask.shape
        for dr, dc in offsets:
            src_r0, src_r1 = max(0, -dr), min(rows, rows - dr)
            src_c0, src_c1 = max(0, -dc), min(cols, cols - dc)
            dst_r0, dst_r1 = src_r0 + dr, src_r1 + dr
            dst_c0, dst_c1 = src_c0 + dc, src_c1 + dc
            counts[dst_r0:dst_r1, dst_c0:dst_c1] += uncovered[src_r0:src_r1, src_c0:src_c1]

        heap: list[tuple[int, float, int]] = []
        for row, col in np.argwhere(original):
            distance = math.hypot(self.x_centers[col] - current[0], self.y_centers[row] - current[1])
            flat = int(row * cols + col)
            heap.append((-int(counts[row, col]), distance, flat))
        heapq.heapify(heap)

        result: list[Point] = []
        remaining = int(np.count_nonzero(uncovered))
        while remaining:
            while heap:
                neg_count, _, flat = heapq.heappop(heap)
                row, col = divmod(flat, cols)
                if original[row, col] and -neg_count == int(counts[row, col]) and counts[row, col] > 0:
                    break
            else:
                raise RuntimeError("greedy clear cover lost all candidates")
            result.append((float(self.x_centers[col]), float(self.y_centers[row])))

            removed: list[tuple[int, int]] = []
            for dr, dc in offsets:
                rr, cc = row + dr, col + dc
                if 0 <= rr < rows and 0 <= cc < cols and uncovered[rr, cc]:
                    uncovered[rr, cc] = False
                    removed.append((rr, cc))
            remaining -= len(removed)

            touched: set[int] = set()
            for rr, cc in removed:
                for dr, dc in offsets:
                    qr, qc = rr - dr, cc - dc
                    if 0 <= qr < rows and 0 <= qc < cols and original[qr, qc]:
                        counts[qr, qc] -= 1
                        touched.add(qr * cols + qc)
            for flat_index in touched:
                qr, qc = divmod(flat_index, cols)
                if counts[qr, qc] > 0:
                    distance = math.hypot(self.x_centers[qc] - current[0], self.y_centers[qr] - current[1])
                    heapq.heappush(heap, (-int(counts[qr, qc]), distance, int(flat_index)))
        return result

    def verify_clear_cover(self, mask: np.ndarray, clear_points: Sequence[Point]) -> bool:
        retained = self.center_points(mask)
        if len(retained) == 0:
            return True
        covered = np.zeros(len(retained), dtype=bool)
        usable_sq = (self.config.clear_radius_m - self.config.cell_radius_m) ** 2
        for point in clear_points:
            delta = retained - np.asarray(point, dtype=float)
            covered |= np.einsum("ij,ij->i", delta, delta) <= usable_sq + 1.0e-9
            if bool(np.all(covered)):
                return True
        return False


def search_points(config: Q3Config) -> list[Point]:
    return [(0.0, 0.0)] + [
        (
            config.ring_radius_m * math.cos(k * math.pi / 3.0),
            config.ring_radius_m * math.sin(k * math.pi / 3.0),
        )
        for k in range(6)
    ]


def search_coverage_certificate(config: Q3Config) -> dict[str, float | bool]:
    r0 = config.ring_radius_m
    outer = config.target_radius_m
    inner = config.guaranteed_receive_radius_m
    angle = math.pi / 6.0

    def nearest_distance(radius: float) -> float:
        return math.sqrt(radius * radius + r0 * r0 - 2.0 * radius * r0 * math.cos(angle))

    d_inner = nearest_distance(inner)
    d_outer = nearest_distance(outer)
    worst = max(d_inner, d_outer)
    return {
        "covered": worst <= inner + 1.0e-9,
        "inner_endpoint_distance_m": d_inner,
        "outer_endpoint_distance_m": d_outer,
        "worst_distance_m": worst,
        "safety_margin_m": inner - worst,
    }


def responsibility_zone_certificate(
    zone_index: int, station: Point, config: Q3Config
) -> dict[str, object]:
    """Exact single-disk containment certificate for one of the seven zones.

    Zone 0 is the central Voronoi hexagon.  Zones 1..6 are bounded by the
    inner Voronoi chord, two radial edges, and the target-circle arc.  A
    convex squared-distance function reaches its maximum on straight edges
    at an endpoint; on the circular arc it is enough to add the antipodal
    direction when that direction lies inside the arc.
    """
    if zone_index not in range(7):
        raise ValueError("responsibility zone index must be in 0..6")
    ring = config.ring_radius_m
    outer = config.target_radius_m
    half_sector = math.pi / 6.0
    candidates: list[Point] = []

    if zone_index == 0:
        inradius = ring / 2.0
        circumradius = inradius / math.cos(half_sector)
        candidates = [
            (
                circumradius * math.cos(half_sector + k * math.pi / 3.0),
                circumradius * math.sin(half_sector + k * math.pi / 3.0),
            )
            for k in range(6)
        ]
    else:
        phi = (zone_index - 1) * math.pi / 3.0
        inner_radius = (ring / 2.0) / math.cos(half_sector)
        for delta in (-half_sector, half_sector):
            angle = phi + delta
            candidates.append((inner_radius * math.cos(angle), inner_radius * math.sin(angle)))
            candidates.append((outer * math.cos(angle), outer * math.sin(angle)))

        sx, sy = station
        if abs(sx) + abs(sy) > 1.0e-15:
            antipodal = math.atan2(-sy, -sx)
            relative = (antipodal - phi + math.pi) % (2.0 * math.pi) - math.pi
            if -half_sector - 1.0e-12 <= relative <= half_sector + 1.0e-12:
                angle = phi + relative
                candidates.append((outer * math.cos(angle), outer * math.sin(angle)))

    witness = max(candidates, key=lambda point: math.dist(station, point))
    maximum = math.dist(station, witness)
    limit = config.guaranteed_receive_radius_m
    return {
        "zone_index": zone_index,
        "station": station,
        "covered": maximum <= limit + 1.0e-9,
        "maximum_distance_m": maximum,
        "safety_margin_m": limit - maximum,
        "witness": witness,
        "checked_boundary_candidates": len(candidates),
    }


def certified_responsibility_zones(station: Point, config: Q3Config) -> set[int]:
    """Return zones wholly contained in one guaranteed-reception disk."""
    return {
        index
        for index in range(7)
        if bool(responsibility_zone_certificate(index, station, config)["covered"])
    }


def _circle_from_two(a: np.ndarray, b: np.ndarray) -> Circle:
    center = (a + b) / 2.0
    return Circle((float(center[0]), float(center[1])), float(np.linalg.norm(a - center)))


def _circle_from_three(a: np.ndarray, b: np.ndarray, c: np.ndarray) -> Circle | None:
    ab = b - a
    ac = c - a
    cross = float(ab[0] * ac[1] - ab[1] * ac[0])
    if abs(cross) <= 1.0e-12:
        return None
    ax, ay = a
    bx, by = b
    cx, cy = c
    d = 2.0 * (ax * (by - cy) + bx * (cy - ay) + cx * (ay - by))
    ux = ((ax * ax + ay * ay) * (by - cy) + (bx * bx + by * by) * (cy - ay) + (cx * cx + cy * cy) * (ay - by)) / d
    uy = ((ax * ax + ay * ay) * (cx - bx) + (bx * bx + by * by) * (ax - cx) + (cx * cx + cy * cy) * (bx - ax)) / d
    center = np.asarray([ux, uy])
    return Circle((float(ux), float(uy)), float(np.linalg.norm(a - center)))


def _contains(circle: Circle, point: np.ndarray) -> bool:
    return math.hypot(point[0] - circle.center[0], point[1] - circle.center[1]) <= circle.radius + 1.0e-8


def _convex_hull(points: np.ndarray) -> np.ndarray:
    """Return 2-D hull vertices using the monotone-chain construction."""
    if len(points) <= 1:
        return points.copy()
    order = np.lexsort((points[:, 1], points[:, 0]))
    ordered = points[order]
    unique = np.ones(len(ordered), dtype=bool)
    unique[1:] = np.any(ordered[1:] != ordered[:-1], axis=1)
    ordered = ordered[unique]
    if len(ordered) <= 2:
        return ordered
    # For each x-coordinate only the lowest and highest point can be a hull
    # vertex.  Grid regions contain hundreds of thousands of vertically
    # interior points, so this exact reduction keeps the dependency-free
    # fallback fast enough for live use.
    starts = np.flatnonzero(np.r_[True, ordered[1:, 0] != ordered[:-1, 0]])
    ends = np.r_[starts[1:] - 1, len(ordered) - 1]
    boundary_indices = np.unique(np.r_[starts, ends])
    ordered = ordered[boundary_indices]
    if len(ordered) <= 2:
        return ordered

    def cross(o: np.ndarray, a: np.ndarray, b: np.ndarray) -> float:
        return float((a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0]))

    lower: list[np.ndarray] = []
    for point in ordered:
        while len(lower) >= 2 and cross(lower[-2], lower[-1], point) <= 0.0:
            lower.pop()
        lower.append(point)
    upper: list[np.ndarray] = []
    for point in ordered[::-1]:
        while len(upper) >= 2 and cross(upper[-2], upper[-1], point) <= 0.0:
            upper.pop()
        upper.append(point)
    return np.asarray(lower[:-1] + upper[:-1], dtype=float)


def minimum_enclosing_circle(points: np.ndarray) -> Circle:
    """Deterministic-seeded randomized incremental exact circle for point sets."""
    if len(points) == 0:
        raise ValueError("minimum enclosing circle needs at least one point")
    if len(points) > 100:
        # Interior points cannot support a minimum enclosing circle. Reducing to the
        # convex hull changes neither the exact circle nor its certificate.
        points = _convex_hull(points)
    order = list(range(len(points)))
    random.Random(20260912).shuffle(order)
    shuffled = points[order]
    circle = Circle((float(shuffled[0, 0]), float(shuffled[0, 1])), 0.0)
    for i, p in enumerate(shuffled):
        if _contains(circle, p):
            continue
        circle = Circle((float(p[0]), float(p[1])), 0.0)
        for j in range(i):
            q = shuffled[j]
            if _contains(circle, q):
                continue
            circle = _circle_from_two(p, q)
            for k in range(j):
                r = shuffled[k]
                if _contains(circle, r):
                    continue
                candidate = _circle_from_three(p, q, r)
                if candidate is None:
                    candidates = [_circle_from_two(p, q), _circle_from_two(p, r), _circle_from_two(q, r)]
                    circle = min((item for item in candidates if all(_contains(item, x) for x in (p, q, r))), key=lambda item: item.radius)
                else:
                    circle = candidate
    return circle


def route_length_from(start: Point, points: Sequence[Point]) -> float:
    total = 0.0
    previous = start
    for point in points:
        total += math.dist(previous, point)
        previous = point
    return total


def order_route(points: Sequence[Point], start: Point) -> list[Point]:
    remaining = list(points)
    ordered: list[Point] = []
    current = start
    while remaining:
        index = min(range(len(remaining)), key=lambda i: math.dist(current, remaining[i]))
        current = remaining.pop(index)
        ordered.append(current)
    # One bounded first-improvement 2-opt pass. The path is open, not a cycle.
    if len(ordered) <= 300:
        base = route_length_from(start, ordered)
        for i in range(len(ordered) - 1):
            for j in range(i + 1, len(ordered)):
                candidate = ordered[:i] + list(reversed(ordered[i : j + 1])) + ordered[j + 1 :]
                length = route_length_from(start, candidate)
                if length + 1.0e-9 < base:
                    ordered = candidate
                    break
            else:
                continue
            break
    return ordered


def worst_clear_completion_time(
    points: Sequence[Point],
    start: Point,
    reconnect: Point | None,
    config: Q3Config,
) -> float:
    if not points:
        return 0.0
    travelled = 0.0
    previous = start
    worst = 0.0
    for index, point in enumerate(points):
        travelled += math.dist(previous, point)
        action_time = (
            config.clear_failure_time_s * index + config.clear_success_time_s
        )
        reconnect_time = (
            math.dist(point, reconnect) / config.speed_mps
            if reconnect is not None
            else 0.0
        )
        worst = max(worst, travelled / config.speed_mps + action_time + reconnect_time)
        previous = point
    return worst


def choose_clear_order(
    points: Sequence[Point],
    start: Point,
    reconnect: Point | None,
    config: Q3Config,
) -> list[Point]:
    """Choose the best of the three confirmed, explainable route candidates."""
    if len(points) <= 1:
        return list(points)
    nearest = order_route(points, start)
    if reconnect is None:
        candidates = [nearest, list(reversed(nearest))]
    else:
        far_to_near = sorted(points, key=lambda point: math.dist(point, reconnect), reverse=True)
        candidates = [far_to_near, list(reversed(far_to_near)), nearest]
    return min(
        candidates,
        key=lambda order: (
            worst_clear_completion_time(order, start, reconnect, config),
            route_length_from(start, order),
            tuple(order),
        ),
    )
