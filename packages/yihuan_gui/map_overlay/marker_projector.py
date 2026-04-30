from __future__ import annotations

from typing import Iterable

from .models import (
    MapCategory,
    MapMarker,
    MapMatchResult,
    MarkerCluster,
    ProjectedMarker,
    ZeroluckMapData,
    marker_display_name_zh,
)
from .settings import MapOverlaySettings


ProjectedItem = ProjectedMarker | MarkerCluster


class MarkerProjector:
    def project(
        self,
        data: ZeroluckMapData,
        match: MapMatchResult,
        settings: MapOverlaySettings,
        markers: Iterable[MapMarker] | None = None,
    ) -> list[ProjectedItem]:
        if not match.success or match.transform is None:
            return []
        try:
            import numpy as np
        except ModuleNotFoundError:
            return []

        categories_by_id = data.categories_by_id
        query = settings.search_text.strip().lower()
        screen_x, screen_y, screen_w, screen_h = match.screen_rect
        margin = max(32, int(settings.icon_size) * 2)
        projected: list[ProjectedMarker] = []
        for marker in markers or data.markers:
            if marker.category_id not in settings.enabled_categories:
                continue
            category = categories_by_id.get(marker.category_id)
            if query and not _matches_query(marker, category, query):
                continue
            tile_x, tile_y = marker.tile_pixel(data.bounds)
            point = np.array([tile_x, tile_y, 1.0], dtype=float)
            transformed = match.transform @ point
            if abs(float(transformed[2])) < 1e-6:
                continue
            px = float(transformed[0] / transformed[2])
            py = float(transformed[1] / transformed[2])
            if settings.viewport_only and not (
                screen_x - margin <= px <= screen_x + screen_w + margin
                and screen_y - margin <= py <= screen_y + screen_h + margin
            ):
                continue
            projected.append(ProjectedMarker(marker=marker, category=category, screen_x=px, screen_y=py))

        if settings.cluster_enabled:
            return _cluster_projected(projected, radius=max(20.0, float(settings.icon_size) * 1.6))
        return projected


def _matches_query(marker: MapMarker, category: MapCategory | None, query: str) -> bool:
    values = (
        marker.id,
        marker.title,
        marker.category_id,
        marker.region_id,
        category.display_name if category is not None else "",
        marker_display_name_zh(marker, category),
    )
    return any(query in str(value).lower() for value in values)


def _cluster_projected(markers: list[ProjectedMarker], *, radius: float) -> list[ProjectedItem]:
    clusters: list[list[ProjectedMarker]] = []
    radius_sq = radius * radius
    for marker in sorted(markers, key=lambda item: item.marker.id):
        target_cluster: list[ProjectedMarker] | None = None
        for cluster in clusters:
            center_x = sum(item.screen_x for item in cluster) / len(cluster)
            center_y = sum(item.screen_y for item in cluster) / len(cluster)
            dx = marker.screen_x - center_x
            dy = marker.screen_y - center_y
            if dx * dx + dy * dy <= radius_sq:
                target_cluster = cluster
                break
        if target_cluster is None:
            clusters.append([marker])
        else:
            target_cluster.append(marker)

    items: list[ProjectedItem] = []
    for cluster_markers in clusters:
        if len(cluster_markers) == 1:
            items.append(cluster_markers[0])
            continue
        x = sum(item.screen_x for item in cluster_markers) / len(cluster_markers)
        y = sum(item.screen_y for item in cluster_markers) / len(cluster_markers)
        items.append(MarkerCluster(markers=cluster_markers, screen_x=x, screen_y=y))
    return sorted(items, key=lambda item: (round(item.screen_y, 2), round(item.screen_x, 2)))
