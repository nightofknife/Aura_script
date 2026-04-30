from __future__ import annotations

import html
import json
import os
import re
import time
import urllib.error
import urllib.request
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Mapping

from PIL import Image

from .models import (
    CATEGORY_TRANSLATIONS_ZH,
    MapBounds,
    MapCategory,
    MapMarker,
    MapRegion,
    TileItem,
    ZeroluckMapData,
)


INTERACTIVE_MAP_URL = "https://zeroluck.gg/nte/interactive-map/"
STARTER_MARKERS_URL = "https://zeroluck.gg/nte/data/interactive-map/starter-markers.en.json"
CATEGORY_MARKERS_URL = "https://zeroluck.gg/nte/data/interactive-map/categories/{category_id}.en.json"
CACHE_MAX_AGE = timedelta(days=1)
BUNDLED_DATA_DIR = Path(__file__).resolve().parent / "data"
BUNDLED_SNAPSHOT_PATH = BUNDLED_DATA_DIR / "zeroluck_nte_snapshot.json"


class ZeroluckDataError(RuntimeError):
    """Raised when ZeroLuck data cannot be loaded from network or cache."""


class ZeroluckRepository:
    def __init__(self, cache_dir: Path | str | None = None, *, timeout_sec: float = 20.0) -> None:
        self.cache_dir = Path(cache_dir) if cache_dir is not None else _default_cache_dir()
        self.timeout_sec = float(timeout_sec)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.tiles_dir.mkdir(parents=True, exist_ok=True)

    @property
    def normalized_cache_path(self) -> Path:
        return self.cache_dir / "normalized-map-data.json"

    @property
    def basemap_gray_path(self) -> Path:
        return self.cache_dir / "basemap-gray.png"

    @property
    def tiles_dir(self) -> Path:
        return self.cache_dir / "tiles"

    def load(self, *, force_refresh: bool = False, allow_network: bool = False) -> ZeroluckMapData:
        """Load the bundled complete ZeroLuck snapshot by default.

        Runtime overlay use is intentionally offline. ``allow_network`` is only
        for maintenance tools that explicitly refresh the snapshot/cache.
        """
        if not force_refresh:
            bundled = self.load_bundled()
            if bundled is not None:
                return bundled

        if not force_refresh and self._cache_is_fresh():
            cached = self.load_cached()
            if cached is not None:
                return cached

        if not allow_network:
            cached = self.load_cached()
            if cached is not None:
                return cached
            raise ZeroluckDataError("缺少内置 ZeroLuck 快照，且当前禁用了运行时联网获取。")

        try:
            return self.refresh()
        except Exception as exc:  # noqa: BLE001
            cached = self.load_cached(mark_using_cache=True)
            if cached is not None:
                return cached
            raise ZeroluckDataError(f"无法获取 ZeroLuck 地图数据：{exc}") from exc

    def load_bundled(self) -> ZeroluckMapData | None:
        if not BUNDLED_SNAPSHOT_PATH.is_file():
            return None
        try:
            payload = json.loads(BUNDLED_SNAPSHOT_PATH.read_text(encoding="utf-8"))
            data = self._map_data_from_json(payload)
            return _replace_cache_flag(data, using_cache=False)
        except Exception:
            return None

    def load_cached(self, *, mark_using_cache: bool = True) -> ZeroluckMapData | None:
        if not self.normalized_cache_path.is_file():
            return None
        try:
            payload = json.loads(self.normalized_cache_path.read_text(encoding="utf-8"))
            data = self._map_data_from_json(payload)
            return _replace_cache_flag(data, using_cache=mark_using_cache)
        except Exception:
            return None

    def refresh(self) -> ZeroluckMapData:
        initial_payload = self._fetch_initial_payload()
        map_payload = _find_initial_map_payload(initial_payload)
        if not map_payload:
            raise ZeroluckDataError("ZeroLuck 页面中未找到互动地图数据。")

        categories = _parse_categories(map_payload.get("categories") or [])
        regions = _parse_regions(map_payload.get("regions") or [])
        bounds, tile_size, cols, rows, tiles = _parse_tiles(map_payload)
        markers = self._fetch_all_markers(categories)
        data = ZeroluckMapData(
            categories=categories,
            regions=regions,
            markers=markers,
            bounds=bounds,
            tile_size=tile_size,
            cols=cols,
            rows=rows,
            tiles=tiles,
            fetched_at=datetime.now(timezone.utc),
            using_cache=False,
        )
        self.normalized_cache_path.write_text(
            json.dumps(self._map_data_to_json(data), ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        return data

    def ensure_basemap_gray(self, data: ZeroluckMapData) -> Path:
        if data.reference_image_path:
            reference_path = BUNDLED_DATA_DIR / data.reference_image_path
            if reference_path.is_file():
                return reference_path

        expected_width = int(round(data.bounds.width))
        expected_height = int(round(data.bounds.height))
        if self.basemap_gray_path.is_file():
            try:
                with Image.open(self.basemap_gray_path) as existing:
                    if existing.size == (expected_width, expected_height):
                        return self.basemap_gray_path
            except Exception:
                pass

        image = Image.new("L", (expected_width, expected_height), 0)
        for tile in data.tiles:
            try:
                tile_path = self._ensure_tile(tile)
                with Image.open(tile_path) as tile_image:
                    gray_tile = tile_image.convert("L").resize((data.tile_size, data.tile_size))
                    image.paste(gray_tile, (tile.col * data.tile_size, tile.row * data.tile_size))
            except Exception:
                # Missing tiles are left black; the matcher still has enough signal in populated areas.
                continue
        image.save(self.basemap_gray_path)
        return self.basemap_gray_path

    def _fetch_initial_payload(self) -> Mapping[str, Any]:
        raw_html = self._fetch_text(INTERACTIVE_MAP_URL)
        match = re.search(
            r'<script id="__NEXT_DATA__" type="application/json">(.*?)</script>',
            raw_html,
            re.DOTALL,
        )
        if not match:
            raise ZeroluckDataError("ZeroLuck 页面缺少 __NEXT_DATA__。")
        return json.loads(html.unescape(match.group(1)))

    def _fetch_all_markers(self, categories: list[MapCategory]) -> list[MapMarker]:
        markers: list[MapMarker] = []
        seen: set[str] = set()
        for category in categories:
            payload: Mapping[str, Any] | None = None
            try:
                payload = self._fetch_json(CATEGORY_MARKERS_URL.format(category_id=category.id))
            except Exception:
                payload = None
            for marker in _parse_marker_payload(payload or {}, fallback_category_id=category.id):
                if marker.id not in seen:
                    markers.append(marker)
                    seen.add(marker.id)

        if markers:
            return markers

        starter_payload = self._fetch_json(STARTER_MARKERS_URL)
        return _parse_marker_payload(starter_payload, fallback_category_id="")

    def _ensure_tile(self, tile: TileItem) -> Path:
        suffix = Path(tile.url.split("?", 1)[0]).suffix or ".png"
        name = re.sub(r"[^A-Za-z0-9_.-]+", "_", tile.asset_id or tile.id) + suffix
        path = self.tiles_dir / name
        if path.is_file() and path.stat().st_size > 0:
            return path
        request = urllib.request.Request(tile.url, headers={"User-Agent": "Aura-Yihuan-MapOverlay/1.0"})
        with urllib.request.urlopen(request, timeout=self.timeout_sec) as response:
            path.write_bytes(response.read())
        time.sleep(0.01)
        return path

    def _fetch_json(self, url: str) -> Mapping[str, Any]:
        return json.loads(self._fetch_text(url))

    def _fetch_text(self, url: str) -> str:
        request = urllib.request.Request(url, headers={"User-Agent": "Aura-Yihuan-MapOverlay/1.0"})
        try:
            with urllib.request.urlopen(request, timeout=self.timeout_sec) as response:
                return response.read().decode("utf-8")
        except urllib.error.HTTPError as exc:
            raise ZeroluckDataError(f"{url} 返回 HTTP {exc.code}") from exc

    def _cache_is_fresh(self) -> bool:
        if not self.normalized_cache_path.is_file():
            return False
        age = datetime.now(timezone.utc) - datetime.fromtimestamp(
            self.normalized_cache_path.stat().st_mtime,
            tz=timezone.utc,
        )
        return age <= CACHE_MAX_AGE

    @staticmethod
    def _map_data_to_json(data: ZeroluckMapData) -> dict[str, Any]:
        return {
            "categories": [category.__dict__ for category in data.categories],
            "regions": [region.__dict__ for region in data.regions],
            "markers": [
                {
                    "id": marker.id,
                    "category_id": marker.category_id,
                    "title": marker.title,
                    "region_id": marker.region_id,
                    "map_x": marker.map_x,
                    "map_y": marker.map_y,
                    "icon_url": marker.icon_url,
                    "metadata": marker.metadata,
                }
                for marker in data.markers
            ],
            "bounds": data.bounds.__dict__,
            "tile_size": data.tile_size,
            "cols": data.cols,
            "rows": data.rows,
            "tiles": [
                {
                    "id": tile.id,
                    "url": tile.url,
                    "bounds": tile.bounds.__dict__,
                    "col": tile.col,
                    "row": tile.row,
                    "asset_id": tile.asset_id,
                }
                for tile in data.tiles
            ],
            "fetched_at": data.fetched_at.isoformat() if data.fetched_at else None,
            "reference_image": {
                "path": data.reference_image_path,
                "scale": data.reference_scale,
                "full_width": data.reference_full_size[0] if data.reference_full_size else int(data.bounds.width),
                "full_height": data.reference_full_size[1] if data.reference_full_size else int(data.bounds.height),
            },
        }

    @staticmethod
    def _map_data_from_json(payload: Mapping[str, Any]) -> ZeroluckMapData:
        bounds = _bounds_from_mapping(payload.get("bounds") or {})
        fetched_at_raw = payload.get("fetched_at") or payload.get("generated_at")
        fetched_at = None
        if fetched_at_raw:
            try:
                fetched_at = datetime.fromisoformat(str(fetched_at_raw))
            except ValueError:
                fetched_at = None
        return ZeroluckMapData(
            categories=[
                MapCategory(
                    id=str(item.get("id") or ""),
                    name_en=str(item.get("name_en") or ""),
                    name_zh=str(
                        CATEGORY_TRANSLATIONS_ZH.get(str(item.get("id") or ""))
                        or item.get("name_zh")
                        or item.get("name_en")
                        or item.get("id")
                        or ""
                    ),
                    total=int(item.get("total") or 0),
                    default_visible=bool(item.get("default_visible")),
                    icon_url=str(item.get("icon_url") or ""),
                )
                for item in payload.get("categories") or []
            ],
            regions=[
                MapRegion(id=str(item.get("id") or ""), name=str(item.get("name") or item.get("id") or ""))
                for item in payload.get("regions") or []
            ],
            markers=[
                MapMarker(
                    id=str(item.get("id") or ""),
                    category_id=str(item.get("category_id") or ""),
                    title=str(item.get("title") or item.get("id") or ""),
                    region_id=str(item.get("region_id") or ""),
                    map_x=float(item.get("map_x") or 0),
                    map_y=float(item.get("map_y") or 0),
                    icon_url=str(item.get("icon_url") or ""),
                    metadata=dict(item.get("metadata") or {}),
                )
                for item in payload.get("markers") or []
            ],
            bounds=bounds,
            tile_size=int(payload.get("tile_size") or 256),
            cols=int(payload.get("cols") or 0),
            rows=int(payload.get("rows") or 0),
            tiles=[
                TileItem(
                    id=str(item.get("id") or ""),
                    url=str(item.get("url") or ""),
                    bounds=_bounds_from_mapping(item.get("bounds") or {}),
                    col=int(item.get("col") or 0),
                    row=int(item.get("row") or 0),
                    asset_id=str(item.get("asset_id") or ""),
                )
                for item in payload.get("tiles") or []
            ],
            fetched_at=fetched_at,
            using_cache=True,
            reference_image_path=str(
                (payload.get("reference_image") or {}).get("path") or payload.get("reference_image_path") or ""
            ),
            reference_scale=float(
                (payload.get("reference_image") or {}).get("scale") or payload.get("reference_scale") or 1.0
            ),
            reference_full_size=_reference_full_size(payload),
        )


def _default_cache_dir() -> Path:
    if os.name == "nt":
        root = Path(os.environ.get("LOCALAPPDATA") or Path.home() / "AppData" / "Local")
        return root / "Aura" / "YihuanGui" / "zeroluck_nte"
    return Path(os.environ.get("XDG_CACHE_HOME") or Path.home() / ".cache") / "aura" / "yihuan_gui" / "zeroluck_nte"


def _find_initial_map_payload(payload: Mapping[str, Any]) -> Mapping[str, Any] | None:
    stack: list[Any] = [payload]
    while stack:
        current = stack.pop()
        if isinstance(current, Mapping):
            if "categories" in current and "map" in current:
                return current
            stack.extend(current.values())
        elif isinstance(current, list):
            stack.extend(current)
    return None


def _parse_categories(items: list[Any]) -> list[MapCategory]:
    categories: list[MapCategory] = []
    for item in items:
        if not isinstance(item, Mapping):
            continue
        category_id = str(item.get("id") or "").strip()
        if not category_id:
            continue
        name_en = str(item.get("label") or item.get("name") or item.get("title") or category_id).strip()
        categories.append(
            MapCategory(
                id=category_id,
                name_en=name_en,
                name_zh=CATEGORY_TRANSLATIONS_ZH.get(category_id, name_en),
                total=int(item.get("total") or item.get("count") or 0),
                default_visible=bool(item.get("defaultVisible") or item.get("default_visible")),
                icon_url=str(item.get("icon_url") or item.get("iconUrl") or ""),
            )
        )
    return categories


def _parse_regions(items: list[Any]) -> list[MapRegion]:
    regions: list[MapRegion] = []
    for item in items:
        if not isinstance(item, Mapping):
            continue
        region_id = str(item.get("id") or "").strip()
        if region_id:
            regions.append(MapRegion(id=region_id, name=str(item.get("label") or item.get("name") or region_id)))
    return regions


def _parse_tiles(map_payload: Mapping[str, Any]) -> tuple[MapBounds, int, int, int, list[TileItem]]:
    map_info = dict(map_payload.get("map") or {})
    tiles_info = dict(map_info.get("tiles") or {})
    bounds = _bounds_from_mapping(tiles_info.get("bounds") or {})
    tile_size = int(tiles_info.get("tileSize") or tiles_info.get("tile_size") or 256)
    cols = int(tiles_info.get("cols") or round(bounds.width / tile_size))
    rows = int(tiles_info.get("rows") or round(bounds.height / tile_size))
    tiles: list[TileItem] = []
    for item in tiles_info.get("items") or []:
        if not isinstance(item, Mapping):
            continue
        tile_bounds = _bounds_from_mapping(item.get("bounds") or {})
        col = int(round((tile_bounds.min_x - bounds.min_x) / tile_size))
        row = int(round((bounds.max_y - tile_bounds.max_y) / tile_size))
        url = str(item.get("url") or "")
        if not url:
            continue
        tiles.append(
            TileItem(
                id=str(item.get("id") or f"tile-{col}-{row}"),
                url=url,
                bounds=tile_bounds,
                col=col,
                row=row,
                asset_id=str(item.get("assetId") or item.get("asset_id") or ""),
            )
        )
    return bounds, tile_size, cols, rows, tiles


def _parse_marker_payload(payload: Mapping[str, Any], *, fallback_category_id: str) -> list[MapMarker]:
    markers: list[MapMarker] = []
    raw_markers = payload.get("markers") or payload.get("items") or []
    for item in raw_markers:
        if not isinstance(item, Mapping):
            continue
        position = dict(item.get("position") or {})
        map_position = dict(position.get("map") or {})
        try:
            map_x = float(map_position.get("x"))
            map_y = float(map_position.get("y"))
        except (TypeError, ValueError):
            continue
        marker_id = str(item.get("id") or "").strip()
        category_id = str(item.get("categoryId") or item.get("category_id") or fallback_category_id).strip()
        if not marker_id:
            marker_id = f"{category_id}:{map_x:.2f}:{map_y:.2f}"
        markers.append(
            MapMarker(
                id=marker_id,
                category_id=category_id,
                title=str(item.get("title") or item.get("name") or marker_id),
                region_id=str(item.get("regionId") or item.get("region_id") or ""),
                map_x=map_x,
                map_y=map_y,
                icon_url=str(item.get("icon_url") or item.get("iconUrl") or ""),
                metadata={key: value for key, value in item.items() if key not in {"position"}},
            )
        )
    return markers


def _bounds_from_mapping(payload: Mapping[str, Any]) -> MapBounds:
    return MapBounds(
        min_x=float(payload.get("minX") if "minX" in payload else payload.get("min_x", 0)),
        max_x=float(payload.get("maxX") if "maxX" in payload else payload.get("max_x", 0)),
        min_y=float(payload.get("minY") if "minY" in payload else payload.get("min_y", 0)),
        max_y=float(payload.get("maxY") if "maxY" in payload else payload.get("max_y", 0)),
    )


def _replace_cache_flag(data: ZeroluckMapData, *, using_cache: bool) -> ZeroluckMapData:
    return ZeroluckMapData(
        categories=data.categories,
        regions=data.regions,
        markers=data.markers,
        bounds=data.bounds,
        tile_size=data.tile_size,
        cols=data.cols,
        rows=data.rows,
        tiles=data.tiles,
        fetched_at=data.fetched_at,
        using_cache=using_cache,
        reference_image_path=data.reference_image_path,
        reference_scale=data.reference_scale,
        reference_full_size=data.reference_full_size,
    )


def _reference_full_size(payload: Mapping[str, Any]) -> tuple[int, int] | None:
    reference = payload.get("reference_image") or {}
    try:
        width = int(reference.get("full_width") or payload.get("reference_full_width"))
        height = int(reference.get("full_height") or payload.get("reference_full_height"))
    except (AttributeError, TypeError, ValueError):
        return None
    if width <= 0 or height <= 0:
        return None
    return width, height
