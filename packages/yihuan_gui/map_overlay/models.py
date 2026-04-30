from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any


CATEGORY_TRANSLATIONS_ZH: dict[str, str] = {
    "fast-travel": "传送点",
    "locker": "储物柜",
    "real-estate": "房产/住宅",
    "city-services": "城市服务",
    "featured-business": "商店与餐厅",
    "oracle-stone": "谕石",
    "currencies": "货币/掉落",
    "collectible": "收集物",
    "stealable-loot": "可拾取/可窃取物资",
    "quest-start": "任务",
    "monsters": "怪物",
    "activities": "活动",
}

MARKER_TITLE_TRANSLATIONS_ZH: dict[str, str] = {
    "Fast Travel Point": "传送点",
    "Taxi Stop": "出租车站",
    "ReroRero Phone Booth": "ReroRero 电话亭",
    "Oracle Stone": "谕石",
    "Gift from \"21\"": "21的赠礼",
    "Wertheimer Tower": "海默塔",
    "Storage Locker": "储物柜",
    "Housing": "住宅",
    "City Service": "城市服务",
    "Shop": "商店",
    "Restaurant": "餐厅",
    "Currency Drop": "货币/掉落",
    "Collectible": "收集物",
    "Stealable Loot": "可拾取/可窃取物资",
    "Quest": "任务",
    "Monster": "怪物",
    "Activity": "活动",
    "2-Four Convenience Store": "2-Four 便利店",
    "Bamboo Pharmacy": "Bamboo 药房",
    "Nekomaru Ramen": "猫丸拉面",
    "Crazy Cat": "Crazy Cat",
    "Food First": "Food First 餐饮",
    "Uncle Prime": "Prime 叔叔",
    "Veggie Shop": "蔬果店",
    "Moby-Dick Bookstore": "白鲸书店",
    "Hillside Blooms": "山坡花坊",
    "Bigmouth Baozi": "大嘴包子",
    "Puka Candy": "Puka 糖果店",
    "Alice's Bakery": "爱丽丝烘焙坊",
    "Felicità Gelato": "Felicità 意式冰淇淋",
    "Marigny Pizza": "Marigny 披萨",
    "Security Office": "治安署",
    "Train Station": "列车站",
    "Leon Estate Group": "Leon 地产集团",
    "The Cafe by Origen": "Origen 咖啡馆",
    "Pink Paws Bank Branch": "粉爪银行支行",
    "Pink Paws Bank HQ": "粉爪银行总部",
    "Midas Arc Workshop": "Midas Arc 工坊",
    "Gold Apple Collection Hall": "金苹果收藏馆",
    "The Witch's House": "魔女之家",
    "Clement Academy": "克莱门特学院",
    "Sterry Express": "Sterry 快递",
    "Hethereau Municipal Hospital": "赫瑟罗市立医院",
    "Hethereau Skytower": "赫瑟罗天塔",
    "Detention Facility": "拘留所",
    "Ebisu Auction House": "惠比寿拍卖行",
    "City Delivery": "城市配送",
    "Your New Electric Scooter?": "你的新电动滑板车？",
    "Friendship Self-Service": "友情自助服务",
    "Illegal Activity": "异象委托",
    "Coastal Drift": "海岸漂移",
    "Full Throttle": "全速前进",
    "Ghost Remnant": "幽灵残迹",
}


def marker_display_name_zh(marker: "MapMarker", category: "MapCategory | None" = None) -> str:
    title = str(marker.title or marker.id or "").strip()
    if not title:
        return category.display_name if category is not None else str(marker.id)
    if title in MARKER_TITLE_TRANSLATIONS_ZH:
        return MARKER_TITLE_TRANSLATIONS_ZH[title]
    if title.replace("?", "").strip("/") == "":
        return "未知点位"
    if category is not None and title == category.name_en:
        return category.display_name
    return title


@dataclass(frozen=True)
class MapBounds:
    min_x: float
    max_x: float
    min_y: float
    max_y: float

    @property
    def width(self) -> float:
        return self.max_x - self.min_x

    @property
    def height(self) -> float:
        return self.max_y - self.min_y


@dataclass(frozen=True)
class TileItem:
    id: str
    url: str
    bounds: MapBounds
    col: int
    row: int
    asset_id: str = ""


@dataclass(frozen=True)
class MapCategory:
    id: str
    name_en: str
    name_zh: str
    total: int = 0
    default_visible: bool = False
    icon_url: str = ""

    @property
    def display_name(self) -> str:
        return self.name_zh or self.name_en or self.id


@dataclass(frozen=True)
class MapRegion:
    id: str
    name: str


@dataclass(frozen=True)
class MapMarker:
    id: str
    category_id: str
    title: str
    region_id: str
    map_x: float
    map_y: float
    icon_url: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)

    def tile_pixel(self, bounds: MapBounds) -> tuple[float, float]:
        """Return ZeroLuck stitched-map pixel coordinates for this marker."""
        return self.map_x - bounds.min_x, bounds.max_y - self.map_y


@dataclass(frozen=True)
class ZeroluckMapData:
    categories: list[MapCategory]
    regions: list[MapRegion]
    markers: list[MapMarker]
    bounds: MapBounds
    tile_size: int
    cols: int
    rows: int
    tiles: list[TileItem]
    fetched_at: datetime | None = None
    using_cache: bool = False
    reference_image_path: str = ""
    reference_scale: float = 1.0
    reference_full_size: tuple[int, int] | None = None

    @property
    def categories_by_id(self) -> dict[str, MapCategory]:
        return {category.id: category for category in self.categories}

    @property
    def regions_by_id(self) -> dict[str, MapRegion]:
        return {region.id: region for region in self.regions}

    @property
    def marker_count(self) -> int:
        return len(self.markers)


@dataclass(frozen=True)
class MapMatchResult:
    success: bool
    confidence: float
    transform: Any | None = None
    visible_polygon: list[tuple[float, float]] = field(default_factory=list)
    screen_rect: tuple[int, int, int, int] = (0, 0, 0, 0)
    message: str = ""
    debug_info: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class ProjectedMarker:
    marker: MapMarker
    category: MapCategory | None
    screen_x: float
    screen_y: float


@dataclass(frozen=True)
class MarkerCluster:
    markers: list[ProjectedMarker]
    screen_x: float
    screen_y: float

    @property
    def count(self) -> int:
        return len(self.markers)
