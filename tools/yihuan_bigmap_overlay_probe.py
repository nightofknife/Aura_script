from __future__ import annotations

import argparse
from pathlib import Path
import sys

from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from packages.yihuan_gui.map_overlay.marker_projector import MarkerProjector
from packages.yihuan_gui.map_overlay.models import MarkerCluster, ProjectedMarker
from packages.yihuan_gui.map_overlay.settings import load_map_overlay_settings
from packages.yihuan_gui.map_overlay.zeroluck_repository import ZeroluckRepository
from packages.yihuan_gui.map_overlay.bigmap_locator import BigMapLocator


def main() -> int:
    parser = argparse.ArgumentParser(description="Probe Yihuan big-map overlay matching with an offline screenshot.")
    parser.add_argument("screenshot", type=Path, help="Path to a game big-map screenshot.")
    parser.add_argument("-o", "--output", type=Path, default=Path("tmp/yihuan_bigmap_overlay_probe.png"))
    parser.add_argument("--labels", action="store_true", help="Draw marker labels for non-clustered markers.")
    args = parser.parse_args()

    if not args.screenshot.is_file():
        parser.error(f"screenshot does not exist: {args.screenshot}")

    repo = ZeroluckRepository()
    data = repo.load()
    reference = repo.ensure_basemap_gray(data)
    screenshot = Image.open(args.screenshot).convert("RGB")
    match = BigMapLocator().locate(screenshot, reference, reference_map_size=data.reference_full_size)
    output = screenshot.copy()
    draw = ImageDraw.Draw(output)

    if match.success:
        settings = load_map_overlay_settings(None, data.categories)
        items = MarkerProjector().project(data, match, settings)
        for item in items:
            if isinstance(item, MarkerCluster):
                radius = 16
                x = round(item.screen_x)
                y = round(item.screen_y)
                draw.ellipse((x - radius, y - radius, x + radius, y + radius), fill=(255, 216, 77), outline=(20, 20, 20), width=3)
                draw.text((x - 6, y - 7), str(item.count), fill=(20, 24, 32))
            elif isinstance(item, ProjectedMarker):
                radius = 8
                x = round(item.screen_x)
                y = round(item.screen_y)
                draw.ellipse((x - radius, y - radius, x + radius, y + radius), fill=(38, 183, 255), outline=(0, 0, 0), width=2)
                if args.labels:
                    draw.text((x + 10, y - 5), item.marker.title, fill=(255, 255, 255), stroke_width=2, stroke_fill=(0, 0, 0))
        status = f"matched: confidence={match.confidence:.3f}, items={len(items)}, {match.debug_info}"
    else:
        status = f"match failed: {match.message}, {match.debug_info}"

    try:
        font = ImageFont.truetype("arial.ttf", 18)
    except Exception:
        font = None
    draw.rectangle((12, 12, min(output.width - 12, 1120), 52), fill=(0, 0, 0, 170))
    draw.text((22, 22), status, fill=(255, 255, 255), font=font)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    output.save(args.output)
    print(status)
    print(args.output)
    return 0 if match.success else 2


if __name__ == "__main__":
    raise SystemExit(main())
