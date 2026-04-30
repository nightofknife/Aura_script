from __future__ import annotations

from pathlib import Path
import unittest

from PIL import Image
from PIL import ImageDraw

from packages.yihuan_gui.map_overlay.presence_detector import BigMapPresenceDetector


DESKTOP = Path.home() / "Desktop"
MAIN_SCREENSHOT = DESKTOP / "QQ20260430-012752.png"
BIG_MAP_SCREENSHOT = DESKTOP / "QQ20260430-012857.png"
MAX_ZOOM_MAP_SCREENSHOT = Path("tmp/yihuan_map_overlay_live/current_game_map_zoom_max.png")


@unittest.skipUnless(
    MAIN_SCREENSHOT.is_file() and BIG_MAP_SCREENSHOT.is_file(),
    "Yihuan reference screenshots are not available on this machine.",
)
class TestYihuanBigMapPresenceDetector(unittest.TestCase):
    def setUp(self) -> None:
        self.detector = BigMapPresenceDetector()

    def test_main_game_screen_is_not_big_map_candidate(self):
        result = self.detector.detect(Image.open(MAIN_SCREENSHOT))

        self.assertFalse(result.is_candidate)
        self.assertLess(result.score, 0.72)
        self.assertEqual(result.reasons, [])
        self.assertFalse(result.debug["urban_fun_template"]["matched"])

    def test_big_map_screen_is_big_map_candidate(self):
        result = self.detector.detect(Image.open(BIG_MAP_SCREENSHOT))

        self.assertTrue(result.is_candidate)
        self.assertGreaterEqual(result.score, 0.72)
        self.assertEqual(result.reasons, ["urban_fun_template"])
        self.assertTrue(result.debug["urban_fun_template"]["matched"])

    def test_max_zoom_big_map_screen_is_big_map_candidate_when_available(self):
        if not MAX_ZOOM_MAP_SCREENSHOT.is_file():
            self.skipTest("Max-zoom Yihuan reference screenshot is not available on this machine.")

        result = self.detector.detect(Image.open(MAX_ZOOM_MAP_SCREENSHOT))

        self.assertTrue(result.is_candidate)
        self.assertGreaterEqual(result.score, 0.72)
        self.assertEqual(result.reasons, ["urban_fun_template"])
        self.assertTrue(result.debug["urban_fun_template"]["matched"])

    def test_big_map_client_crop_and_resize_still_match(self):
        image = Image.open(BIG_MAP_SCREENSHOT)
        client_crop = image.crop((26, 56, 1306, 802))
        resized = client_crop.resize((1024, 596))

        crop_result = self.detector.detect(client_crop)
        resized_result = self.detector.detect(resized)

        self.assertTrue(crop_result.is_candidate)
        self.assertEqual(crop_result.reasons, ["urban_fun_template"])
        self.assertTrue(resized_result.is_candidate)
        self.assertEqual(resized_result.reasons, ["urban_fun_template"])

    def test_big_map_still_matches_when_overlay_panel_covers_right_side(self):
        image = Image.open(BIG_MAP_SCREENSHOT).convert("RGB")
        draw = ImageDraw.Draw(image)
        width, height = image.size
        draw.rounded_rectangle(
            (int(width * 0.66), int(height * 0.04), int(width * 0.98), int(height * 0.88)),
            radius=14,
            fill=(248, 250, 252),
        )

        result = self.detector.detect(image)

        self.assertTrue(result.is_candidate)
        self.assertEqual(result.reasons, ["urban_fun_template"])

    def test_big_map_is_not_candidate_when_urban_fun_icon_is_covered(self):
        image = Image.open(BIG_MAP_SCREENSHOT).convert("RGB")
        draw = ImageDraw.Draw(image)
        width, height = image.size
        draw.rectangle(
            (0, int(height * 0.72), int(width * 0.18), height),
            fill=(0, 0, 0),
        )

        result = self.detector.detect(image)

        self.assertFalse(result.is_candidate)
        self.assertEqual(result.reasons, [])
        self.assertFalse(result.debug["urban_fun_template"]["matched"])

    def test_plain_dark_screen_is_not_big_map_candidate(self):
        image = Image.new("RGB", (1280, 720), (0, 0, 0))
        result = self.detector.detect(image)

        self.assertFalse(result.is_candidate)
        self.assertLess(result.score, 0.72)


if __name__ == "__main__":
    unittest.main()
