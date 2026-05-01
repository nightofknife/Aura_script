from __future__ import annotations

from pathlib import Path
import unittest

import yaml


class TestYihuanOneCafeTask(unittest.TestCase):
    def setUp(self) -> None:
        self.repo_root = Path(__file__).resolve().parents[1]
        self.task_path = self.repo_root / "plans" / "yihuan" / "tasks" / "one_cafe" / "revenue_restock.yaml"
        self.manifest_path = self.repo_root / "plans" / "yihuan" / "manifest.yaml"
        self.config_path = self.repo_root / "plans" / "yihuan" / "config.yaml"
        self.template_dir = self.repo_root / "plans" / "yihuan" / "data" / "one_cafe" / "default_1280x720_cn"
        self.task = yaml.safe_load(self.task_path.read_text(encoding="utf-8"))["revenue_restock"]
        self.manifest = yaml.safe_load(self.manifest_path.read_text(encoding="utf-8"))
        self.config = yaml.safe_load(self.config_path.read_text(encoding="utf-8"))

    def test_one_cafe_uses_only_existing_generic_actions(self):
        allowed_actions = {
            "plans/aura_base/press_key",
            "plans/aura_base/wait_for_image",
            "plans/aura_base/find_image_and_click",
            "plans/aura_base/wait_for_any_template_in_set",
            "plans/aura_base/wait_for_templates_in_set_to_disappear",
            "plans/aura_base/click",
            "plans/aura_base/move_to",
            "plans/aura_base/mouse_down",
            "plans/aura_base/mouse_up",
            "plans/aura_base/log",
            "plans/aura_base/sleep",
        }

        actions = {step["action"] for step in self.task["steps"].values()}

        self.assertLessEqual(actions, allowed_actions)
        self.assertNotIn("yihuan_one_cafe_run_session", actions)
        self.assertNotIn("yihuan_one_cafe_summarize_result", actions)

    def test_one_cafe_entry_uses_fixed_click_with_success_detection(self):
        steps = self.task["steps"]

        self.assertEqual(steps["wait_city_tycoon_map"]["params"]["template"], "data/one_cafe/{{ inputs.profile_name }}/city_tycoon_title.png")
        self.assertEqual(steps["move_to_one_cafe_marker"]["action"], "plans/aura_base/move_to")
        self.assertEqual(steps["move_to_one_cafe_marker"]["params"], {"x": 530, "y": 545, "duration": 0.25})
        self.assertEqual(steps["hold_one_cafe_marker_down"]["action"], "plans/aura_base/mouse_down")
        self.assertEqual(steps["hold_one_cafe_marker"]["action"], "plans/aura_base/sleep")
        self.assertGreaterEqual(steps["hold_one_cafe_marker"]["params"]["seconds"], 0.35)
        self.assertEqual(steps["release_one_cafe_marker"]["action"], "plans/aura_base/mouse_up")
        self.assertEqual(steps["detect_shop_management_after_first_click"]["depends_on"], "pause_after_one_cafe_marker")
        self.assertEqual(steps["detect_shop_management_after_first_click"]["params"]["timeout"], 0)
        self.assertEqual(steps["detect_city_map_after_first_click"]["when"], "{{ not nodes.detect_shop_management_after_first_click.output.found }}")
        self.assertEqual(steps["detect_city_map_after_first_click"]["params"]["timeout"], 0)
        self.assertEqual(
            steps["wait_shop_management_after_marker"]["when"],
            "{{ (not nodes.detect_shop_management_after_first_click.output.found) and (not nodes.detect_city_map_after_first_click.output.found) }}",
        )
        self.assertEqual(steps["wait_shop_management_after_marker"]["params"]["timeout"], 4)
        self.assertIn("nodes.detect_city_map_after_first_click.output.found", steps["retry_move_to_one_cafe_marker"]["when"])
        self.assertEqual(steps["retry_hold_one_cafe_marker_down"]["action"], "plans/aura_base/mouse_down")
        self.assertEqual(steps["retry_hold_one_cafe_marker"]["action"], "plans/aura_base/sleep")
        self.assertGreaterEqual(steps["retry_hold_one_cafe_marker"]["params"]["seconds"], 0.35)
        self.assertEqual(steps["retry_release_one_cafe_marker"]["action"], "plans/aura_base/mouse_up")
        self.assertEqual(steps["wait_shop_management"]["depends_on"], {"pause_after_retry_one_cafe_marker": "success|skipped"})
        self.assertEqual(steps["wait_shop_management"]["on_result"]["retry_when"], "{{ not result.found }}")

    def test_one_cafe_withdraw_and_restock_match_confirmed_flow(self):
        steps = self.task["steps"]

        self.assertEqual(steps["click_withdraw"]["params"]["template"], "data/one_cafe/{{ inputs.profile_name }}/withdraw_button.png")
        self.assertEqual(steps["detect_no_revenue_toast"]["params"]["template"], "data/one_cafe/{{ inputs.profile_name }}/no_revenue_toast.png")
        self.assertEqual(steps["detect_withdraw_report"]["params"]["template"], "data/one_cafe/{{ inputs.profile_name }}/withdraw_report_title.png")
        self.assertEqual(steps["wait_reward_popup"]["params"]["template"], "data/one_cafe/{{ inputs.profile_name }}/reward_popup.png")
        self.assertEqual(steps["close_reward_popup"]["action"], "plans/aura_base/click")
        self.assertEqual(steps["close_reward_popup"]["params"], {"x": 640, "y": 650})
        self.assertEqual(
            steps["wait_restock_hours"]["params"]["templates_ref"],
            "data/one_cafe/{{ inputs.profile_name }}/restock_{{ inputs.restock_hours }}h_*.png",
        )
        self.assertEqual(steps["wait_restock_hours"]["params"]["region"], [260, 535, 380, 70])
        self.assertEqual(steps["detect_delivery_prompt"]["params"]["template"], "data/one_cafe/{{ inputs.profile_name }}/delivery_prompt.png")
        self.assertEqual(steps["wait_delivery_cost_prompt"]["params"]["template"], "data/one_cafe/{{ inputs.profile_name }}/delivery_cost_prompt.png")

    def test_restock_hour_templates_cover_selected_and_unselected_states(self):
        expected = {
            "restock_4h_selected.png",
            "restock_4h_unselected.png",
            "restock_24h_selected.png",
            "restock_24h_unselected.png",
            "restock_72h_selected.png",
            "restock_72h_unselected.png",
        }

        existing = {path.name for path in self.template_dir.glob("restock_*h_*.png")}

        self.assertLessEqual(expected, existing)

    def test_click_timing_is_slower_for_game_ui(self):
        steps = self.task["steps"]
        input_config = self.config["runtime"]["input"]

        self.assertGreaterEqual(input_config["mouse_move_duration_ms"], 200)
        self.assertGreaterEqual(input_config["click_post_delay_ms"], 100)
        self.assertGreaterEqual(input_config["key_interval_ms"], 100)

        for name, step in steps.items():
            if step["action"] == "plans/aura_base/find_image_and_click":
                self.assertGreaterEqual(step["params"]["move_duration"], 0.3, name)

        expected_pause_steps = {
            "pause_after_one_cafe_marker",
            "pause_after_withdraw_click",
            "pause_after_confirm_withdraw_report",
            "pause_after_close_reward_popup",
            "pause_after_restock_entry_click",
            "pause_after_restock_hours_click",
            "pause_after_inventory_restock_click",
            "pause_after_delivery_button_click",
            "pause_after_delivery_cost_confirm",
            "pause_before_exit_to_city_map",
            "pause_after_first_exit_esc",
            "pause_after_city_tycoon_close",
        }

        self.assertLessEqual(expected_pause_steps, set(steps))
        for name in expected_pause_steps:
            self.assertEqual(steps[name]["action"], "plans/aura_base/sleep")
            self.assertGreaterEqual(steps[name]["params"]["seconds"], 0.7, name)

    def test_one_cafe_exit_returns_to_world_scene(self):
        steps = self.task["steps"]

        self.assertEqual(steps["exit_to_city_map_first_esc"]["params"]["key"], "esc")
        self.assertEqual(steps["exit_to_city_map_second_esc"]["when"], "{{ not nodes.detect_city_map_after_first_esc.output.found }}")
        self.assertEqual(steps["click_city_tycoon_close"]["params"], {"x": 1222, "y": 42})
        self.assertNotIn("exit_world_retry_esc", steps)
        self.assertEqual(steps["wait_city_tycoon_disappear_after_close"]["action"], "plans/aura_base/wait_for_templates_in_set_to_disappear")
        self.assertEqual(
            steps["wait_city_tycoon_disappear_after_close"]["params"]["templates_ref"],
            "data/one_cafe/{{ inputs.profile_name }}/city_tycoon_title.png",
        )
        self.assertEqual(steps["wait_world_scene"]["depends_on"], "wait_city_tycoon_disappear_after_close")
        self.assertEqual(steps["wait_world_scene"]["params"]["template"], "data/one_cafe/{{ inputs.profile_name }}/world_hud.png")
        self.assertEqual(
            self.task["returns"]["returned_to_world"],
            "{{ nodes.wait_world_scene.output.found or nodes.wait_city_tycoon_disappear_after_close.output }}",
        )

    def test_manifest_exports_one_cafe_task_but_no_one_cafe_action(self):
        task_ids = {item["id"] for item in self.manifest["exports"]["tasks"]}
        action_names = {item["name"] for item in self.manifest["exports"]["actions"]}

        self.assertIn("one_cafe/revenue_restock", task_ids)
        self.assertIn("one_cafe/revenue_restock/revenue_restock", task_ids)
        self.assertNotIn("yihuan_one_cafe_run_session", action_names)
        self.assertNotIn("yihuan_one_cafe_summarize_result", action_names)
