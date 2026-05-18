from __future__ import annotations

from typing import Any

from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMessageBox,
    QPlainTextEdit,
    QPushButton,
    QScrollArea,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)

from ..config_repository import QUICK_STOP_HOTKEY_OPTIONS
from ..logic import (
    CafeRunDefaults,
    CombatRunDefaults,
    FishingRunDefaults,
    GuiPreferences,
    MahjongRunDefaults,
    OneCafeRunDefaults,
    RhythmRunDefaults,
    TetrominoesRunDefaults,
    RuntimeSettings,
    TASK_AUTO_LOOP,
    TASK_CAFE_AUTO_LOOP,
    TASK_COMBAT_AUTO_LOOP,
    TASK_MAHJONG_AUTO_LOOP,
    TASK_ONE_CAFE_REVENUE_RESTOCK,
    TASK_PIANO_PLAY_MIDI,
    TASK_RHYTHM_AUTO_LOOP,
    TASK_TETROMINOES_AUTO_LOOP,
    build_settings_sections,
)


class SettingsPageMixin:
    def _build_settings_page(self) -> None:
        layout = QVBoxLayout(self._settings_page)
        scroll = QScrollArea(self._settings_page)
        scroll.setWidgetResizable(True)
        body = QWidget(scroll)
        body_layout = QVBoxLayout(body)
        scroll.setWidget(body)
        layout.addWidget(scroll, 1)

        runtime_group = QGroupBox("运行环境", body)
        runtime_form = QFormLayout(runtime_group)
        self._title_regex_edit = QLineEdit(runtime_group)
        self._exclude_titles_edit = QPlainTextEdit(runtime_group)
        self._exclude_titles_edit.setPlaceholderText("每行一个排除标题")
        self._allow_borderless_check = QCheckBox("允许匹配无边框窗口", runtime_group)
        self._capture_backend_combo = QComboBox(runtime_group)
        self._input_backend_combo = QComboBox(runtime_group)
        self._input_profile_combo = QComboBox(runtime_group)
        runtime_form.addRow("窗口标题匹配规则", self._title_regex_edit)
        runtime_form.addRow("排除窗口标题", self._exclude_titles_edit)
        runtime_form.addRow("", self._allow_borderless_check)
        runtime_form.addRow("截图后端", self._capture_backend_combo)
        runtime_form.addRow("输入后端", self._input_backend_combo)
        runtime_form.addRow("默认输入档案", self._input_profile_combo)
        body_layout.addWidget(runtime_group)

        defaults_group = QGroupBox("任务默认档案", body)
        defaults_form = QFormLayout(defaults_group)
        self._fishing_profile_combo = QComboBox(defaults_group)
        self._cafe_profile_combo = QComboBox(defaults_group)
        self._one_cafe_profile_combo = QComboBox(defaults_group)
        self._mahjong_profile_combo = QComboBox(defaults_group)
        self._combat_profile_combo = QComboBox(defaults_group)
        self._tetrominoes_profile_combo = QComboBox(defaults_group)
        self._rhythm_profile_combo = QComboBox(defaults_group)
        defaults_form.addRow("钓鱼识别档案", self._fishing_profile_combo)
        defaults_form.addRow("沙威玛识别档案", self._cafe_profile_combo)
        defaults_form.addRow("一咖舍识别档案", self._one_cafe_profile_combo)
        defaults_form.addRow("麻将识别档案", self._mahjong_profile_combo)
        defaults_form.addRow("自动战斗识别档案", self._combat_profile_combo)
        defaults_form.addRow("俄罗斯方块识别档案", self._tetrominoes_profile_combo)
        defaults_form.addRow("四键音游识别档案", self._rhythm_profile_combo)
        body_layout.addWidget(defaults_group)

        ui_group = QGroupBox("界面偏好", body)
        ui_form = QFormLayout(ui_group)
        self._history_limit_spin = QSpinBox(ui_group)
        self._history_limit_spin.setMinimum(1)
        self._history_limit_spin.setMaximum(500)
        self._auto_probe_check = QCheckBox("启动时自动执行运行时探针", ui_group)
        self._task_start_delay_spin = QSpinBox(ui_group)
        self._task_start_delay_spin.setMinimum(0)
        self._task_start_delay_spin.setMaximum(60)
        self._task_start_delay_spin.setToolTip("0 表示点击开始后立即提交任务。")
        self._quick_stop_hotkey_combo = QComboBox(ui_group)
        ui_form.addRow("历史记录显示条数", self._history_limit_spin)
        ui_form.addRow("", self._auto_probe_check)
        ui_form.addRow("任务启动延迟秒数（0 = 立即执行）", self._task_start_delay_spin)
        ui_form.addRow("快捷停止键", self._quick_stop_hotkey_combo)
        body_layout.addWidget(ui_group)

        action_row = QHBoxLayout()
        self._reload_settings_button = QPushButton("重新加载设置", body)
        self._reload_settings_button.clicked.connect(self._load_settings_widgets)
        self._save_settings_button = QPushButton("保存设置", body)
        self._save_settings_button.clicked.connect(self._save_settings)
        action_row.addWidget(self._reload_settings_button)
        action_row.addWidget(self._save_settings_button)
        action_row.addStretch(1)
        body_layout.addLayout(action_row)

        hint = QLabel(
            "设置页只负责运行环境、任务默认档案和界面偏好；玩法识别阈值、区域坐标与时序参数仍在各识别档案中维护。"
            " 自动弹钢琴的文件与演奏参数会在任务页自动记忆。",
            body,
        )
        hint.setWordWrap(True)
        body_layout.addWidget(hint)
        body_layout.addStretch(1)

    def _load_settings_widgets(self) -> None:
        self._runtime_settings = self._repo.get_runtime_settings()
        self._ui_preferences = self._repo.get_ui_preferences()
        available_profiles = self._repo.list_input_profiles()
        sections = build_settings_sections(self._runtime_settings, self._ui_preferences, available_profiles)
        runtime_section = next(section for section in sections if section.section_id == "runtime")
        ui_section = next(section for section in sections if section.section_id == "ui")
        runtime_map = {field.key: field for field in runtime_section.fields}
        ui_map = {field.key: field for field in ui_section.fields}

        self._title_regex_edit.setText(str(runtime_map["runtime.target.title_regex"].value))
        self._exclude_titles_edit.setPlainText(str(runtime_map["runtime.target.exclude_titles"].value))
        self._allow_borderless_check.setChecked(bool(runtime_map["runtime.target.allow_borderless"].value))
        self._set_combo_items(
            self._capture_backend_combo,
            runtime_map["runtime.capture.backend"].value,
            ["gdi", "dxgi", "wgc", "printwindow"],
        )
        self._set_combo_items(
            self._input_backend_combo,
            runtime_map["runtime.input.backend"].value,
            ["sendinput", "window_message"],
        )
        profile_field = runtime_map["input.profile"]
        self._set_combo_items(self._input_profile_combo, profile_field.value, list(profile_field.options))

        self._fishing_defaults = self._repo.get_fishing_defaults(self._task_rows.get(TASK_AUTO_LOOP))
        self._set_combo_items(
            self._fishing_profile_combo,
            self._fishing_defaults.profile_name,
            self._repo.list_fishing_profiles(),
        )
        self._sync_fishing_widgets_from_defaults()

        self._cafe_defaults = self._repo.get_cafe_defaults(self._task_rows.get(TASK_CAFE_AUTO_LOOP))
        self._set_combo_items(
            self._cafe_profile_combo,
            self._cafe_defaults.profile_name,
            self._repo.list_cafe_profiles(),
        )
        self._cafe_profile_label.setText(self._cafe_defaults.profile_name)
        self._cafe_max_seconds_spin.setValue(int(self._cafe_defaults.max_seconds))
        self._cafe_max_orders_spin.setValue(int(self._cafe_defaults.max_orders))
        self._cafe_start_game_check.setChecked(bool(self._cafe_defaults.start_game))
        self._cafe_wait_level_started_check.setChecked(bool(self._cafe_defaults.wait_level_started))
        self._cafe_full_assist_auto_hammer_check.setChecked(bool(self._cafe_defaults.full_assist_auto_hammer_mode))
        self._refresh_cafe_limit_hint()

        self._one_cafe_defaults = self._repo.get_one_cafe_defaults(
            self._task_rows.get(TASK_ONE_CAFE_REVENUE_RESTOCK)
        )
        self._set_combo_items(
            self._one_cafe_profile_combo,
            self._one_cafe_defaults.profile_name,
            self._repo.list_one_cafe_profiles(),
        )
        self._sync_one_cafe_widgets_from_defaults()

        self._mahjong_defaults = self._repo.get_mahjong_defaults(self._task_rows.get(TASK_MAHJONG_AUTO_LOOP))
        self._set_combo_items(
            self._mahjong_profile_combo,
            self._mahjong_defaults.profile_name,
            self._repo.list_mahjong_profiles(),
        )
        self._sync_mahjong_widgets_from_defaults()

        self._combat_defaults = self._repo.get_combat_defaults(self._task_rows.get(TASK_COMBAT_AUTO_LOOP))
        self._set_combo_items(
            self._combat_profile_combo,
            self._combat_defaults.profile_name,
            self._repo.list_combat_profiles(),
        )
        self._sync_combat_widgets_from_defaults()

        self._tetrominoes_defaults = self._repo.get_tetrominoes_defaults(self._task_rows.get(TASK_TETROMINOES_AUTO_LOOP))
        self._set_combo_items(
            self._tetrominoes_profile_combo,
            self._tetrominoes_defaults.profile_name,
            self._repo.list_tetrominoes_profiles(),
        )
        self._sync_tetrominoes_widgets_from_defaults()

        self._rhythm_defaults = self._repo.get_rhythm_defaults(self._task_rows.get(TASK_RHYTHM_AUTO_LOOP))
        self._set_combo_items(
            self._rhythm_profile_combo,
            self._rhythm_defaults.profile_name,
            self._repo.list_rhythm_profiles(),
        )
        self._sync_rhythm_widgets_from_defaults()

        self._piano_defaults = self._repo.get_piano_defaults(self._task_rows.get(TASK_PIANO_PLAY_MIDI))
        self._sync_piano_widgets_from_defaults()

        self._history_limit_spin.setValue(int(ui_map["gui.history_limit"].value))
        self._auto_probe_check.setChecked(bool(ui_map["gui.auto_runtime_probe_on_startup"].value))
        self._task_start_delay_spin.setValue(int(ui_map["gui.task_start_delay_sec"].value))
        self._set_combo_items(
            self._quick_stop_hotkey_combo,
            str(ui_map["gui.quick_stop_hotkey"].value).strip().upper(),
            list(QUICK_STOP_HOTKEY_OPTIONS),
        )
        self._stop_button.setText(f"停止任务（{self._ui_preferences.quick_stop_hotkey}）")

    @staticmethod
    def _set_combo_items(combo: QComboBox, current_value: Any, options: list[str]) -> None:
        combo.clear()
        seen: set[str] = set()
        for option in [str(current_value or "")] + [str(item) for item in options]:
            if not option or option in seen:
                continue
            combo.addItem(option, option)
            seen.add(option)
        index = combo.findData(str(current_value or ""))
        if index >= 0:
            combo.setCurrentIndex(index)

    def _save_settings(self) -> None:
        runtime_settings = RuntimeSettings(
            title_regex=self._title_regex_edit.text().strip(),
            exclude_titles=self._repo.exclude_titles_from_text(self._exclude_titles_edit.toPlainText()),
            allow_borderless=self._allow_borderless_check.isChecked(),
            capture_backend=str(self._capture_backend_combo.currentData() or self._capture_backend_combo.currentText()),
            input_backend=str(self._input_backend_combo.currentData() or self._input_backend_combo.currentText()),
            input_profile=str(self._input_profile_combo.currentData() or self._input_profile_combo.currentText()),
        )
        ui_preferences = GuiPreferences(
            history_limit=int(self._history_limit_spin.value()),
            auto_runtime_probe_on_startup=self._auto_probe_check.isChecked(),
            expand_developer_tools=self._ui_preferences.expand_developer_tools,
            task_start_delay_sec=int(self._task_start_delay_spin.value()),
            quick_stop_hotkey=str(
                self._quick_stop_hotkey_combo.currentData() or self._quick_stop_hotkey_combo.currentText()
            ).strip()
            .upper(),
        )
        fishing_defaults = FishingRunDefaults(
            profile_name=str(self._fishing_profile_combo.currentData() or self._fishing_profile_combo.currentText())
        )
        cafe_defaults = CafeRunDefaults(
            profile_name=str(self._cafe_profile_combo.currentData() or self._cafe_profile_combo.currentText()),
            max_seconds=self._cafe_defaults.max_seconds,
            max_orders=self._cafe_defaults.max_orders,
            start_game=self._cafe_defaults.start_game,
            wait_level_started=self._cafe_defaults.wait_level_started,
            full_assist_auto_hammer_mode=self._cafe_defaults.full_assist_auto_hammer_mode,
            min_order_interval_sec=self._cafe_defaults.min_order_interval_sec,
            min_order_duration_sec=self._cafe_defaults.min_order_duration_sec,
        )
        one_cafe_defaults = OneCafeRunDefaults(
            profile_name=str(
                self._one_cafe_profile_combo.currentData() or self._one_cafe_profile_combo.currentText()
            ),
            withdraw_enabled=self._one_cafe_defaults.withdraw_enabled,
            restock_enabled=self._one_cafe_defaults.restock_enabled,
            restock_hours=self._one_cafe_defaults.restock_hours,
        )
        mahjong_defaults = MahjongRunDefaults(
            profile_name=str(self._mahjong_profile_combo.currentData() or self._mahjong_profile_combo.currentText()),
            max_seconds=self._mahjong_defaults.max_seconds,
            start_game=self._mahjong_defaults.start_game,
            auto_hu=self._mahjong_defaults.auto_hu,
            auto_peng=self._mahjong_defaults.auto_peng,
            auto_discard=self._mahjong_defaults.auto_discard,
        )
        combat_defaults = CombatRunDefaults(
            profile_name=str(self._combat_profile_combo.currentData() or self._combat_profile_combo.currentText()),
            strategy_name=self._combat_defaults.strategy_name,
            max_seconds=self._combat_defaults.max_seconds,
            max_encounters=self._combat_defaults.max_encounters,
            auto_target=self._combat_defaults.auto_target,
            auto_dodge=self._combat_defaults.auto_dodge,
            debug_enabled=self._combat_defaults.debug_enabled,
            capture_debug_enabled=self._combat_defaults.capture_debug_enabled,
            capture_interval_sec=self._combat_defaults.capture_interval_sec,
            capture_max_images=self._combat_defaults.capture_max_images,
            capture_raw_enabled=self._combat_defaults.capture_raw_enabled,
        )
        tetrominoes_defaults = TetrominoesRunDefaults(
            profile_name=str(
                self._tetrominoes_profile_combo.currentData() or self._tetrominoes_profile_combo.currentText()
            ),
            max_seconds=self._tetrominoes_defaults.max_seconds,
            max_pieces=self._tetrominoes_defaults.max_pieces,
            start_game=self._tetrominoes_defaults.start_game,
        )
        rhythm_defaults = RhythmRunDefaults(
            profile_name=str(self._rhythm_profile_combo.currentData() or self._rhythm_profile_combo.currentText()),
            loop_count=self._rhythm_defaults.loop_count,
            max_seconds=self._rhythm_defaults.max_seconds,
            start_game=self._rhythm_defaults.start_game,
            close_result=self._rhythm_defaults.close_result,
            lane_keys=self._rhythm_defaults.lane_keys,
            lane_y_offset_px=self._rhythm_defaults.lane_y_offset_px,
            debug_enabled=self._rhythm_defaults.debug_enabled,
        )

        try:
            self._repo.update_runtime_settings(runtime_settings)
            self._repo.update_fishing_defaults(fishing_defaults)
            self._repo.update_cafe_defaults(cafe_defaults)
            self._repo.update_one_cafe_defaults(one_cafe_defaults)
            self._repo.update_mahjong_defaults(mahjong_defaults)
            self._repo.update_combat_defaults(combat_defaults)
            self._repo.update_tetrominoes_defaults(tetrominoes_defaults)
            self._repo.update_rhythm_defaults(rhythm_defaults)
            self._repo.save_ui_preferences(ui_preferences)
        except Exception as exc:  # noqa: BLE001
            QMessageBox.warning(self, "保存设置失败", str(exc))
            return

        self._runtime_settings = runtime_settings
        self._ui_preferences = ui_preferences
        self._fishing_defaults = self._repo.get_fishing_defaults(self._task_rows.get(TASK_AUTO_LOOP))
        self._cafe_defaults = self._repo.get_cafe_defaults(self._task_rows.get(TASK_CAFE_AUTO_LOOP))
        self._one_cafe_defaults = self._repo.get_one_cafe_defaults(
            self._task_rows.get(TASK_ONE_CAFE_REVENUE_RESTOCK)
        )
        self._mahjong_defaults = self._repo.get_mahjong_defaults(self._task_rows.get(TASK_MAHJONG_AUTO_LOOP))
        self._combat_defaults = self._repo.get_combat_defaults(self._task_rows.get(TASK_COMBAT_AUTO_LOOP))
        self._tetrominoes_defaults = self._repo.get_tetrominoes_defaults(self._task_rows.get(TASK_TETROMINOES_AUTO_LOOP))
        self._rhythm_defaults = self._repo.get_rhythm_defaults(self._task_rows.get(TASK_RHYTHM_AUTO_LOOP))
        self._sync_fishing_widgets_from_defaults()
        self._cafe_profile_label.setText(self._cafe_defaults.profile_name)
        self._sync_one_cafe_widgets_from_defaults()
        self._sync_mahjong_widgets_from_defaults()
        self._sync_combat_widgets_from_defaults()
        self._sync_tetrominoes_widgets_from_defaults()
        self._sync_rhythm_widgets_from_defaults()
        self._refresh_cafe_limit_hint()
        self._install_quick_stop_hotkey(ui_preferences.quick_stop_hotkey, show_warning=True)
        self.request_apply_preferences.emit(ui_preferences)
        self.request_refresh_history.emit()
        self.statusBar().showMessage("设置已保存，新配置将在后续任务和探针中生效。", 6000)
        self._append_log("设置已保存，新配置将在后续任务和探针中生效。")
        self._apply_task_guard()
