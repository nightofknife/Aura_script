from __future__ import annotations

from pathlib import Path
from typing import Any

from PySide6.QtWidgets import QFileDialog

from ..logic import TASK_PIANO_PLAY_MIDI, PianoRunDefaults, piano_conflict_policy_display_name


class DefaultsSyncMixin:
    def _sync_fishing_widgets_from_defaults(self) -> None:
        self._fishing_profile_label.setText(self._fishing_defaults.profile_name)
        self._sell_fish_every_rounds_spin.setValue(int(self._fishing_defaults.sell_fish_every_rounds))
        self._bait_buy_repeat_count_spin.setValue(int(self._fishing_defaults.bait_buy_repeat_count))
        self._sell_before_buy_bait_check.setChecked(bool(self._fishing_defaults.sell_before_buy_bait))

    def _sync_one_cafe_widgets_from_defaults(self) -> None:
        self._one_cafe_profile_label.setText(self._one_cafe_defaults.profile_name)
        self._one_cafe_withdraw_check.setChecked(bool(self._one_cafe_defaults.withdraw_enabled))
        self._one_cafe_restock_check.setChecked(bool(self._one_cafe_defaults.restock_enabled))
        index = self._one_cafe_restock_hours_combo.findData(int(self._one_cafe_defaults.restock_hours))
        if index < 0:
            index = self._one_cafe_restock_hours_combo.findData(24)
        if index >= 0:
            self._one_cafe_restock_hours_combo.setCurrentIndex(index)

    def _sync_mahjong_widgets_from_defaults(self) -> None:
        self._mahjong_profile_label.setText(self._mahjong_defaults.profile_name)
        self._mahjong_max_seconds_spin.setValue(int(self._mahjong_defaults.max_seconds))
        self._mahjong_start_game_check.setChecked(bool(self._mahjong_defaults.start_game))
        self._mahjong_auto_hu_check.setChecked(bool(self._mahjong_defaults.auto_hu))
        self._mahjong_auto_peng_check.setChecked(bool(self._mahjong_defaults.auto_peng))
        self._mahjong_auto_discard_check.setChecked(bool(self._mahjong_defaults.auto_discard))

    def _sync_combat_widgets_from_defaults(self) -> None:
        self._combat_profile_label.setText(self._combat_defaults.profile_name)
        self._set_combo_items(
            self._combat_strategy_combo,
            self._combat_defaults.strategy_name,
            self._repo.list_combat_strategy_names(self._combat_defaults.profile_name),
        )
        self._combat_max_seconds_spin.setValue(int(self._combat_defaults.max_seconds))
        self._combat_max_encounters_spin.setValue(int(self._combat_defaults.max_encounters))
        self._combat_auto_target_check.setChecked(bool(self._combat_defaults.auto_target))
        self._combat_auto_dodge_check.setChecked(bool(self._combat_defaults.auto_dodge))
        self._combat_debug_enabled_check.setChecked(bool(self._combat_defaults.debug_enabled))
        self._combat_capture_debug_enabled_check.setChecked(bool(self._combat_defaults.capture_debug_enabled))
        self._combat_capture_interval_spin.setValue(float(self._combat_defaults.capture_interval_sec))
        self._combat_capture_max_images_spin.setValue(int(self._combat_defaults.capture_max_images))
        self._combat_capture_raw_enabled_check.setChecked(bool(self._combat_defaults.capture_raw_enabled))
        self._sync_combat_capture_widgets_enabled()

    def _sync_tetrominoes_widgets_from_defaults(self) -> None:
        self._tetrominoes_profile_label.setText(self._tetrominoes_defaults.profile_name)
        self._tetrominoes_max_seconds_spin.setValue(int(self._tetrominoes_defaults.max_seconds))
        self._tetrominoes_max_pieces_spin.setValue(int(self._tetrominoes_defaults.max_pieces))
        self._tetrominoes_start_game_check.setChecked(bool(self._tetrominoes_defaults.start_game))

    def _sync_piano_widgets_from_defaults(self) -> None:
        widgets = (
            self._piano_file_edit,
            self._piano_recent_combo,
            self._piano_conflict_policy_combo,
            self._piano_tempo_scale_spin,
            self._piano_start_delay_spin,
            self._piano_transpose_spin,
            self._piano_roll_note_spin,
            self._piano_velocity_threshold_spin,
            self._piano_focus_window_check,
            self._piano_dry_run_check,
        )
        for widget in widgets:
            widget.blockSignals(True)
        try:
            self._piano_file_edit.setText(self._piano_defaults.file_path)
            self._set_piano_recent_files(self._piano_defaults.recent_files, self._piano_defaults.file_path)
            index = self._piano_conflict_policy_combo.findData(self._piano_defaults.conflict_policy)
            if index < 0:
                index = 0
            self._piano_conflict_policy_combo.setCurrentIndex(index)
            self._piano_tempo_scale_spin.setValue(float(self._piano_defaults.tempo_scale))
            self._piano_start_delay_spin.setValue(int(self._piano_defaults.start_delay_ms))
            self._piano_transpose_spin.setValue(int(self._piano_defaults.transpose_semitones))
            self._piano_roll_note_spin.setValue(int(self._piano_defaults.roll_note_ms))
            self._piano_velocity_threshold_spin.setValue(int(self._piano_defaults.velocity_threshold))
            self._piano_focus_window_check.setChecked(bool(self._piano_defaults.focus_window))
            self._piano_dry_run_check.setChecked(bool(self._piano_defaults.dry_run))
        finally:
            for widget in widgets:
                widget.blockSignals(False)
        self._refresh_piano_file_status_label()
        self._sync_piano_roll_note_widget_enabled()

    def _set_piano_recent_files(self, recent_files: tuple[str, ...], current_file: str = "") -> None:
        self._piano_recent_combo.blockSignals(True)
        try:
            self._piano_recent_combo.clear()
            self._piano_recent_combo.addItem("最近使用的 MIDI 文件", "")
            seen: set[str] = set()
            for path in [str(current_file).strip(), *recent_files]:
                normalized = str(path or "").strip()
                if not normalized or normalized in seen:
                    continue
                self._piano_recent_combo.addItem(normalized, normalized)
                seen.add(normalized)
            self._piano_recent_combo.setCurrentIndex(0)
        finally:
            self._piano_recent_combo.blockSignals(False)

    def _refresh_piano_file_status_label(self) -> None:
        raw_path = self._piano_file_edit.text().strip()
        if not raw_path:
            self._piano_file_status_label.setText("尚未选择 MIDI 文件。")
            return
        try:
            resolved = self._resolve_piano_file_path(raw_path, require_exists=True)
        except ValueError as exc:
            self._piano_file_status_label.setText(str(exc))
            return
        self._piano_file_status_label.setText(f"已选择：{resolved.name}")

    @staticmethod
    def _resolve_piano_file_path(file_path: str, *, require_exists: bool) -> Path:
        normalized = str(file_path or "").strip()
        if not normalized:
            raise ValueError("请选择 MIDI 文件。")
        candidate = Path(normalized).expanduser()
        if not candidate.is_absolute():
            candidate = Path.cwd() / candidate
        candidate = candidate.resolve()
        if candidate.suffix.lower() not in {".mid", ".midi"}:
            raise ValueError("请选择 .mid 或 .midi 文件。")
        if require_exists and not candidate.is_file():
            raise ValueError(f"MIDI 文件不存在：{candidate}")
        return candidate

    def _current_piano_defaults_from_widgets(self, *, resolved_file_path: str | None = None) -> PianoRunDefaults:
        file_path = str(resolved_file_path or self._piano_file_edit.text().strip())
        recent_files: list[str] = []
        if resolved_file_path:
            recent_files.append(str(resolved_file_path))
        recent_files.extend(self._piano_defaults.recent_files)

        last_directory = str(self._piano_defaults.last_directory or "")
        if resolved_file_path:
            last_directory = str(Path(resolved_file_path).parent)
        elif file_path:
            try:
                last_directory = str(self._resolve_piano_file_path(file_path, require_exists=False).parent)
            except ValueError:
                pass

        return PianoRunDefaults(
            file_path=file_path,
            recent_files=tuple(recent_files),
            last_directory=last_directory,
            conflict_policy=str(
                self._piano_conflict_policy_combo.currentData() or self._piano_conflict_policy_combo.currentText()
            ).strip().lower(),
            transpose_semitones=int(self._piano_transpose_spin.value()),
            tempo_scale=float(self._piano_tempo_scale_spin.value()),
            start_delay_ms=int(self._piano_start_delay_spin.value()),
            roll_note_ms=int(self._piano_roll_note_spin.value()),
            velocity_threshold=int(self._piano_velocity_threshold_spin.value()),
            focus_window=self._piano_focus_window_check.isChecked(),
            dry_run=self._piano_dry_run_check.isChecked(),
        )

    def _persist_piano_defaults_from_widgets(self, *, resolved_file_path: str | None = None) -> None:
        defaults = self._current_piano_defaults_from_widgets(resolved_file_path=resolved_file_path)
        self._repo.save_piano_defaults(defaults)
        self._piano_defaults = self._repo.get_piano_defaults(self._task_rows.get(TASK_PIANO_PLAY_MIDI))
        self._set_piano_recent_files(self._piano_defaults.recent_files, self._piano_file_edit.text().strip())
        self._refresh_piano_file_status_label()
        self._sync_piano_roll_note_widget_enabled()
        self._apply_task_guard()

    def _sync_piano_roll_note_widget_enabled(self) -> None:
        is_roll = str(self._piano_conflict_policy_combo.currentData() or "").strip().lower() == "roll"
        self._piano_roll_note_spin.setEnabled(is_roll)

    def _browse_piano_midi_file(self) -> None:
        initial_dir = str(self._piano_defaults.last_directory or Path.cwd())
        selected_path, _selected_filter = QFileDialog.getOpenFileName(
            self,
            "选择 MIDI 文件",
            initial_dir,
            "MIDI 文件 (*.mid *.midi);;所有文件 (*.*)",
        )
        if not selected_path:
            return
        self._piano_file_edit.setText(selected_path)
        self._persist_piano_defaults_from_widgets(resolved_file_path=str(Path(selected_path).resolve()))

    def _on_piano_recent_file_selected(self, index: int) -> None:
        if index <= 0:
            return
        file_path = str(self._piano_recent_combo.itemData(index) or "").strip()
        if not file_path:
            return
        self._piano_file_edit.setText(file_path)
        self._persist_piano_defaults_from_widgets(resolved_file_path=file_path)

    def _on_piano_file_editing_finished(self) -> None:
        self._persist_piano_defaults_from_widgets()

    def _on_piano_defaults_widget_changed(self, *_args: object) -> None:
        self._persist_piano_defaults_from_widgets()

    def _sync_combat_capture_widgets_enabled(self, *_args: object) -> None:
        enabled = bool(self._combat_capture_debug_enabled_check.isChecked())
        self._combat_capture_interval_spin.setEnabled(enabled)
        self._combat_capture_max_images_spin.setEnabled(enabled)
        self._combat_capture_raw_enabled_check.setEnabled(enabled)

    def _refresh_cafe_limit_hint(self) -> None:
        runtime_defaults = self._repo.get_cafe_profile_runtime_defaults(self._cafe_defaults.profile_name)
        default_seconds = runtime_defaults.get("max_seconds")
        if default_seconds is None:
            default_text = "当前档案默认运行时长未知"
        else:
            seconds_text = int(default_seconds) if float(default_seconds).is_integer() else round(default_seconds, 1)
            default_text = f"当前档案默认运行时长：{seconds_text} 秒"

        interval_text = self._format_optional_seconds(runtime_defaults.get("min_order_interval_sec"))
        duration_text = self._format_optional_seconds(runtime_defaults.get("min_order_duration_sec"))
        fake_customer_text = "开启" if bool(runtime_defaults.get("fake_customer_enabled", True)) else "关闭"
        order_guard_text = "开启" if bool(runtime_defaults.get("fake_customer_order_guard_enabled", True)) else "关闭"
        self._cafe_limit_hint_label.setText(
            f"{default_text}；订单间隔：{interval_text}；单订单最短耗时：{duration_text}；"
            f"假顾客驱赶：{fake_customer_text}；订单守卫：{order_guard_text}。"
            "最大运行秒数填 0 表示不按时间停止；最大订单数填 0 表示不按订单数停止。"
        )

    @staticmethod
    def _format_optional_seconds(value: Any) -> str:
        if value is None:
            return "未配置"
        try:
            number = float(value)
        except (TypeError, ValueError):
            return str(value)
        if number <= 0:
            return "不限制"
        return f"{number:.3f} 秒"
