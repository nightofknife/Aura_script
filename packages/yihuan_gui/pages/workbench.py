from __future__ import annotations

from pathlib import Path
from typing import Any

from PySide6.QtCore import QSize, Qt
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QFileDialog,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QListWidget,
    QListWidgetItem,
    QMessageBox,
    QPlainTextEdit,
    QPushButton,
    QSplitter,
    QSpinBox,
    QStackedWidget,
    QTabWidget,
    QTextBrowser,
    QVBoxLayout,
    QWidget,
)

from ..logic import (
    TASK_AUTO_LOOP,
    TASK_CAFE_AUTO_LOOP,
    TASK_COMBAT_AUTO_LOOP,
    TASK_MAHJONG_AUTO_LOOP,
    TASK_ONE_CAFE_REVENUE_RESTOCK,
    TASK_PIANO_PLAY_MIDI,
    TASK_RHYTHM_AUTO_LOOP,
    TASK_TETROMINOES_AUTO_LOOP,
    build_auto_loop_inputs,
    build_cafe_loop_inputs,
    build_combat_loop_inputs,
    build_mahjong_loop_inputs,
    build_one_cafe_inputs,
    build_piano_play_midi_inputs,
    build_rhythm_loop_inputs,
    build_tetrominoes_loop_inputs,
    is_runtime_interacting_task,
    task_display_name,
    task_is_enabled,
)
from ..models import PendingLaunch
from ..task_specs import WORKBENCH_TASK_GROUPS, WORKBENCH_TASKS
from ..widgets.run_detail import RunDetailViewer


class WorkbenchPageMixin:
    def _build_workbench_page(self) -> None:
        layout = QVBoxLayout(self._workbench_page)
        layout.setContentsMargins(12, 10, 12, 12)
        layout.setSpacing(10)
        layout.addWidget(self._build_status_bar_widget())

        splitter = QSplitter(Qt.Horizontal, self._workbench_page)
        layout.addWidget(splitter, 1)

        splitter.addWidget(self._build_task_column(splitter))
        splitter.addWidget(self._build_parameter_column(splitter))
        splitter.addWidget(self._build_log_column(splitter))
        splitter.setStretchFactor(0, 0)
        splitter.setStretchFactor(1, 3)
        splitter.setStretchFactor(2, 2)
        splitter.setSizes([240, 760, 470])
        self._select_task_id("one_cafe")

    def _build_status_bar_widget(self) -> QWidget:
        group = QWidget(self._workbench_page)
        group.setObjectName("topStatusBar")
        row = QHBoxLayout(group)
        row.setContentsMargins(14, 10, 14, 10)
        self._admin_label = QLabel("-", group)
        self._runner_label = QLabel("框架启动中", group)
        self._runtime_summary_label = QLabel("运行时探针：等待探测", group)
        self._last_event_label = QLabel("最近事件：-", group)
        self._last_error_label = QLabel("最近错误：-", group)
        self._runtime_summary_label.setWordWrap(True)
        self._last_error_label.setWordWrap(True)
        title = QLabel("AURA 控制台", group)
        title.setObjectName("sectionTitle")
        row.addWidget(title)
        row.addSpacing(14)
        row.addWidget(QLabel("管理员", group))
        row.addWidget(self._admin_label)
        row.addSpacing(12)
        row.addWidget(QLabel("框架", group))
        row.addWidget(self._runner_label)
        row.addSpacing(12)
        row.addWidget(self._runtime_summary_label, 1)
        row.addWidget(self._last_event_label, 1)
        row.addWidget(self._last_error_label, 1)
        self._refresh_probe_button = QPushButton("刷新探针", group)
        self._refresh_probe_button.clicked.connect(self._request_refresh_runtime_probe_from_ui)
        refresh_plan_button = QPushButton("刷新方案包", group)
        refresh_plan_button.clicked.connect(self.request_refresh_plan_info.emit)
        row.addWidget(self._refresh_probe_button)
        row.addWidget(refresh_plan_button)
        return group

    def _build_task_column(self, parent: QWidget) -> QWidget:
        panel = QWidget(parent)
        layout = QVBoxLayout(panel)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(8)

        title = QLabel("任务导航", panel)
        title.setObjectName("sectionTitle")
        layout.addWidget(title)

        self._task_list = QListWidget(panel)
        self._task_list.setObjectName("taskNav")
        self._task_list.setWordWrap(True)
        for group_name, task_ids in WORKBENCH_TASK_GROUPS:
            group_item = QListWidgetItem(group_name)
            group_item.setData(Qt.UserRole, "")
            group_item.setFlags(group_item.flags() & ~Qt.ItemIsEnabled & ~Qt.ItemIsSelectable)
            group_item.setSizeHint(QSize(0, 32))
            self._task_list.addItem(group_item)
            for task_id in task_ids:
                spec = WORKBENCH_TASKS[task_id]
                item = QListWidgetItem(self._task_nav_item_text(task_id, "等待加载"))
                item.setData(Qt.UserRole, task_id)
                item.setToolTip(f"{spec['category']} / {spec['kind']}：{spec['description']}")
                item.setSizeHint(QSize(0, 72))
                self._task_list.addItem(item)
        self._task_list.currentItemChanged.connect(self._on_task_item_changed)
        layout.addWidget(self._task_list, 1)

        self._selected_task_status_label = QLabel("等待任务列表加载。", panel)
        self._selected_task_status_label.setObjectName("mutedText")
        self._selected_task_status_label.setWordWrap(True)
        layout.addWidget(self._selected_task_status_label)

        return panel

    def _build_parameter_column(self, parent: QWidget) -> QWidget:
        panel = QWidget(parent)
        layout = QVBoxLayout(panel)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(10)

        header = QWidget(panel)
        header.setObjectName("taskCard")
        header_layout = QVBoxLayout(header)
        header_layout.setContentsMargins(18, 14, 18, 14)
        meta_row = QHBoxLayout()
        self._task_category_label = QLabel("小游戏", panel)
        self._task_category_label.setObjectName("categoryBadge")
        self._task_kind_label = QLabel("循环采集", panel)
        self._task_kind_label.setObjectName("kindBadge")
        meta_row.addWidget(self._task_category_label)
        meta_row.addWidget(self._task_kind_label)
        meta_row.addStretch(1)
        title_row = QHBoxLayout()
        self._task_title_label = QLabel("钓鱼", panel)
        self._task_title_label.setObjectName("pageTitle")
        self._task_badge_label = QLabel("框架启动中", panel)
        self._task_badge_label.setObjectName("statusBadge")
        title_row.addWidget(self._task_title_label)
        title_row.addStretch(1)
        title_row.addWidget(self._task_badge_label)
        self._task_description_label = QLabel("", panel)
        self._task_description_label.setObjectName("mutedText")
        self._task_description_label.setWordWrap(True)
        self._task_preflight_label = QLabel("运行前确认：请确保游戏窗口已打开，并确认当前识别档案适用于 1280x720。", panel)
        self._task_preflight_label.setObjectName("mutedText")
        self._task_preflight_label.setWordWrap(True)
        header_layout.addLayout(meta_row)
        header_layout.addLayout(title_row)
        header_layout.addWidget(self._task_description_label)
        header_layout.addWidget(self._task_preflight_label)
        layout.addWidget(header)

        self._parameter_stack = QStackedWidget(panel)
        self._parameter_pages: dict[str, QWidget] = {}
        self._parameter_pages["fishing"] = self._build_fishing_parameters(self._parameter_stack)
        self._parameter_pages["cafe"] = self._build_cafe_parameters(self._parameter_stack)
        self._parameter_pages["one_cafe"] = self._build_one_cafe_parameters(self._parameter_stack)
        self._parameter_pages["mahjong"] = self._build_mahjong_parameters(self._parameter_stack)
        self._parameter_pages["combat"] = self._build_combat_parameters(self._parameter_stack)
        self._parameter_pages["tetrominoes"] = self._build_tetrominoes_parameters(self._parameter_stack)
        self._parameter_pages["rhythm"] = self._build_rhythm_parameters(self._parameter_stack)
        self._parameter_pages["piano"] = self._build_piano_parameters(self._parameter_stack)
        for task_id in WORKBENCH_TASKS:
            self._parameter_stack.addWidget(self._parameter_pages[task_id])
        layout.addWidget(self._parameter_stack, 1)

        action_bar = QWidget(panel)
        action_bar.setObjectName("actionBar")
        action_layout = QHBoxLayout(action_bar)
        action_layout.setContentsMargins(14, 10, 14, 10)
        self._action_status_label = QLabel("空闲", action_bar)
        self._action_status_label.setObjectName("mutedText")
        self._start_button = QPushButton("开始执行", action_bar)
        self._start_button.setObjectName("primaryButton")
        self._start_button.clicked.connect(self._request_start_selected_task)
        self._stop_button = QPushButton(f"停止任务（{self._ui_preferences.quick_stop_hotkey}）", action_bar)
        self._stop_button.setObjectName("dangerButton")
        self._stop_button.clicked.connect(self._request_stop_current_task)
        action_layout.addWidget(self._action_status_label, 1)
        action_layout.addWidget(self._start_button)
        action_layout.addWidget(self._stop_button)
        layout.addWidget(action_bar)
        return panel

    def _build_fishing_parameters(self, parent: QWidget) -> QWidget:
        page = QWidget(parent)
        layout = QVBoxLayout(page)

        group = QGroupBox("钓鱼参数", page)
        form = QFormLayout(group)
        self._fishing_profile_label = QLabel("-", group)
        self._max_rounds_spin = QSpinBox(group)
        self._max_rounds_spin.setMinimum(0)
        self._max_rounds_spin.setMaximum(99999)
        self._max_rounds_spin.setValue(0)
        self._max_rounds_spin.setToolTip("0 表示不限制轮数，直到任务自身停止。")
        self._sell_fish_every_rounds_spin = QSpinBox(group)
        self._sell_fish_every_rounds_spin.setMinimum(0)
        self._sell_fish_every_rounds_spin.setMaximum(99999)
        self._sell_fish_every_rounds_spin.setValue(0)
        self._sell_fish_every_rounds_spin.setToolTip("0 表示不自动卖鱼；大于 0 时每隔指定轮数尝试卖鱼。")
        self._bait_buy_repeat_count_spin = QSpinBox(group)
        self._bait_buy_repeat_count_spin.setMinimum(0)
        self._bait_buy_repeat_count_spin.setMaximum(99999)
        self._bait_buy_repeat_count_spin.setValue(1)
        self._bait_buy_repeat_count_spin.setToolTip("0 表示缺饵时不自动购买；大于 0 时表示每次买鱼饵流程重复尝试的次数。")
        self._sell_before_buy_bait_check = QCheckBox("买鱼饵前先尝试卖鱼", group)
        form.addRow("钓鱼识别档案", self._fishing_profile_label)
        form.addRow("最大轮数（0 = 不限制轮数）", self._max_rounds_spin)
        form.addRow("卖鱼间隔轮数（0 = 不自动卖鱼）", self._sell_fish_every_rounds_spin)
        form.addRow("买鱼饵重复次数（0 = 不自动购买）", self._bait_buy_repeat_count_spin)
        form.addRow("", self._sell_before_buy_bait_check)
        layout.addWidget(group)

        hint = QLabel("只在这里放本次运行最常用的快捷参数；识别档案请到设置界面调整。", page)
        hint.setWordWrap(True)
        layout.addWidget(hint)
        layout.addStretch(1)
        return page

    def _build_cafe_parameters(self, parent: QWidget) -> QWidget:
        page = QWidget(parent)
        layout = QVBoxLayout(page)

        group = QGroupBox("沙威玛参数", page)
        form = QFormLayout(group)
        self._cafe_profile_label = QLabel("-", group)
        self._cafe_max_seconds_spin = QSpinBox(group)
        self._cafe_max_seconds_spin.setMinimum(0)
        self._cafe_max_seconds_spin.setMaximum(99999)
        self._cafe_max_seconds_spin.setToolTip("0 表示不按运行时间停止。")
        self._cafe_max_orders_spin = QSpinBox(group)
        self._cafe_max_orders_spin.setMinimum(0)
        self._cafe_max_orders_spin.setMaximum(99999)
        self._cafe_max_orders_spin.setToolTip("0 表示不按订单数量停止。")
        self._cafe_start_game_check = QCheckBox("自动点击开始游戏", group)
        self._cafe_wait_level_started_check = QCheckBox("等待关卡开始", group)
        self._cafe_full_assist_auto_hammer_check = QCheckBox("全辅助自动锤子模式", group)
        self._cafe_limit_hint_label = QLabel("", group)
        self._cafe_limit_hint_label.setWordWrap(True)
        form.addRow("沙威玛识别档案", self._cafe_profile_label)
        form.addRow("最大运行秒数（0 = 不按时间停止）", self._cafe_max_seconds_spin)
        form.addRow("最大订单数（0 = 不按订单数停止）", self._cafe_max_orders_spin)
        form.addRow("", self._cafe_start_game_check)
        form.addRow("", self._cafe_wait_level_started_check)
        form.addRow("", self._cafe_full_assist_auto_hammer_check)
        form.addRow("档案说明", self._cafe_limit_hint_label)
        layout.addWidget(group)
        layout.addStretch(1)
        return page

    def _build_one_cafe_parameters(self, parent: QWidget) -> QWidget:
        page = QWidget(parent)
        layout = QVBoxLayout(page)

        group = QGroupBox("一咖舍参数", page)
        form = QFormLayout(group)
        self._one_cafe_profile_label = QLabel("-", group)
        self._one_cafe_withdraw_check = QCheckBox("领取收益", group)
        self._one_cafe_restock_check = QCheckBox("执行补货", group)
        self._one_cafe_restock_hours_combo = QComboBox(group)
        for hours in (4, 24, 72):
            self._one_cafe_restock_hours_combo.addItem(f"{hours} 小时", hours)
        form.addRow("一咖舍识别档案", self._one_cafe_profile_label)
        form.addRow("", self._one_cafe_withdraw_check)
        form.addRow("", self._one_cafe_restock_check)
        form.addRow("补货时长", self._one_cafe_restock_hours_combo)
        layout.addWidget(group)

        hint = QLabel(
            "任务会按 F5 打开都市大亨，进入一咖舍后执行收益领取和补货，完成后尝试返回世界场景。"
            "识别档案请到设置界面调整。",
            page,
        )
        hint.setWordWrap(True)
        layout.addWidget(hint)
        layout.addStretch(1)
        return page

    def _build_mahjong_parameters(self, parent: QWidget) -> QWidget:
        page = QWidget(parent)
        layout = QVBoxLayout(page)

        group = QGroupBox("麻将参数", page)
        form = QFormLayout(group)
        self._mahjong_profile_label = QLabel("-", group)
        self._mahjong_max_seconds_spin = QSpinBox(group)
        self._mahjong_max_seconds_spin.setMinimum(0)
        self._mahjong_max_seconds_spin.setMaximum(99999)
        self._mahjong_max_seconds_spin.setToolTip("0 表示不按运行时间停止，直到任务自身检测到结算排行页。")
        self._mahjong_start_game_check = QCheckBox("自动点击开始游戏", group)
        self._mahjong_auto_hu_check = QCheckBox("自动开启胡", group)
        self._mahjong_auto_peng_check = QCheckBox("自动开启碰", group)
        self._mahjong_auto_discard_check = QCheckBox("自动开启出牌", group)
        form.addRow("麻将识别档案", self._mahjong_profile_label)
        form.addRow("最大运行秒数（0 = 不按时间停止）", self._mahjong_max_seconds_spin)
        form.addRow("", self._mahjong_start_game_check)
        form.addRow("", self._mahjong_auto_hu_check)
        form.addRow("", self._mahjong_auto_peng_check)
        form.addRow("", self._mahjong_auto_discard_check)
        layout.addWidget(group)

        hint = QLabel(
            "任务会使用游戏内置自动开关完成麻将局；调试模式和 dry-run 不在正式任务界面暴露。",
            page,
        )
        hint.setWordWrap(True)
        layout.addWidget(hint)
        layout.addStretch(1)
        return page

    def _build_combat_parameters(self, parent: QWidget) -> QWidget:
        page = QWidget(parent)
        layout = QVBoxLayout(page)

        group = QGroupBox("自动战斗参数", page)
        form = QFormLayout(group)
        self._combat_profile_label = QLabel("-", group)
        self._combat_max_seconds_spin = QSpinBox(group)
        self._combat_max_seconds_spin.setMinimum(0)
        self._combat_max_seconds_spin.setMaximum(99999)
        self._combat_max_seconds_spin.setToolTip("0 表示持续监控，不按运行时长停止。")
        self._combat_strategy_combo = QComboBox(group)
        self._combat_strategy_combo.setToolTip("选择当前战斗识别档案中的战斗策略。")
        self._combat_max_encounters_spin = QSpinBox(group)
        self._combat_max_encounters_spin.setMinimum(0)
        self._combat_max_encounters_spin.setMaximum(99999)
        self._combat_max_encounters_spin.setToolTip("作战次数；0 表示不按次数停止。")
        self._combat_auto_target_check = QCheckBox("自动中键锁定目标", group)
        self._combat_auto_dodge_check = QCheckBox("启用音频自动闪避", group)
        self._combat_debug_enabled_check = QCheckBox("启用调试模式", group)
        self._combat_capture_debug_enabled_check = QCheckBox("启用截图调试", group)
        self._combat_capture_interval_spin = QDoubleSpinBox(group)
        self._combat_capture_interval_spin.setMinimum(0.1)
        self._combat_capture_interval_spin.setMaximum(60.0)
        self._combat_capture_interval_spin.setSingleStep(0.1)
        self._combat_capture_interval_spin.setDecimals(1)
        self._combat_capture_interval_spin.setSuffix(" 秒")
        self._combat_capture_interval_spin.setToolTip("战斗中按固定间隔保存调试截图。")
        self._combat_capture_max_images_spin = QSpinBox(group)
        self._combat_capture_max_images_spin.setMinimum(1)
        self._combat_capture_max_images_spin.setMaximum(99999)
        self._combat_capture_max_images_spin.setToolTip("本次战斗调试最多保存的截图数量。")
        self._combat_capture_raw_enabled_check = QCheckBox("同时保存原始截图", group)
        self._combat_capture_debug_enabled_check.toggled.connect(self._sync_combat_capture_widgets_enabled)
        form.addRow("自动战斗识别档案", self._combat_profile_label)
        form.addRow("战斗策略", self._combat_strategy_combo)
        form.addRow("最大运行秒数（0 = 持续监控）", self._combat_max_seconds_spin)
        form.addRow("作战次数（0 = 不限制）", self._combat_max_encounters_spin)
        form.addRow("", self._combat_auto_target_check)
        form.addRow("", self._combat_auto_dodge_check)
        form.addRow("", self._combat_debug_enabled_check)
        form.addRow("", self._combat_capture_debug_enabled_check)
        form.addRow("截图间隔", self._combat_capture_interval_spin)
        form.addRow("最大截图数", self._combat_capture_max_images_spin)
        form.addRow("", self._combat_capture_raw_enabled_check)
        layout.addWidget(group)

        hint = QLabel(
            "任务会持续监控敌人血条，发现战斗后自动锁定目标并执行 Q / E / 普通攻击。"
            "音频自动闪避依赖 SoundCard 和系统回采设备，如未安装或设备不可用，战斗任务仍可继续运行。",
            page,
        )
        hint.setWordWrap(True)
        layout.addWidget(hint)
        layout.addStretch(1)
        return page

    def _build_tetrominoes_parameters(self, parent: QWidget) -> QWidget:
        page = QWidget(parent)
        layout = QVBoxLayout(page)

        group = QGroupBox("俄罗斯方块参数", page)
        form = QFormLayout(group)
        self._tetrominoes_profile_label = QLabel("-", group)
        self._tetrominoes_max_seconds_spin = QSpinBox(group)
        self._tetrominoes_max_seconds_spin.setMinimum(0)
        self._tetrominoes_max_seconds_spin.setMaximum(99999)
        self._tetrominoes_max_seconds_spin.setToolTip("0 表示不按时间停止。")
        self._tetrominoes_max_pieces_spin = QSpinBox(group)
        self._tetrominoes_max_pieces_spin.setMinimum(0)
        self._tetrominoes_max_pieces_spin.setMaximum(99999)
        self._tetrominoes_max_pieces_spin.setToolTip("0 表示不按方块数停止。")
        self._tetrominoes_start_game_check = QCheckBox("自动点击开始游戏", group)
        form.addRow("俄罗斯方块识别档案", self._tetrominoes_profile_label)
        form.addRow("最大运行秒数（0 = 不按时间停止）", self._tetrominoes_max_seconds_spin)
        form.addRow("最大方块数（0 = 不按方块数停止）", self._tetrominoes_max_pieces_spin)
        form.addRow("", self._tetrominoes_start_game_check)
        layout.addWidget(group)

        hint = QLabel(
            "任务页只保留高频参数；dry_run、debug_enabled 继续固定关闭。"
            " 棋盘采样、落点求解和结果页识别参数仍在俄罗斯方块识别档案中维护。",
            page,
        )
        hint.setWordWrap(True)
        layout.addWidget(hint)
        layout.addStretch(1)
        return page

    def _build_rhythm_parameters(self, parent: QWidget) -> QWidget:
        page = QWidget(parent)
        layout = QVBoxLayout(page)

        group = QGroupBox("四键音游参数", page)
        form = QFormLayout(group)
        self._rhythm_profile_label = QLabel("-", group)
        self._rhythm_loop_count_spin = QSpinBox(group)
        self._rhythm_loop_count_spin.setMinimum(0)
        self._rhythm_loop_count_spin.setMaximum(99999)
        self._rhythm_loop_count_spin.setToolTip("0 表示持续循环，直到达到最大运行秒数或手动停止。")
        self._rhythm_max_seconds_spin = QSpinBox(group)
        self._rhythm_max_seconds_spin.setMinimum(0)
        self._rhythm_max_seconds_spin.setMaximum(99999)
        self._rhythm_max_seconds_spin.setToolTip("0 表示不按总时长停止。")
        self._rhythm_lane_keys_edit = QLineEdit(group)
        self._rhythm_lane_keys_edit.setPlaceholderText("d,f,j,k")
        self._rhythm_lane_y_offset_spin = QSpinBox(group)
        self._rhythm_lane_y_offset_spin.setMinimum(-240)
        self._rhythm_lane_y_offset_spin.setMaximum(240)
        self._rhythm_lane_y_offset_spin.setToolTip("当前判定线坐标为 0；正数向下，负数向上。")
        self._rhythm_start_game_check = QCheckBox("自动点击开始演奏", group)
        self._rhythm_close_result_check = QCheckBox("结束后自动关闭结算页", group)
        self._rhythm_debug_enabled_check = QCheckBox("保存调试截图", group)
        form.addRow("识别档案", self._rhythm_profile_label)
        form.addRow("循环次数（0 = 无限）", self._rhythm_loop_count_spin)
        form.addRow("最大运行秒数（0 = 不限制）", self._rhythm_max_seconds_spin)
        form.addRow("四轨按键", self._rhythm_lane_keys_edit)
        form.addRow("判定线偏移(px)", self._rhythm_lane_y_offset_spin)
        form.addRow("", self._rhythm_start_game_check)
        form.addRow("", self._rhythm_close_result_check)
        form.addRow("", self._rhythm_debug_enabled_check)
        layout.addWidget(group)

        hint = QLabel(
            "运行前确认已在鼓组选歌页并选好曲目。第一版使用固定判定线小区域亮度检测，"
            "如果命中偏早或偏晚，优先调整 rhythm 识别档案里的判定点和阈值。",
            page,
        )
        hint.setWordWrap(True)
        layout.addWidget(hint)
        layout.addStretch(1)
        return page

    def _build_piano_parameters(self, parent: QWidget) -> QWidget:
        page = QWidget(parent)
        layout = QVBoxLayout(page)

        file_group = QGroupBox("乐谱文件", page)
        file_layout = QVBoxLayout(file_group)
        file_row = QHBoxLayout()
        self._piano_file_edit = QLineEdit(file_group)
        self._piano_file_edit.setPlaceholderText("选择 .mid 或 .midi 文件")
        self._piano_file_edit.editingFinished.connect(self._on_piano_file_editing_finished)
        self._piano_browse_button = QPushButton("浏览...", file_group)
        self._piano_browse_button.clicked.connect(self._browse_piano_midi_file)
        file_row.addWidget(self._piano_file_edit, 1)
        file_row.addWidget(self._piano_browse_button)
        file_layout.addLayout(file_row)
        self._piano_recent_combo = QComboBox(file_group)
        self._piano_recent_combo.currentIndexChanged.connect(self._on_piano_recent_file_selected)
        file_layout.addWidget(self._piano_recent_combo)
        self._piano_file_status_label = QLabel("尚未选择 MIDI 文件。", file_group)
        self._piano_file_status_label.setWordWrap(True)
        file_layout.addWidget(self._piano_file_status_label)
        layout.addWidget(file_group)

        group = QGroupBox("演奏参数", page)
        form = QFormLayout(group)
        self._piano_conflict_policy_combo = QComboBox(group)
        self._piano_conflict_policy_combo.addItem("严格", "strict")
        self._piano_conflict_policy_combo.addItem("滚奏拆分", "roll")
        self._piano_conflict_policy_combo.currentIndexChanged.connect(self._on_piano_defaults_widget_changed)
        self._piano_tempo_scale_spin = QDoubleSpinBox(group)
        self._piano_tempo_scale_spin.setRange(0.05, 10.0)
        self._piano_tempo_scale_spin.setSingleStep(0.05)
        self._piano_tempo_scale_spin.setDecimals(2)
        self._piano_tempo_scale_spin.setValue(1.0)
        self._piano_tempo_scale_spin.setSuffix(" x")
        self._piano_tempo_scale_spin.valueChanged.connect(self._on_piano_defaults_widget_changed)
        self._piano_start_delay_spin = QSpinBox(group)
        self._piano_start_delay_spin.setRange(0, 60000)
        self._piano_start_delay_spin.setSuffix(" ms")
        self._piano_start_delay_spin.valueChanged.connect(self._on_piano_defaults_widget_changed)
        self._piano_dry_run_check = QCheckBox("仅预演，不发送按键", group)
        self._piano_dry_run_check.toggled.connect(self._on_piano_defaults_widget_changed)
        form.addRow("冲突策略", self._piano_conflict_policy_combo)
        form.addRow("速度倍率", self._piano_tempo_scale_spin)
        form.addRow("起始延迟", self._piano_start_delay_spin)
        form.addRow("", self._piano_dry_run_check)
        layout.addWidget(group)

        advanced_group = QGroupBox("高级参数", page)
        advanced_form = QFormLayout(advanced_group)
        self._piano_transpose_spin = QSpinBox(advanced_group)
        self._piano_transpose_spin.setRange(-36, 36)
        self._piano_transpose_spin.setSuffix(" 半音")
        self._piano_transpose_spin.valueChanged.connect(self._on_piano_defaults_widget_changed)
        self._piano_roll_note_spin = QSpinBox(advanced_group)
        self._piano_roll_note_spin.setRange(1, 5000)
        self._piano_roll_note_spin.setSuffix(" ms")
        self._piano_roll_note_spin.valueChanged.connect(self._on_piano_defaults_widget_changed)
        self._piano_velocity_threshold_spin = QSpinBox(advanced_group)
        self._piano_velocity_threshold_spin.setRange(0, 127)
        self._piano_velocity_threshold_spin.valueChanged.connect(self._on_piano_defaults_widget_changed)
        self._piano_focus_window_check = QCheckBox("执行前尝试聚焦游戏窗口", advanced_group)
        self._piano_focus_window_check.toggled.connect(self._on_piano_defaults_widget_changed)
        advanced_form.addRow("移调", self._piano_transpose_spin)
        advanced_form.addRow("滚奏音符间隔", self._piano_roll_note_spin)
        advanced_form.addRow("力度阈值", self._piano_velocity_threshold_spin)
        advanced_form.addRow("", self._piano_focus_window_check)
        layout.addWidget(advanced_group)

        hint = QLabel(
            "自动弹钢琴会先解析 MIDI 再执行。严格模式下遇到物理按键冲突会直接失败；"
            "滚奏拆分会把冲突和弦按很短的时间差顺序弹出。当前参数会自动记住。",
            page,
        )
        hint.setWordWrap(True)
        layout.addWidget(hint)
        layout.addStretch(1)
        return page

    def _build_log_column(self, parent: QWidget) -> QWidget:
        panel = QWidget(parent)
        layout = QVBoxLayout(panel)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(8)

        header_row = QHBoxLayout()
        title = QLabel("运行面板", panel)
        title.setObjectName("sectionTitle")
        refresh_history_button = QPushButton("刷新最近记录", panel)
        refresh_history_button.clicked.connect(self.request_refresh_history.emit)
        header_row.addWidget(title)
        header_row.addStretch(1)
        header_row.addWidget(refresh_history_button)
        layout.addLayout(header_row)

        tabs = QTabWidget(panel)
        layout.addWidget(tabs, 1)

        current_tab = QWidget(tabs)
        current_layout = QVBoxLayout(current_tab)
        current_layout.setContentsMargins(12, 12, 12, 12)
        current_card = QWidget(current_tab)
        current_card.setObjectName("sideCard")
        current_card_layout = QVBoxLayout(current_card)
        current_card_layout.setContentsMargins(12, 12, 12, 12)
        self._current_run_label = QLabel("当前没有运行中的任务。", current_card)
        self._current_run_label.setWordWrap(True)
        self._current_run_label.setObjectName("mutedText")
        current_card_layout.addWidget(self._current_run_label)
        current_layout.addWidget(current_card)
        current_layout.addStretch(1)
        tabs.addTab(current_tab, "当前运行")

        result_tab = QWidget(tabs)
        result_layout = QVBoxLayout(result_tab)
        result_layout.setContentsMargins(8, 8, 8, 8)
        self._detail_viewer = RunDetailViewer(result_tab)
        result_layout.addWidget(self._detail_viewer)
        tabs.addTab(result_tab, "最近结果")

        log_tab = QWidget(tabs)
        log_layout = QVBoxLayout(log_tab)
        log_layout.setContentsMargins(8, 8, 8, 8)
        self._log_view = QPlainTextEdit(log_tab)
        self._log_view.setObjectName("logView")
        self._log_view.setReadOnly(True)
        log_layout.addWidget(self._log_view)
        tabs.addTab(log_tab, "日志")

        history_tab = QWidget(tabs)
        history_layout = QVBoxLayout(history_tab)
        history_layout.setContentsMargins(8, 8, 8, 8)
        detail_splitter = QSplitter(Qt.Horizontal, history_tab)
        self._history_list = QListWidget(detail_splitter)
        self._history_list.currentItemChanged.connect(self._on_history_item_changed)
        detail_splitter.addWidget(self._history_list)
        self._history_detail_viewer = RunDetailViewer(detail_splitter)
        detail_splitter.addWidget(self._history_detail_viewer)
        detail_splitter.setStretchFactor(0, 0)
        detail_splitter.setStretchFactor(1, 1)
        history_layout.addWidget(detail_splitter)
        tabs.addTab(history_tab, "历史")
        return panel

    def _select_task_id(self, task_id: str) -> None:
        for index in range(self._task_list.count()):
            item = self._task_list.item(index)
            if str(item.data(Qt.UserRole) or "") == task_id:
                self._task_list.setCurrentItem(item)
                return

    def _on_task_item_changed(self, current: QListWidgetItem | None, _previous: QListWidgetItem | None) -> None:
        if current is None:
            return
        task_id = str(current.data(Qt.UserRole) or "")
        if not task_id:
            return
        if task_id not in WORKBENCH_TASKS:
            return
        self._selected_task_id = task_id
        spec = WORKBENCH_TASKS[task_id]
        self._task_title_label.setText(spec["label"])
        self._task_category_label.setText(spec["category"])
        self._task_kind_label.setText(spec["kind"])
        self._task_description_label.setText(spec["description"])
        self._parameter_stack.setCurrentWidget(self._parameter_pages[task_id])
        self._refresh_task_preflight_label()
        self._apply_task_guard()

    def _request_start_selected_task(self) -> None:
        if self._pending_launch is not None:
            self._append_log("已有任务处于倒计时等待中，请先停止或等待其开始。", level="warning")
            return
        active_cid, _active_run = self._active_runtime_run()
        if active_cid:
            self._append_log("已有会操作游戏窗口的任务正在运行，无法同时启动新任务。", level="warning")
            return
        if not self._runner_ready:
            QMessageBox.warning(self, "框架尚未就绪", "Aura 框架仍在后台启动，请稍后再执行任务。")
            return

        task_ref = self._selected_task_ref()
        if not self._task_available(task_ref):
            QMessageBox.warning(self, "任务缺失", f"当前工作区没有找到任务：{task_display_name(task_ref)}")
            return

        try:
            inputs = self._collect_selected_task_inputs()
        except ValueError as exc:
            QMessageBox.warning(self, "任务参数无效", str(exc))
            self._append_log(str(exc), level="warning")
            return
        delay_sec = int(self._ui_preferences.task_start_delay_sec)
        if delay_sec <= 0:
            self._append_log(f"正在提交任务：{task_display_name(task_ref)}")
            self._dispatch_task(task_ref, inputs)
            return

        self._pending_launch = PendingLaunch(
            task_ref=task_ref,
            inputs=inputs,
            selected_task_id=self._selected_task_id,
            remaining_sec=delay_sec,
            total_sec=delay_sec,
        )
        self._pending_timer.start()
        self._append_log(
            f"{task_display_name(task_ref)} 将在 {delay_sec} 秒后开始，可按 "
            f"{self._ui_preferences.quick_stop_hotkey} 或点击停止取消。"
        )
        self.statusBar().showMessage(f"任务将在 {delay_sec} 秒后开始。", 5000)
        self._apply_task_guard()

    def _request_refresh_runtime_probe_from_ui(self) -> None:
        if self._pending_launch is not None or self._active_runtime_run()[0]:
            self._append_log("已有会操作游戏窗口的任务正在运行或等待启动，暂不能刷新运行时探针。", level="warning")
            return
        self.request_refresh_runtime_probe.emit()

    def _on_pending_timer_tick(self) -> None:
        if self._pending_launch is None:
            self._pending_timer.stop()
            return
        remaining = self._pending_launch.remaining_sec - 1
        self._pending_launch = PendingLaunch(
            task_ref=self._pending_launch.task_ref,
            inputs=self._pending_launch.inputs,
            selected_task_id=self._pending_launch.selected_task_id,
            remaining_sec=remaining,
            total_sec=self._pending_launch.total_sec,
        )
        if remaining <= 0:
            self._dispatch_pending_launch()
            return
        self._append_log(f"倒计时：{remaining} 秒后开始执行 {task_display_name(self._pending_launch.task_ref)}。")
        self.statusBar().showMessage(f"倒计时：{remaining} 秒后开始。", 1500)
        self._apply_task_guard()

    def _dispatch_pending_launch(self) -> None:
        if self._pending_launch is None:
            return
        pending = self._pending_launch
        self._pending_launch = None
        self._pending_timer.stop()
        self._append_log(f"倒计时结束，正在提交任务：{task_display_name(pending.task_ref)}")
        self._dispatch_task(pending.task_ref, pending.inputs)
        self._apply_task_guard()

    def _dispatch_task(self, task_ref: str, inputs: dict[str, Any]) -> None:
        self._settings_store.setValue("window/last_task_ref", task_ref)
        self.statusBar().showMessage(f"正在提交任务：{task_display_name(task_ref)}", 5000)
        self.request_run_task.emit(task_ref, inputs)

    def _request_stop_current_task(self) -> None:
        if self._pending_launch is not None:
            task_name = task_display_name(self._pending_launch.task_ref)
            self._pending_timer.stop()
            self._pending_launch = None
            self._append_log(f"已取消待启动任务：{task_name}")
            self.statusBar().showMessage("已取消待启动任务。", 4000)
            self._apply_task_guard()
            return

        active_cid, active_run = self._active_runtime_run()
        cid = active_cid or self._current_active_cid
        if cid:
            task_name = task_display_name((active_run or {}).get("task_name") or self._cid_to_task_ref.get(cid))
            self._append_log(f"正在请求停止任务：{task_name} / {cid}")
            self.statusBar().showMessage("正在请求停止任务。", 5000)
            self.request_cancel_task.emit(cid)
            return

        self._append_log("当前没有可停止的任务。", level="warning")
        self.statusBar().showMessage("当前没有可停止的任务。", 4000)

    def _selected_task_ref(self) -> str:
        return WORKBENCH_TASKS[self._selected_task_id]["task_ref"]

    def _task_available(self, task_ref: str) -> bool:
        return task_ref in self._task_rows

    def _collect_selected_task_inputs(self) -> dict[str, Any]:
        if self._selected_task_id == "fishing":
            return build_auto_loop_inputs(
                self._max_rounds_spin.value(),
                self._sell_fish_every_rounds_spin.value(),
                self._bait_buy_repeat_count_spin.value(),
                self._sell_before_buy_bait_check.isChecked(),
                self._fishing_defaults,
            )
        if self._selected_task_id == "cafe":
            return build_cafe_loop_inputs(
                self._cafe_max_seconds_spin.value(),
                self._cafe_max_orders_spin.value(),
                self._cafe_start_game_check.isChecked(),
                self._cafe_wait_level_started_check.isChecked(),
                self._cafe_full_assist_auto_hammer_check.isChecked(),
                self._cafe_defaults,
            )
        if self._selected_task_id == "one_cafe":
            return build_one_cafe_inputs(
                self._one_cafe_withdraw_check.isChecked(),
                self._one_cafe_restock_check.isChecked(),
                self._one_cafe_restock_hours_combo.currentData()
                or self._one_cafe_restock_hours_combo.currentText().split()[0],
                self._one_cafe_defaults,
            )
        if self._selected_task_id == "mahjong":
            return build_mahjong_loop_inputs(
                self._mahjong_max_seconds_spin.value(),
                self._mahjong_start_game_check.isChecked(),
                self._mahjong_auto_hu_check.isChecked(),
                self._mahjong_auto_peng_check.isChecked(),
                self._mahjong_auto_discard_check.isChecked(),
                self._mahjong_defaults,
            )
        if self._selected_task_id == "combat":
            return build_combat_loop_inputs(
                self._combat_max_seconds_spin.value(),
                self._combat_max_encounters_spin.value(),
                self._combat_auto_target_check.isChecked(),
                self._combat_auto_dodge_check.isChecked(),
                self._combat_strategy_combo.currentData() or self._combat_strategy_combo.currentText(),
                self._combat_debug_enabled_check.isChecked(),
                self._combat_capture_debug_enabled_check.isChecked(),
                self._combat_capture_interval_spin.value(),
                self._combat_capture_max_images_spin.value(),
                self._combat_capture_raw_enabled_check.isChecked(),
                self._combat_defaults,
            )
        if self._selected_task_id == "tetrominoes":
            return build_tetrominoes_loop_inputs(
                self._tetrominoes_max_seconds_spin.value(),
                self._tetrominoes_max_pieces_spin.value(),
                self._tetrominoes_start_game_check.isChecked(),
                self._tetrominoes_defaults,
            )
        if self._selected_task_id == "rhythm":
            return build_rhythm_loop_inputs(
                self._rhythm_loop_count_spin.value(),
                self._rhythm_max_seconds_spin.value(),
                self._rhythm_start_game_check.isChecked(),
                self._rhythm_close_result_check.isChecked(),
                self._rhythm_lane_keys_edit.text(),
                self._rhythm_lane_y_offset_spin.value(),
                self._rhythm_debug_enabled_check.isChecked(),
                self._rhythm_defaults,
            )
        if self._selected_task_id == "piano":
            resolved_file_path = str(self._resolve_piano_file_path(self._piano_file_edit.text(), require_exists=True))
            payload = build_piano_play_midi_inputs(
                resolved_file_path,
                self._piano_conflict_policy_combo.currentData() or self._piano_conflict_policy_combo.currentText(),
                self._piano_transpose_spin.value(),
                self._piano_tempo_scale_spin.value(),
                self._piano_start_delay_spin.value(),
                self._piano_roll_note_spin.value(),
                self._piano_velocity_threshold_spin.value(),
                self._piano_focus_window_check.isChecked(),
                self._piano_dry_run_check.isChecked(),
            )
            self._persist_piano_defaults_from_widgets(resolved_file_path=resolved_file_path)
            return payload
        raise ValueError(f"未知任务：{self._selected_task_id}")

    def _active_runtime_run(self) -> tuple[str | None, dict[str, Any] | None]:
        active_runs = self._live_state.get("active_runs") or {}
        for cid, run in active_runs.items():
            task_name = str((run or {}).get("task_name") or "").strip()
            if is_runtime_interacting_task(task_name):
                return str(cid), dict(run)
        return None, None

    def _selected_task_local_error(self) -> str | None:
        if self._selected_task_id != "piano":
            return None
        raw_path = self._piano_file_edit.text().strip()
        if not raw_path:
            return "请选择 MIDI 文件"
        try:
            self._resolve_piano_file_path(raw_path, require_exists=True)
        except ValueError as exc:
            return str(exc)
        return None

    def _refresh_task_preflight_label(self) -> None:
        profile_names = {
            "fishing": self._fishing_defaults.profile_name,
            "cafe": self._cafe_defaults.profile_name,
            "one_cafe": self._one_cafe_defaults.profile_name,
            "mahjong": self._mahjong_defaults.profile_name,
            "combat": self._combat_defaults.profile_name,
            "tetrominoes": self._tetrominoes_defaults.profile_name,
            "rhythm": self._rhythm_defaults.profile_name,
            "piano": "MIDI 演奏配置",
        }
        hints = {
            "fishing": "运行前确认：角色已在钓鱼点附近，背包和鱼饵流程可正常打开。",
            "cafe": "运行前确认：已进入沙威玛小游戏入口或关卡界面，分辨率与识别档案一致。",
            "one_cafe": "运行前确认：可从都市大亨进入一咖舍；该任务归入日常领取，适合每天批量执行。",
            "mahjong": "运行前确认：已进入麻将桌或可开始对局页面，游戏内自动开关可用。",
            "combat": "运行前确认：角色在大世界待命，技能键位保持默认，自动锁定和闪避按当前配置执行。",
            "tetrominoes": "运行前确认：已进入俄罗斯方块小游戏入口或游戏页，棋盘区域无遮挡。",
            "rhythm": "运行前确认：已进入鼓组选歌页并选好曲目，四键按键与页面配置一致。",
            "piano": "运行前确认：已选择 MIDI 文件，并进入异环钢琴演奏界面。",
        }
        profile_name = profile_names.get(self._selected_task_id, "-")
        hint = hints.get(self._selected_task_id, "运行前确认：请确保游戏窗口已打开。")
        self._task_preflight_label.setText(f"{hint} 当前档案：{profile_name}")

    def _apply_task_guard(self) -> None:
        active_cid, active_run = self._active_runtime_run()
        active_runs = self._live_state.get("active_runs") or {}
        task_ref = self._selected_task_ref()
        task_available = self._task_available(task_ref)
        local_error = self._selected_task_local_error()
        start_allowed = (
            self._runner_ready
            and self._tasks_ready
            and task_available
            and local_error is None
            and self._pending_launch is None
            and not active_cid
            and task_is_enabled(task_ref, active_runs)
        )
        self._start_button.setEnabled(start_allowed)
        self._stop_button.setEnabled(self._pending_launch is not None or bool(active_cid or self._current_active_cid))
        if hasattr(self, "_refresh_probe_button"):
            self._refresh_probe_button.setEnabled(self._pending_launch is None and not active_cid)

        if self._pending_launch is not None:
            self._start_button.setText(f"等待启动（{self._pending_launch.remaining_sec}s）")
            status = f"倒计时中：{self._pending_launch.remaining_sec} 秒后执行"
        else:
            self._start_button.setText("开始执行")
            if not self._runner_ready:
                status = "框架启动中"
            elif not self._tasks_ready:
                status = "正在加载任务列表"
            elif not task_available:
                status = "任务缺失"
            elif local_error:
                status = local_error
            elif active_cid:
                status = f"运行中：{task_display_name((active_run or {}).get('task_name'))}"
            else:
                status = "空闲"
        self._selected_task_status_label.setText(status)
        if hasattr(self, "_action_status_label"):
            self._action_status_label.setText(status)
        if hasattr(self, "_task_badge_label"):
            self._task_badge_label.setText(status)
        if hasattr(self, "_current_run_label"):
            if self._pending_launch is not None:
                pending_name = task_display_name(self._pending_launch.task_ref)
                self._current_run_label.setText(
                    f"等待启动：{pending_name}\n剩余：{self._pending_launch.remaining_sec} 秒"
                )
            elif active_cid:
                active_name = task_display_name((active_run or {}).get("task_name") or self._cid_to_task_ref.get(active_cid))
                self._current_run_label.setText(f"运行中：{active_name}\nCID：{active_cid}")
            elif self._current_active_cid:
                self._current_run_label.setText(f"停止中或等待结果回传\nCID：{self._current_active_cid}")
            else:
                self._current_run_label.setText("当前没有运行中的任务。")
        self._refresh_task_list_labels()

    def _task_nav_item_text(self, task_id: str, state: str) -> str:
        spec = WORKBENCH_TASKS[task_id]
        return f"{spec['label']}  |  {spec['kind']}\n{spec['summary']}\n{state}"

    def _refresh_task_list_labels(self) -> None:
        active_cid, active_run = self._active_runtime_run()
        active_task = str((active_run or {}).get("task_name") or "")
        for index in range(self._task_list.count()):
            item = self._task_list.item(index)
            task_id = str(item.data(Qt.UserRole) or "")
            spec = WORKBENCH_TASKS.get(task_id)
            if not spec:
                continue
            task_ref = spec["task_ref"]
            if self._pending_launch and self._pending_launch.task_ref == task_ref:
                state = "等待启动"
            elif active_cid and active_task == task_ref:
                state = "运行中"
            elif task_ref not in self._task_rows:
                state = "任务缺失"
            else:
                state = "可用"
            item.setText(self._task_nav_item_text(task_id, state))
