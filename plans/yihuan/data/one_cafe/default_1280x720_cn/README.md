# 一咖舍模板清单

此目录用于存放中文客户端 `1280x720` 基准截图裁剪出的模板资源。
运行时请尽量从游戏客户端内容区截图裁剪，避免把 Windows 标题栏、窗口边框裁进模板。

v1 任务 `tasks:one_cafe:revenue_restock` 使用固定坐标点击动态的一咖舍地图标记，因此不要裁剪带呼吸动画的整块 `一咖舍 / NTE` 背景作为点击模板。

## 必需模板

入口与退出：

- `city_tycoon_title.png`：都市大亨地图页稳定标题或左上角固定标识。
- `shop_management_title.png`：商铺管理页稳定标题或顶部固定标识。
- `world_hud.png`：世界场景 HUD 的稳定小区域，用于确认任务已退出都市大亨并回到世界。

收益领取：

- `withdraw_button.png`：商铺管理页左侧的「提取收益」按钮。
- `no_revenue_toast.png`：没有收益时出现的「没有收益可以领取」横幅提示。
- `withdraw_report_title.png`：有收益时出现的「营收报告」弹窗标题或稳定内容。
- `withdraw_report_confirm.png`：营收报告里的粉色「确定」按钮。
- `reward_popup.png`：「获得物品」弹窗的稳定标题或主体区域。

补货：

- `restock_entry_button.png`：商铺管理页左侧的「补货」入口按钮。
- `inventory_title.png`：原料库存页标题或稳定标识。
- `restock_4h_selected.png` / `restock_4h_unselected.png`：原料库存页的 4 小时选项两种状态。
- `restock_24h_selected.png` / `restock_24h_unselected.png`：原料库存页的 24 小时选项两种状态。
- `restock_72h_selected.png` / `restock_72h_unselected.png`：原料库存页的 72 小时选项两种状态。
- `inventory_restock_button.png`：原料库存页右下角「补货」按钮。
- `inventory_full_toast.png`：「库存已满无法存入更多材料」横幅提示。
- `delivery_prompt.png`：材料可补货时出现的送货上门提示弹窗主体。
- `delivery_button.png`：送货上门提示中的「送货上门」按钮。
- `delivery_cost_prompt.png`：送货花费确认弹窗主体。
- `delivery_cost_confirm.png`：送货花费确认弹窗中的「确认」按钮。
- `restock_success_toast.png`：「补货成功」提示。

## 固定坐标

以下坐标按 `1280x720` 客户端内容区配置，不包含 Windows 标题栏：

- 一咖舍地图标记点击点：`(530, 545)`。
- 获得物品弹窗空白关闭点：`(640, 650)`。
- 都市大亨右上角关闭按钮点击点：`(1222, 42)`。

这些坐标都配有后续识别校验：一咖舍点击后会等待 `shop_management_title.png`，关闭都市大亨后会等待 `world_hud.png`。如果你的客户端裁剪区域不同，请优先修正运行时捕获区域或对应坐标。
