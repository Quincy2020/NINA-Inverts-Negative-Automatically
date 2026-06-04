# NINA 架构拆解计划

这份计划的目标是：在功能不动的前提下，把 `qnegative/ui/main_window.py` 从“所有东西都在一个文件里”逐步拆成更高内聚、低耦合的模块。

核心原则：

- 每一步只搬代码，不改功能、不改算法、不调 UI。
- 每一步都能独立验证、独立提交、独立回滚。
- 先拆纯 UI / 纯 worker / 纯 helper，最后再拆状态机。
- MainWindow 最终只负责组装窗口、持有当前状态、连接模块。

## 0. 当前问题

`qnegative/ui/main_window.py` 当前同时承担这些职责：

- 主窗口布局、菜单、快捷键。
- 批量导出设置窗口和导出队列窗口。
- RAW preview worker、自动框线 worker、预反转 worker、正片 preview render worker。
- Export worker、格式选择、TIFF/PNG/JPEG 写入。
- preview cache key、render stage cache、raw preview cache、positive preview cache。
- 文件夹切图、roll session 保存/恢复。
- 自动框线、自动预反转、导出、预览渲染之间的状态调度。

这会导致：

- 新增控件时容易牵动很多无关逻辑。
- 快捷键、菜单、状态恢复互相影响。
- preview 偶发丢失这类 bug 很难定位。
- 后续做暗角、高光曲线、色彩 preset 时会继续把 MainWindow 撑大。

## 1. 目标模块结构

建议逐步走向这个结构：

```text
qnegative/
  ui/
    main_window.py          # 只保留主窗口组装和顶层状态协调
    menus.py                # File / Edit / View / Options / Developer 菜单
    shortcuts.py            # 所有主窗口快捷键注册
    export_dialogs.py       # BatchExportSettingsDialog, BatchExportDialog
    export_tasks.py         # ImageExportTask, ExportSignals, image write helpers
    preview_tasks.py        # RawPreviewTask, PreviewRenderTask, PreInvertTask, AutoDetectTask
    preview_cache.py        # PreviewStageCache, cache key helpers, CachedPreviewResult
    workflow_state.py       # 当前图片状态、切图、session 恢复/保存的轻量协调
```

不是一次性拆完。推荐每次只新增一个模块，把对应代码搬过去。

## 2. 拆分顺序

### Step 1: `ui/shortcuts.py` - done

搬出内容：

- `_build_shortcuts()`
- `_add_shortcut()`

新模块接口：

```python
def install_main_window_shortcuts(window) -> list[QShortcut]:
    ...
```

MainWindow 保留：

```python
self._shortcuts = install_main_window_shortcuts(self)
```

验收标准：

- `Tab` 切换 Origin / Preview。
- `I` 触发反转。
- `K` 触发自动框线。
- `Q/A/W/S/E/D/R/F` 生效。
- `[` / `]` 调整中性点。
- `Ctrl+Z / Ctrl+Y` 生效。
- 焦点在左侧滑块、右侧画布、底部 filmstrip 时快捷键都能用。

风险：

- 低。
- 这是最适合第一步做的拆分。

### Step 2: `ui/export_dialogs.py` - done

搬出内容：

- `BatchExportSettings`
- `BatchExportSettingsDialog`
- `BatchExportDialog`

MainWindow 保留：

- `export_completed()`
- batch export 队列调度方法。

验收标准：

- 批量导出设置窗口能打开。
- 命名模式、输出目录、格式、overwrite 设置仍然生效。
- 批量导出队列窗口能显示、暂停、继续、取消。

风险：

- 低。
- 基本是纯 UI 搬家。

### Step 3: `ui/export_tasks.py` - done

搬出内容：

- `ExportSignals`
- `ExportCancelled`
- `TiffExportTask`
- `transform_preview_array()`
- `linear_to_srgb16()`
- `linear_to_srgb8()`
- `export_format_from_path()`
- `export_format_from_filter()`
- `export_format_extension()`
- `export_format_label()`
- `encode_export_rgb()`
- `write_export_image()`

建议顺便改名：

```text
TiffExportTask -> ImageExportTask
```

因为现在已经支持 TIFF / PNG / JPEG，继续叫 Tiff 会误导维护。

MainWindow 保留：

- `export_current()`
- `export_completed()`
- `_start_next_batch_export()`
- `_export_finished()`
- `_export_failed()`
- `_export_cancelled()`

验收标准：

- 单张导出 TIFF 16-bit 正常。
- 单张导出 TIFF 8-bit、PNG 16-bit、PNG 8-bit、JPEG 正常。
- 批量导出正常。
- export timing 日志仍正常。
- preview/export 颜色趋势不变。

风险：

- 中低。
- 注意不要改 `_process_export()` 算法，只搬 task 和 helper。

### Step 4: `ui/preview_cache.py` - done

搬出内容：

- `PreviewStageCache`
- `PreviewRenderOutput`
- `CachedPreviewResult`
- `CachedRawPreview`
- cache key helpers：
  - `current_levels()`
  - `image_point_key()`
  - `image_rect_key()`
  - `matrix_key()`
  - `cmy_offsets_key()`
  - `file_identity_key()`
  - `color_balance_key()`
  - `lens_correction_key()`
  - `adjustments_preview_cache_key()`
  - `preview_result_cache_key_for()`
  - `base_stage_key()`
  - `lab_print_*_key()`

MainWindow 保留：

- cache 字典本身。
- 何时清 cache、何时恢复 cache 的决策。

验收标准：

- 调整滑块时 stage cache 仍然生效。
- 切图回来能恢复 preview。
- 改 frame / lens / WB 后不会错误复用旧 preview。

风险：

- 中。
- cache key 是 preview 丢失和错图复用的关键，需要只搬不改。

### Step 5: `ui/preview_tasks.py` - done

搬出内容：

- `PreviewRenderSignals`
- `AutoDetectSignals`
- `RawPreviewSignals`
- `PreInvertSignals`
- `ModelWarmupSignals`
- `RawPreviewTask`
- `PreInvertTask`
- `ModelWarmupTask`
- `AutoDetectTask`
- `PreviewRenderTask`
- `PreInvertOutput`
- `AutoDetectOutput`
- `scaled_raw_preview()`
- `_cmy_offsets_to_state()`

MainWindow 保留：

- 任务何时排队。
- 任务完成后如何更新 UI 和状态。
- `_preview_render_finished()`、`_raw_preview_finished()`、`_preinvert_finished()` 这类 slot。

验收标准：

- 打开 RAW 后 raw preview 正常。
- 自动框线任务正常。
- preview render 正常。
- 自动预反转相邻图片正常。
- 切图时不会把上一张结果写到当前张。

风险：

- 中高。
- 这里最容易碰到 stale job、job_id、缓存归属问题。
- 必须保留当前 job_id 检查逻辑。

### Step 6: `ui/menus.py`

搬出内容：

- `_build_menus()`

新模块接口：

```python
def build_main_menus(window) -> None:
    ...
```

MainWindow 保留：

- QAction 调用的方法。
- QAction 状态对应的 setter。

验收标准：

- File / Edit / View / Options / Developer 菜单都存在。
- GPU preview 开关正常。
- 自动反转、自动框线、预反转半径、roll session autosave 选项正常。
- Developer 里只保留真正需要的开发项。

风险：

- 低到中。
- 注意 QActionGroup 需要保存到 window 上，否则可能被回收。

### Step 7: `ui/workflow_state.py`

这是最后拆，不能太早动。

可拆内容：

- 当前文件夹序列状态。
- `image_states` 保存/恢复。
- roll session autosave。
- filmstrip badge 恢复。
- 切图前保存当前状态。

MainWindow 保留：

- 当前 UI 控件更新。
- 预览渲染/自动框线/导出调度。

目标接口可以是：

```python
class RollWorkflowState:
    def save_current(...)
    def restore_for_path(...)
    def load_roll_session(...)
    def save_roll_session(...)
```

验收标准：

- 打开文件夹后能读取 `.nina/roll_session.json`。
- 每张图的 frame、参数、CMY offsets、positive badge 能恢复。
- 切图不会丢当前图调整。
- 关闭软件前能保存 session。

风险：

- 高。
- preview 丢失 bug 很可能就在这里和 render queue 的交界处。
- 前面步骤没完成前不要急着拆。

## 3. 每一步固定验证

每一步拆完都跑：

```powershell
python -m compileall -q qnegative
$env:QT_QPA_PLATFORM='offscreen'; python -c "import sys; from PySide6.QtWidgets import QApplication; from qnegative.ui.main_window import MainWindow; app=QApplication(sys.argv); window=MainWindow(); print('main window construct ok')"
```

手动冒烟：

1. 启动 app。
2. 打开一个 RAW 文件夹。
3. 切换 2-3 张图。
4. 自动框线。
5. 手动调曝光、WB、中性点。
6. 切到另一张再切回来。
7. 单张导出。
8. 批量导出已完成图片。

## 4. 提交策略

每一步一个 commit。

推荐 commit 形状：

```text
Refactor shortcut registration
Move batch export dialogs out of MainWindow
Move image export task out of MainWindow
Move preview cache helpers out of MainWindow
Move preview worker tasks out of MainWindow
Move menu construction out of MainWindow
Prepare roll workflow state extraction
```

每个 commit 都应该能独立运行。

## 5. 不在这轮做的事

为了保持重构干净，这轮不要顺手做：

- 暗角算法改进。
- 高光/阴影算法改进。
- Auto WB 调教。
- 自动框线候选生成优化。
- GPU export。
- 新 UI 风格调整。
- 删除 `pipeline.py` 里的旧反转算法。

旧代码可以先留在 core 里。真正删除前，需要确认：

- 没有 UI 入口。
- 没有 session 入口。
- 没有 export/preview 入口。
- README 和文档不再把它当作支持功能。

## 6. 最终成功标准

这轮架构清理完成后：

- `main_window.py` 明显变短，主要保留窗口组装和状态协调。
- Export、Preview、Shortcut、Menu 都有独立文件。
- 新增一个控制面板或工具时，不需要翻完整个 `main_window.py`。
- 快捷键稳定，不再依赖“先点一下左侧菜单”。
- preview 丢失 bug 更容易定位，因为 render task、cache、session restore 的边界更清楚。
