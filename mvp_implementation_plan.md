# NINA MVP 实现流程

## 1. MVP 核心目标

NINA 的 MVP 不是先做一个孤立的命令行算法，而是先做一个能真实辅助判断的可视化负片反转工作台。

第一版最重要的闭环：

1. 打开 RAW/ARW 或后续支持的图像文件。
2. 在 GUI 中看到负片预览。
3. 用户可以选取片基区域。
4. 用户可以选取有效底片区域，排除边框、片夹、背景和齿孔。
5. 软件基于片基区域计算去色罩白平衡。
6. 软件只基于有效底片区域计算反转和直方图拉伸。
7. 用户可以调整基础滑块。
8. 用户可以导出正片结果。

因此，MVP 第一阶段应当是“可视化 GUI 沙盒”，而不是纯 CLI。

## 2. 为什么先做 GUI 沙盒

负片反转依赖两个关键人工判断：

- 片基区域：用于计算 `C_mask`，决定去色罩和初始白平衡。
- 有效底片区域：用于裁剪画面，并且限定直方图统计范围。

如果一开始只做命令行，需要手动输入坐标，很难判断是否选对。更重要的是，边框、胶片夹、齿孔和背景如果进入直方图统计，会直接污染黑白点和通道拉伸。

所以第一版 GUI 至少要提供：

```text
预览
片基区域选择
底片有效区域选择
反转预览入口
基础滑块占位
导出入口占位
```

## 3. GUI 方向

参考 SilverFast 一类扫描软件的工作台布局：

```text
左侧：工具、参数、滑块、状态
右侧：大图预览与交互画布
```

左侧不是营销式信息面板，而是密集、清晰、可重复使用的操作区。

初始左侧模块建议：

1. 文件
   - 打开 RAW
   - 导出
2. 选择工具
   - 移动/预览
   - 片基吸管
   - 片基框选
   - 底片区域框选
3. 当前选区
   - 片基 RGB
   - 片基区域状态
   - 底片区域状态
4. 反转
   - 反转预览
   - 重置
5. 基础调整
   - 曝光
   - 对比度
   - 黑点
   - 白点
6. 输出
   - TIFF
   - JPEG

右侧图像区优先支持：

- 适应窗口显示。
- 缩放占位。
- 平移占位。
- 矩形选区绘制。
- 点击采样占位。

第一版区域选择先用矩形，不立即做四点透视。四点透视后续作为增强功能加入。

## 4. 推荐技术栈

优先使用 PySide6。

建议依赖：

```text
PySide6          GUI
rawpy            RAW/ARW 读取
numpy            图像矩阵计算
opencv-python    边框检测、透视校正、预览缩放
tifffile         16-bit TIFF 导出
Pillow           JPEG 导出、辅助图像转换
```

## 5. 推荐项目结构

```text
qnegative/
  __init__.py
  app.py
  core/
    __init__.py
    models.py          # 参数、选区、工具模式
    raw_loader.py      # RAW 读取，后续加入
    preview.py         # 预览图生成，后续加入
    pipeline.py        # 去色罩、反转、拉伸，后续加入
    export.py          # TIFF/JPEG 导出，后续加入
  ui/
    __init__.py
    main_window.py     # 主窗口
    control_panel.py   # 左侧工具和滑块
    image_view.py      # 右侧预览画布和选区交互
```

## 6. 新阶段顺序

### 阶段 0：可视化 GUI 沙盒

目标：

- 先搭出软件的基本工作台。
- 后续功能都能逐步加到这个 GUI 上。

先做：

1. PySide6 主窗口。
2. 左侧控制面板。
3. 右侧图像预览画布。
4. 打开文件按钮。
5. 片基工具按钮。
6. 底片区域工具按钮。
7. 曝光、对比度、黑点、白点滑块占位。
8. 状态栏。

完成标准：

- 程序能启动。
- 界面是左控件、右预览的工作台布局。
- 可以通过按钮切换当前工具模式。
- 右侧画布能显示占位图或已打开的普通图片。
- 选区和滑块已经有 UI 位置，后续只需要接功能。

### 阶段 1：图像预览链路

目标：

- 打开文件后能显示预览。

先做：

1. 普通图片预览支持，用于快速测试 GUI。
2. RAW/ARW 预览读取。
3. 将大图缩放到最长边约 2048 px。
4. 保留原图尺寸和预览尺寸映射关系。

完成标准：

- 打开 ARW 后能显示一张负片预览。

- 状态栏显示文件名、图像尺寸和缩放比例。
- 不做反转时也能检查构图和边框。

### 阶段 2：片基区域和底片区域选择

目标：

- 让用户能通过 GUI 给算法提供必要输入。

先做：

1. 片基吸管：点击一个点，取周围区域中位数。
2. 片基框选：拖出一个小矩形，取区域中位数。
3. 底片区域框选：拖出有效画面区域。
4. 将预览坐标映射回图像坐标。
5. 在左侧面板显示当前选区状态。

完成标准：

- 用户能明确看到片基区域和底片区域。
- 选区数据能被 pipeline 读取。
- 底片区域能排除非画面边框。

### 阶段 3：反转 Pipeline 接入

目标：

- 在 GUI 内得到第一张正片预览。

流程：

```text
RAW 线性读取
生成预览矩阵
用户选择片基区域
用户选择底片区域
计算 C_mask
只在底片区域内做去色罩、反转和 percentile stretch
显示正片预览
```

完成标准：

- 点击“反转预览”后能得到可看的正片。
- 片基选择变化会影响色罩消除。
- 底片区域变化会影响直方图统计。

### 阶段 4：基础调整滑块

目标：

- 让用户能手动修正亮度和层次。

先做：

1. 曝光。
2. 对比度。
3. 黑点。
4. 白点。
5. 灰点或白平衡吸管占位。

完成标准：

- 拖动滑块能更新预览。
- 滑块参数能被保存到当前处理状态。
- 参数重置可用。

### 阶段 5：双层画布和性能

目标：

- 预览实时，导出高质量。

机制：

```text
计算画布：全尺寸 float32，用于最终导出
预览画布：最长边约 2048 px，用于 UI 响应
```

完成标准：

- 滑块拖动时只重算预览。
- 导出时才处理全尺寸。
- 大图不会因为滑块操作明显卡死。

### 阶段 6：导出

目标：

- 输出可用的正片文件。

先支持：

1. TIFF 16-bit / 8-bit。
2. PNG 16-bit / 8-bit。
3. JPEG 8-bit。

完成标准：

- 导出图与预览色彩趋势一致。
- TIFF / PNG / JPEG 可以被常见图像软件打开。
- 导出时有忙碌状态或进度提示。

### 阶段 7：自动辅助

目标：

- 在手动流程稳定后，再减少人工操作。

优先级：

1. 自动片基采样。
2. 自动边框检测。
3. 自动透视校正。

自动功能必须可失败，不能替代手动选择。

## 7. MVP 暂不做

第一版先不做：

- 批量处理。
- 胶片品牌 preset。
- 曲线编辑器。
- 完整 ICC 色彩管理。
- 多图浏览器。
- 图层或历史记录。
- AI 自动调色。
- 商业级降噪和锐化。

## 8. 第一轮开发任务

当前马上开始的任务：

1. 创建 PySide6 项目骨架。
2. 建立左侧控制面板。
3. 建立右侧图像画布。
4. 支持工具模式切换。
5. 支持打开普通图片作为预览测试。
6. 预留 RAW 打开入口。
7. 预留片基选区和底片区域选区状态。

第一轮完成后，软件虽然还没有真正反转，但已经具备后续接入算法的工作台。


重做 contrast：
现在 contrast 是后期线性 (x - 0.5) * contrast + 0.5，太暴力。应该改成 luminance-preserving 的温和曲线，或者把它当作 print grade 调整。
加两个新滑块：
Saturation：整体饱和度
Vibrance：只增强低饱和区域，保护已经高饱和/肤色区域
highlight的滑块会导致某个点开始反向

重做 auto WB 采样
避开暗部、避开高光、避开大面积蓝黑区域；更偏向中间亮度的低饱和像素，甚至可以加 skin/neutral candidate 过滤。

你觉得什么的嫌疑最大，片基理论上我觉得可以放弃，我们不用片基？因为本身画面通道里的最暗和最亮已经可以用于铺满。。
如果是片基什么的，那么我们需要去修复，以及不知道你有没有发现，我们的正片是没有什么品红的。。。。

还有一个可以继续抠的点：导出其实只需要最终 processed_linear_rgb，不需要 8-bit display preview 和 histogram。现在为了复用 preview result 还在算这些，下一步如果要继续提速，可以做专门的 export result，但这次我先保持改动小一些。

## 9. 当前实现状态

截至 2026-06-04，MVP 主流程已经基本成型。当前方向从“继续堆功能”转为“收敛旧分支、提高代码可维护性、稳定快捷键和预览状态”。

### 已实现

- [x] PySide6 主窗口和左侧控制面板。
- [x] 右侧 Origin / Preview 双页面画布。
- [x] RAW/ARW 读取与线性预览链路。
- [x] 普通图片/RAW 文件打开入口。
- [x] 文件夹序列读取、底部胶片条、左右切换同文件夹图片。
- [x] 片基点选工具保留为开发/高级工具；默认 Lab Print 工作流不再强制依赖片基点。
- [x] 可旋转底片区域框选，支持拖动、缩放边、旋转角、右键重置。
- [x] 旋转框接入 pipeline，支持 warp 裁切。
- [x] 每张图片单独缓存处理状态，切图后参数不会互相污染。
- [x] 反转预览入口和自动触发预览。
- [x] Lab Print 反转模型：RAW linear / camera-WB raw -> Lab Print base -> levels -> print curve -> color/WB -> display。
- [x] 用户可见 InvertMode 收敛为 Lab Print only；Density / Log Bounds / Simple 从 UI 和启动入口隐藏，后续逐步删除无用代码。
- [x] 直方图黑点、白点、中性点滑块。
- [x] 自动黑白中性点。
- [x] 自动白平衡与手动白平衡控制。
- [x] 白平衡吸管。
- [x] Global / Mid / High / Shadow 白平衡分页控制。
- [x] 曝光、对比度、高光、阴影、饱和度基础滑块。
- [x] 高光滑块反向增亮 bug 已修复。
- [x] 底部预览条当前图片高亮、已处理缩略图替换。
- [x] Preview 页面缩放、拖动、右键水平翻转、垂直翻转、旋转 90 度。
- [x] 分段 preview cache：negative / levels / color / display。
- [x] 拖动策略：拖动中低分辨率交互预览，松手后最终预览。
- [x] 2K OpenGL Preview Display Layer。
- [x] TIFF/JPEG/PNG 导出，支持 TIFF 16-bit/8-bit、PNG 16-bit/8-bit、JPEG 8-bit。
- [x] Export 进度条。
- [x] 批量导出队列窗口、暂停/继续/取消。
- [x] Export 快路径：跳过不必要的 display transform，Lab Print 导出走分段处理，导出专用 linear result。
- [x] 文件夹级 roll session 缓存，每张图保存 frame、调整参数、CMY auto WB offset、预览状态等。
- [x] Lens Correction 初版：Radial correction、Flat-frame profile、强度控制、Apply All / Unprocessed / Completed。
- [x] NINA 品牌 UI：深色/琥珀色系、Banner、启动 splash。

### MVP 仍需完成或确认

- [ ] Vibrance 滑块。
- [ ] 更温和、可解释的 contrast 曲线，避免线性后期 contrast 过暴力。
- [ ] Auto WB 继续优化：避开暗部、高光、大面积蓝黑区域，优先中间亮度低饱和候选。
- [ ] 自动亮度/自动 levels 继续调教，尤其是黑点 buffer、视觉中灰和 print curve 后的目标亮度。
- [ ] 色彩 preset / 胶片 look 的 MVP 级保存与载入。
- [ ] 最终导出与预览色彩趋势再做更多样片校验。

### 当前维护方向

详细拆分步骤见 `architecture_refactor_plan.md`。这里保留方向性摘要。

- [ ] 拆分 `qnegative/ui/main_window.py`：
  - export dialogs / export workers；
  - preview render workers；
  - session/state controller；
  - shortcut registration；
  - menu/options setup。
- [ ] 保持 ControlPanel 高内聚：
  - Basic / White Balance / Lens Correction / Output 各自面板化；
  - 参数只向外发结构化 values，不直接操纵 MainWindow 状态。
- [ ] 快捷键改用窗口级 `QShortcut`，减少焦点落在控件/画布/filmstrip 时快捷键失效。
- [ ] 逐步删除 Density / Log Bounds / Simple 的 UI、菜单、启动参数和无用分支；核心 pipeline 内部代码可等 Lab Print 稳定后再安全移除。
- [ ] 排查偶发 preview 丢失：切图、自动框线、后台预反转、缓存恢复之间需要更清晰的状态机。

### 未来方向


### 后续 Export 优化记录

Export 已经可用，但后续仍有继续加速空间，暂时标记为未来优化项：

- [ ] 做真正专用的 export result，进一步跳过所有预览专用数据结构。
- [ ] Tiled export：先用预览/抽样确定 auto levels 和 WB，再全分辨率分块处理，降低内存峰值。
- [ ] 减少中间数组 copy，把 log、levels、curve、WB、saturation 等步骤尽量融合。
- [ ] 用 LUT 加速 print curve / gamma / sRGB 转换。
- [ ] 评估 numexpr / Numba / CuPy / OpenCL 等 backend，只在收益明确后接入。
- [ ] 批量导出时做多进程并行，单张图仍优先优化 pipeline 本身。


Frame:
1. preview resize 到 max edge 1200-1600
2. luminance normalize 到 uint8
3. Canny / threshold / morphology 三路生成候选 mask
4. findContours + minAreaRect
5. 按面积、矩形度、边缘支持、中心先验、aspect 评分
6. 输出 ImageRect + confidence

Base:
1. 如果 frame 找到，在 frame 外四边条带采样
2. 如果 frame 没找到，在图像边缘带采样
3. 候选按低方差、不 clipping、亮度、橙色/片基色倾向评分
4. 输出 mask_point 或 mask_rgb + confidence

## 10. 2026-06-03 自动框线、片基 fallback 与维护记录

这一轮新增的重点是：把“自动辅助选框”的实验链路跑通，同时让默认 Lab Print 工作流不再强制依赖片基点。

### 新增功能

- [x] Lab Print 模式允许无片基反转。
  - `qnegative/core/pipeline.py::build_negative_base_preview()` 在 `mask_point is None` 时使用 `[1.0, 1.0, 1.0]` 作为占位 base。
  - 这个 fallback 只应该视为“无片基占位”，不是实际采样到的片基。
  - UI 状态会显示 `Base fallback: none`，避免误认为已经采样到片基 RGB。
- [x] 用户可见工作流收敛为 Lab Print only。
  - `Density`、`Simple`、`Log Bounds` 不再作为用户可选模式出现。
  - 旧模式代码如仍存在，只视为待清理的历史实现，不应继续接入 UI 或默认流程。
- [x] Lab Print 导出也允许无片基。
  - 导出检查逻辑与预览一致：只要求有效 frame。
- [x] 新增 `qnegative/core/frame_ranker.py`。
  - 作用：加载轻量 frame ranker 模型，为自动框线提供候选排序。
  - 默认模型搜索顺序：
    1. `models/frame_ranker.joblib`
    2. `models/frame_ranker_dual_smoke.joblib`
  - 当前推理配置：
    - 预览最长边：`384`
    - 全局候选：`1400`
    - 预筛候选保留：`360`
  - 推理结果返回 `RankedFrameCandidate`，包含 `rect`、`confidence`、`score`、`format_hint`、`method`。
- [x] `auto_detect.py` 接入 ranker。
  - `detect_film_frame()` 会先尝试 `_ranker_frame_candidates()`。
  - ranker 不可用、置信度不够或异常时，自动 fallback 到原来的 OpenCV contour/projection 检测。
  - 这保证实验模型不会破坏原有手动/传统检测路径。
- [x] 新增 `qnegative/tools/generate_frame_labels.py`。
  - 作用：用已经裁切好的正片 reference 和对应未裁切 RAW/负片生成 frame label。
  - 按文件名 stem 匹配 negative/positive，支持递归扫描。
  - 支持 reference 方向搜索：`identity`、`rot90`、`rot180`、`rot270`、`flip_h` 及其旋转组合。
  - 输出 JSONL label、summary 和 debug overlay/contact sheet。
- [x] 新增 `qnegative/tools/train_frame_ranker.py`。
  - 作用：训练 ExtraTreesRegressor 候选框 ranker。
  - 训练目标是候选框与标注框的 IoU。
  - 使用 confidence 加权，高置信 label 权重更高。
  - 支持小角度旋转、缩放、平移的数据增强。
  - 新增 dual-mode prefilter：
    - `base_ring_score`：适合框外有亮片基、稳定片基环绕的情况。
    - `tight_crop_score`：适合翻拍台紧裁、框外偏黑但框本身正确的情况。
  - 不再把“框外黑”直接作为强惩罚，因为部分高质量翻拍确实紧贴黑色翻拍台。
- [x] 依赖新增：
  - `scikit-learn`
  - `joblib`

### 当前烟测结果

本轮可用烟测模型：

```text
models/frame_ranker_dual_smoke.joblib
```

训练/评估命令：

```text
python -m qnegative.tools.train_frame_ranker ^
  --labels calibration\frame_labels_expanded.jsonl ^
  --max-labels 60 ^
  --preview-max-size 512 ^
  --candidates-per-image 120 ^
  --global-candidates 2600 ^
  --augmentations 1 ^
  --out-dir calibration\frame_ranker_smoke_dual_60 ^
  --model-out models\frame_ranker_dual_smoke.joblib
```

烟测指标：

```text
labels: 60
train: 45
test: 15
train samples: 8640
Top1 mean IoU: 0.909
Top1 median IoU: 0.924
Top3 mean best IoU: 0.916
Top3 median best IoU: 0.924
Raw oracle mean IoU: 0.926
Top1 IoU >= 0.80: 100%
Top1 IoU >= 0.85: 86.7%
Top3 IoU >= 0.85: 93.3%
```

维护判断：

- 这个结果足够用于“自动建议框线 + 用户检查/微调”。
- 还不应该直接作为“完全自动裁切并批量应用”的最终版本。
- 模型 Top1 已经接近 raw oracle，说明当前瓶颈主要是候选生成器，而不是 ExtraTrees 排序器。

### 当前数据与产物

- `calibration/frame_labels_expanded.jsonl`
  - 当前主 label 集。
  - 由正片 reference 反投影到负片得到。
- `calibration/frame_ranker_smoke_dual_60/frame_ranker_report.json`
  - 当前 smoke report。
- `models/frame_ranker_dual_smoke.joblib`
  - 当前可用于实验性自动框线的模型。

注意：

- `negative file/`、`posituve file/`、TIFF、RAW、JPG/PNG debug 图都在 `.gitignore` 中，不应提交。
- `*.stackdump` 已加入 `.gitignore`，不要提交 shell 崩溃 dump。

### 自动片基与无片基策略

当前策略：

```text
Lab Print:
  frame 有效即可预览/导出
  mask_point 可选
  无 mask_point 时显示 Base fallback: none
```

原因：

- 当前默认 Lab Print 主要依据 frame 内部的 log bounds、levels、print curve 和 WB 流程工作。
- 它本身已有 `LAB_PRINT_ANALYSIS_INSET = 0.05`，会在分析黑白中性点时避开裁切边缘 5%，降低边框/片基污染直方图的概率。
- Density / Simple / Log Bounds 已经从用户可见工作流隐藏；未来应逐步删除，而不是继续维护它们的片基分支。

未来自动片基 scorer 建议：

1. 在 frame 外四条边生成条带候选。
2. 沿旋转框方向采样多个小 patch。
3. 排除：
   - clipped 区域；
   - 过暗区域；
   - 方差过高区域；
   - 明显图像内容区域。
4. 对候选加分：
   - 局部亮度稳定；
   - RGB/亮度方差低；
   - 与 frame 内 5% inset 后的底片主体相比，密度明显更低；
   - 多个候选 patch 的 RGB 中位数一致。
5. 如果找不到可信片基：
   - Lab Print：继续使用无片基 fallback。
   - UI 文案继续区分 `Base fallback: none` 和真实 `Base RGB ...`。

### 自动框线后续维护建议

优先优化候选生成器，而不是继续盲目加大模型：

- 当前候选生成器仍偏粗暴，很多时间花在对大量候选做 mask/ring stats。
- 每个候选现在都会 rasterize rect mask、计算 inside/outside 特征，正式全量训练会很慢。
- 下一步应该做候选生成器评估：

```text
对每张 label 图：
  生成候选
  计算 max IoU / top oracle IoU
  记录候选数量和耗时
```

目标：

```text
候选数量：200 - 800
Top oracle IoU >= 0.90 的比例 > 95%
```

达到这个目标后，再训练 CNN 或更复杂模型才更有意义。

### 不要踩的坑

- 不要把 “框外黑” 简单视为错误。
  - 紧裁翻拍台常常框外就是黑色，但框是正确的。
- 不要让自动框线静默覆盖用户手动框。
  - 自动功能应当是建议，或者中置信度时让用户确认。
- 不要提交完整 RAW/TIFF 数据集。
  - 当前 repo 只保留 label、report、轻量模型和工具代码。
- 不要让 Lab Print fallback base 误导成真实片基。
  - UI 文案必须继续区分 `Base fallback: none` 和 `Base RGB ...`。

滑块调节时虽然有 stage cache，但仍然会启动完整 render task
它会检查 key 后跳过一些 stage，但任务创建、pixmap 更新、histogram/status 更新仍然频繁。

部分参数的 cache key 还可以更精细
现在 display key 包含 highlights/shadows/saturation，color key 包含 exposure/contrast/curve/WB 等。下一步可以继续把“只影响显示层”的东西拆得更干净。

## 11. Future Plan: Film Stitching Workflow

### 背景

部分 6x6 / 6x7 / 宽幅底片在翻拍时可能需要分两张或多张拍摄，再拼成一张完整底片。Lightroom 的 Pano DNG 可以完成拼接，但会生成 Linear/Pano DNG，当前 rawpy RAW pipeline 不能稳定处理。因此，未来 NINA 可以考虑内置一个“为底片翻拍准备的拼接 workflow”，避免依赖 Lightroom Pano DNG。

### 为什么这比普通全景更简单

底片拼接不是风景全景：

- 目标通常是平面底片，而不是三维远景。
- 图像数量少，多数是 2 张或 3 张。
- 画幅比例有强先验，例如 6x6、6x7、645、135。
- 翻拍台和相机通常固定，透视变化小。
- 输出目标不是球面/柱面投影，而是一个平面矩形底片。

因此第一版不需要完整 panorama engine，可以先做受约束的 planar stitch。

### 建议 MVP 流程

```text
选择 2-3 张相邻 RAW
-> 生成线性 preview
-> 先做镜头暗角/flat-frame 校正
-> 粗略自动找每张底片 frame
-> 在 frame 内做特征匹配或相位相关
-> RANSAC 求 homography / affine transform
-> 将多张图 warp 到同一画布
-> overlap 区域做亮度/颜色匹配
-> feather / multiband blend
-> 输出 stitched linear RAW-like buffer
-> 进入现有 Lab Print pipeline
```

### 开源项目/库参考

- OpenCV Stitcher / stitching detailed sample
  - 可参考 feature detection、matching、homography、warper、seam finder、exposure compensator、blend 等模块。
  - 优点：我们项目已经依赖 OpenCV，适合做内置 MVP。
  - 风险：默认 panorama mode 偏向普通全景，需要禁用/简化球面投影、wave correction 等不适合平面底片的步骤。

- Hugin / libpano13 / nona / enblend
  - Hugin 是成熟开源全景工具链，可参考 control points、remap、blend 的整体工作流。
  - `nona` 负责根据项目参数做 remap，`enblend` 负责 seam blending。
  - 优点：质量成熟，命令行工具链可作为外部集成方向。
  - 风险：依赖复杂、GPL 兼容性需要确认、和 NINA 的 RAW/linear pipeline 集成成本较高。

- Autopano-sift-C / control point generators
  - 可参考自动控制点生成思路。
  - 作为历史/算法参考即可，不建议第一版直接依赖。

### 第一版实现建议

优先自研一个轻量模块，而不是直接接完整 Hugin：

```text
qnegative/core/stitching.py
  detect_overlap_features()
  match_pair()
  estimate_pair_transform()
  warp_pair_to_canvas()
  blend_overlap()
```

第一版只支持两张水平/垂直拼接：

- 用户选择两张 RAW。
- 用户指定拼接方向：left-right / top-bottom。
- 自动估计 overlap。
- 失败时允许用户手动放 2-4 对控制点。
- 输出一个 stitched preview，并允许用户继续进入 frame selection / Lab Print。

### 关键注意点

- 拼接应发生在反转前，最好在线性 RAW/camera RGB 或 camera-WB linear preview 上完成。
- 必须先做 lens falloff / flat-frame 校正，否则 overlap 区域会出现亮度接缝。
- 颜色和曝光匹配只应做局部 gain/offset，不应提前套 print curve。
- 如果用于最终导出，必须保存每张源图的 transform，而不是只保存 stitched preview。
- 不应将 Lightroom Pano DNG 作为主工作流依赖；它可以作为未来 fallback 输入，但不是当前主线。

### 开发优先级

当前不进入 MVP 主线，标记为未来增强：

1. 先完成当前 RAW/DNG 普通输入、自动框线、颜色和导出稳定性。
2. 再做两张图 planar stitch prototype。
3. 最后再考虑多图、Hugin/enblend 外部工具链、Linear/Pano DNG fallback。

1. 暗角 / Lens Correction
   - Flat-frame profile 需要继续优化，目标接近 Lightroom 的平滑校正效果。
   - 重点处理边缘低对比度、高噪点、强度感不稳定的问题。
   - 研究 raw black level、per-channel shading map、gain cap、blur radius、中心/边缘归一化策略。
2. 高光调整
   - 当前高光滑块仍需要确认是否真正恢复/压缩高光，而不是简单改变白色区域灰度。
3. 曲线调节
   - 增加用户可控的 curve smoothing。
   - 可以分别 smooth 右半部分高光 roll-off 和左半部分阴影 toe。
4. 色彩调节
   - 增加更完整的 saturation / vibrance / color balance / preset 保存。
   - 继续调教 Auto WB 和 CMY 自动偏移。
5. 自动框线
   - 当前自动框线仍会被翻拍台、相邻底片、复杂边缘干扰。
   - 继续优化中心发散、边缘候选、format 先验和轻量 ranker。
6. Preview 稳定性
   - 仍存在偶发 preview 丢失或状态恢复不完整，需要作为架构清理的一部分处理。
