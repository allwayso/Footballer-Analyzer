# 使用指南

[项目首页](../README.md) · [文档索引](README.md) · [架构与数据](architecture.md) · [验证与限制](validation.md)

本页承接原 README 中的安装、操作、CLI 与参数说明；命令从项目根目录执行。

## 快速开始（Windows / PowerShell）

在项目根目录执行，推荐 Python 3.12；本机已建好的 `.venv` 可直接复用。

```powershell
python -m venv .venv
& .\.venv\Scripts\python.exe -m pip install --upgrade pip
& .\.venv\Scripts\python.exe -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
& .\.venv\Scripts\python.exe -m pip install -e . -c configs/constraints-tested.txt
```

需要用 NVIDIA 显卡加速检测时，把 CPU 版 PyTorch 换成 CUDA 版（只需 NVIDIA 驱动，wheel 自带 CUDA 运行时）：

```powershell
& .\.venv\Scripts\python.exe -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cu130
```

之后用配置项 `device: gpu`、命令行 `--device gpu` 或前端下拉框切换；GPU 不可用时会明确报错，不会静默退回 CPU。详见 [22 核显推理与设备切换](22-核显推理与设备切换.md)。

已安装本项目时，可直接从这里开始：

```powershell
# 视频放在 data/clips，模型放在 models/yolov8n.pt
& .\.venv\Scripts\python.exe -m footballer_analyzer serve --port 8766
```

打开 **[http://127.0.0.1:8766/](http://127.0.0.1:8766/)**。默认使用 8766，避开 AnkiConnect 常用的 8765；端口冲突时可通过 `--port` 更换。服务按 `Ctrl+C` 停止；换用其他输出目录时加 `serve --output-root <目录>`。视频、模型、球员档案与分析产物保存在本机，未纳入 Git。

### 前端操作顺序

0. 在“步骤 1 · 人物追踪”选择视频，点击“开始人物追踪”。此步不计算场地；后续标定默认设置中，`clip` 固定镜头可选“八人制 / 手动固定”，移动战术镜头选“十一人制 / 逐帧白线”。默认 1920 检测输入、逐源帧处理整段；可选 1280 或限定区间。刷新可恢复任务进度，服务重启会终止未完成任务。
1. 左侧选择运行；点击列表、图像中的球员框或球场圆点，联动核对相同轨迹 ID。
2. 用上一帧、下一帧、滑条或输入源帧号后点击“跳转”定位。预览最高 10 FPS；源帧号与原始时间保持一致。
3. 点击“步骤 2 · 场地标定”的“打开场地标定”。固定镜头选择模型及“手动固定”，暂停在清晰帧，点击“以当前帧开始标定”。直接在“球场位置”地图选参考点，再点原图对应地面位置；至少四对，建议六对以上，选择分布充分的场角或禁区角。八人制地图纵向显示，下端为近端／相机侧，上端为远端；十一人制保持横向。可重选、删除；中线无需可见。
4. 点击“预览标定与球员投影”，检查首、中、末帧及未参与拟合的参考点，勾选验收并保存。默认覆盖当前运行全部帧，自动白线失败不会影响固定投影。移动镜头则运行逐帧白线标定，播放检查黄色模型，青色仅为白色线索；失败帧不投影，检查后保存验收。
5. 在“步骤 3 · 足球追踪与复核”选择模型，点击“运行足球追踪”。默认“足球 YOLO11n · 小目标切片”，本地权重为 `models/yolo11n-football-ball.pt`；缺少权重的模型不可运行。已有结果可直接切换模型查看，无需重跑人物和场地。黄色倒三角尖端指向球心；地图为地面近似投影，腾空球不代表真实落点。用“下一足球缺失帧”检查漏检，空帧不插值。模型选择在本浏览器内记忆，结果刷新后可恢复。
6. 在“身份与轨迹”建立球员档案并绑定区间。同一人断轨后把新 ID 绑定到同一档案；同 ID 换人时分别绑定不同区间；裁判或场外人员可以排除。编辑已有绑定时先移除原条目，再保存替代区间。
7. 在“跑动统计”点击“生成 / 重算报告”；点击统计行查看速度与热图，打开完整报告下载叠加视频和数据。修改绑定或标定后会提示旧报告过期，重算保留历史版本。
8. “导出对拍图”生成当前帧的画面与球场并排 PNG，再点“保存 PNG”。

排查漏检时勾选“未关联检测”：红色虚线表示有检测但尚未形成轨迹，并显示本帧检测与关联数量。原始检测包括场外人员，不等于在场球员数；旧运行缺少检测缓存时需重新跑 `track`。

人物尚未绑定真实身份时也显示空心投影点，可直接检查位置；自动模式中场地匹配失败帧不显示投影，手动模式使用固定映射。素材筛选中的“尚无人物追踪”表示没有已完成的人物运行，白线独立验收和失败运行不计入；已有短区间运行不代表整段视频已处理，可取消筛选重新分析。每次分析新建运行，保留历史结果。场地计算异常时可打开已完成的人物结果，再到“场地标定”重试。当前每次只允许一个人物任务和一个足球任务；场地与足球都可单独重试。

终端仍可单独处理，随后在页面刷新选择运行：

```powershell
& .\.venv\Scripts\python.exe -m footballer_analyzer track --config configs/tracking_quality.yaml
# 将占位路径替换为 track 输出的实际运行目录
$runDir = 'outputs/run_...'
& .\.venv\Scripts\python.exe -m footballer_analyzer calibrate-lines --run $runDir --pitch-length 105 --pitch-width 68
```

`track_id` 只在一次运行内有效，`player_id` 才是跨场身份；`home/away` 为人工确认的本场队伍，颜色簇不自动代表真实主客队。帧区间均为左闭右开 `[start_frame, end_frame)`。

### 独立场内轨迹校对

从左侧“场内轨迹校对”进入独立的 `/field-review` 页面。选择已完成人物追踪的运行；场地标定经人工确认后，页面按有效场内观测列出轨迹片段。原视频位于上方，片段列表和操作历史位于下方。

点击片段首帧或末帧，可跳转到对应的实际观测帧：所选行加深，所选轨迹框显示为加粗鲜红色。这里的末帧指最后一帧真实观测，不是身份绑定区间的右开边界；没有选中片段时不显示红色选中框。

点击片段右侧“并入”，输入融合目标源帧，并在该帧实际存在的场内轨迹中选择目标，确认后保存。操作历史支持逐项撤销；刷新或重启服务后可恢复。融合记录单独存入运行目录的 `field_review.json`，用于人工校对分组，不改写原始轨迹，也尚未自动同步到球员身份绑定或跑动报告。

点击“分析轨迹关联”可复用已有视频和轨迹，按球场位置与运动分析断轨，颜色辅助；无可靠标定时用稳定邻近球员作相对参照。高分且通过竞争与整组冲突检查的关联自动融合，其余可在“可能融合对象”栏定位、确认。历史区分自动／手动，支持批次撤销与阻止重合；输入变化后旧自动项暂停生效。新运行也在追踪中积累证据，结束后完整复核。当前参数为未完成调优的工程初值，85分钟素材初测128→116组，**尚未通过减半和独立精度验收**。算法、复现命令和完整边界见 [17 球员轨迹重关联与验收](17-球员轨迹重关联与验收.md)。

常规跑动报告仍按 `bindings.yaml` 处理身份；[独立跑动可视化](20-球员跑动可视化.md) 则在副本中读取有效且身份无冲突的融合组，不改写轨迹或绑定。两种报告不能混用身份口径。

### 八人制固定标定与复查

默认采用 [JFA 八人制推荐规范](https://www.jfa.jp/youth_development/players_first/pdf/8system.pdf)：68 × 50 m，球门 5 × 2.15 m，罚球区 12 × 29 m，球门区 4 × 13 m，罚球点距门线 8 m，中圈及罚球弧半径 7 m。整体长宽可调整，禁区尺寸不会随整体等比缩放。

已处理素材直接选择原运行，在“场地标定”切换模型即可复用人物缓存。点选期间锁定参考帧；更换模型、尺寸或参考帧会清空对应草稿，修改后必须重新预览。相机、焦距或裁切变化时需重新标定。高级设置可限定有效帧区间及可靠区域，默认不会将投影限制在参考点凸包内。

```powershell
& .\.venv\Scripts\python.exe scripts/review_manual_calibration.py --run outputs/run_...
# 可加 --reference <JSON>，计算未参与拟合参考点的像素误差
```

诊断图和 JSON 位于运行目录的 `diagnostics/manual_calibration/`。当前 `clip_70m00s` 已保存第 147 帧的四点标定草稿；首、中、末帧可持续投影，原始人物缓存保持不变。素材球门与 JFA 预设仍有差异，独立门柱参考点误差约 40 / 23 px，因此尚未通过全场几何精度验收。操作、接口和验收边界见 [14 八人制模型与手动标定](14-八人制模型与手动标定.md)。

### 独立验收运动镜头白线

无需先运行人物检测即可生成前端可查看的白线运行：

```powershell
& .\.venv\Scripts\python.exe -m footballer_analyzer calibrate-lines --video data/clips/barca_girona_45m00s.mp4 --pitch-length 105 --pitch-width 68
# 可加 --start-frame / --end-frame 限定验收片段；前端刷新后选择新运行
& .\.venv\Scripts\python.exe scripts/review_pitch_calibration.py --run outputs/run_... --reference configs/calibration_review_barca.json
```

复查脚本输出 `white-lines.mp4`、抽样图、失败区间和独立人工参考点误差。参考文件绑定视频 SHA256，仅用于验收，算法不读取这些点。两段移动素材及强光白线修复见 [13 逐帧白线标定与验收](13-逐帧白线标定与验收.md)。

### 检测与跟踪参数

默认跟踪器为 `motion-color-v1`，采用用户回看后选定的整组参数：`tracking_max_cost: 0.99`、`tracking_position_tolerance: 3.0`、`tracking_competition_margin: 0`、`tracking_new_threshold: 0.75`。页面分析、`track` 和 `retrack` 均使用这组默认值；YAML 或命令行显式设置时以设置为准。已有运行保持原结果，需新建运行才能使用新参数。

演示配置现用 `imgsz: 1280`，页面和质量配置使用1920；检测下限为 `confidence: 0.1`，NMS 为 `nms_iou: 0.5`。所有达到检测下限的框均可续接已有轨迹；新身份至少需两次真实观测，且最高检测分数达到0.75才能确认，最多延迟0.2秒。`retrack --new-threshold` 可覆盖新身份确认阈值。

`lost_track_buffer_s: 3.0` 限制最长续接缺口，只有画面运动时最多1秒。`track_activation_threshold: 0.25` 与 `minimum_matching_threshold: 0.8` 是 `bytetrack-legacy` 的关联参数，不替代新版新建阈值。缓冲保留关联状态，漏检帧仍没有真实观测，不能用来补算路程。修改参数后需要重新执行 `track`，新旧运行的编号不能直接互用。

追踪精度优先时使用 `track --config configs/tracking_quality.yaml`（1920 输入）。人工抽样复查：`python scripts/review_tracking.py --run outputs/run_... --reference configs/tracking_review_demo.json`。对比与剩余串号见 [09 追踪丢失诊断与修复](09-追踪丢失诊断与修复.md)。

### 复用缓存重跟踪

已有 `detections.parquet` 的运行可复用检测，不重跑 YOLO；命令仍需原视频提取颜色。输出为新运行，不自动迁移旧身份绑定或人工融合：

```powershell
& .\.venv\Scripts\python.exe -m footballer_analyzer retrack --run outputs/run_...
# 仅需对照旧算法时使用
& .\.venv\Scripts\python.exe -m footballer_analyzer retrack --run outputs/run_... --algorithm bytetrack-legacy
```

`--max-cost`、`--position-tolerance`、`--competition-margin` 与 `--new-threshold` 可显式覆盖参数。当前默认由用户回看选定，不代表已经完成系统化准确率和覆盖验收；详见 [19 运动颜色前置与缓存重跟踪](19-ByteTrack运动颜色前置与缓存重跟踪.md)。

### 阶段性能基准

对人物追踪、逐帧白线和足球追踪做优化时，使用固定协议重新执行，而不是读取旧运行耗时：

```powershell
& .\.venv\Scripts\python.exe -m footballer_analyzer benchmark `
  --config configs/performance_benchmark.yaml `
  --repeat 3 --warmup 1
```

可用 `--stages tracking`、`--stages pitch` 或 `--stages ball` 单独测试；每次输出独立 `result.json`，可用 `--baseline <旧 result.json>` 比较速度倍数。详细字段和基准边界见 [阶段计算速度分析](16-各阶段计算速度分析.md)。

### CLI 报告与历史

已人工校验的运行可生成独立的球员轨迹回放、活动热图、速度图与跑动对比页，支持分组筛选及 CSV／JSON／PNG 导出。执行 `python scripts/visualize_player_motion.py --run <运行目录>`，从现有服务打开生成的报告或直接打开 HTML；不另起端口。有效融合仅用于可视化身份副本，不改写原始轨迹与绑定，缺口不补算距离。使用与统计口径见 [20 球员跑动可视化](20-球员跑动可视化.md)。

```powershell
# 需已确认标定并至少绑定一名球员
& .\.venv\Scripts\python.exe -m footballer_analyzer report --run $runDir

# 显式指定每段素材要采用的报告版本
& .\.venv\Scripts\python.exe -m footballer_analyzer history --player player_001 --reports outputs/run_.../reports/report_...
```

`history --reports` 可接受多个报告目录，同一片段只选一个版本。重叠片段会被拒绝；同场片段偏移未知时只逐片展示。显示的是“已分析片段累计”，不外推整场跑动。

## 静态演示与离线展示

公开[演示站](https://allwayso.github.io/Footballer-Analyzer-Demo/) 使用预计算结果，不需要本地服务。`scripts/export_static_demo.py` 从指定运行和可视化报告生成三个只读页面、逐帧图与离线预览入口；需要 FFmpeg，输出目录必须不存在。发布步骤和完整命令见 [21 评委静态演示站](21-评委静态演示站.md)。

## 常见问题

| 现象 | 检查方法 |
|---|---|
| 页面找不到原素材或旧运行 | 检查 `data/clips/`、`--output-root`、源文件路径及运行状态；切换端口不会迁移数据。 |
| 缺少模型权重 | 确认人物权重位于 `models/yolov8n.pt`；足球模型需另备对应权重，缺失时不可执行。 |
| 有检测但没有轨迹 | 查看“未关联检测”，区分漏检与关联失败；新版新身份还需两次观测及确认阈值。 |
| 球场没有投影点 | 检查标定确认状态、有效区间和可靠区域；自动标定失败帧会暂停投影。 |
| 无法生成常规报告 | 先确认标定并至少绑定一名球员，再检查区间冲突。 |
| 修改身份后统计没变 | 重算对应类型的报告，并确认查看的是新版本；融合与绑定的作用范围见上文。 |

完整命令可通过 `python -m footballer_analyzer --help` 及各子命令的 `--help` 查看。精度与性能证据见[验证文档](validation.md)。
