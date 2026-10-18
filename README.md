# FootballerAnalyzer

支持固定低俯拍与运动战术视角的足球球员追踪、场地标定与跑动分析原型。

**当前阶段**：调研、需求、选型与设计已完成；最小闭环原型已实现。实片追踪与前端对拍已验证，距离精度仍待独立参考路程验证。

小组成员开始开发前，请先阅读 [小组协作约定](CODE_OF_CONDUCT.md)。
Agent 开发规范见 [AGENTS.md](AGENTS.md)。

球识别经验见 [模型对照与跨帧筛选](docs/15-球检测模型对照与跨帧筛选.md)。浏览器已支持独立足球追踪、模型选择和倒三角标记复核；报告尚未纳入球数据。

## 现在可以做什么

- 浏览器选择 `data/clips` 中尚无人物追踪的素材，独立执行人物追踪、场地标定和足球追踪，查看阶段进度与结果。
- YOLOv8n 人物检测、ByteTrack 跟踪、球衣颜色分组，缓存原始像素轨迹。
- 浏览器同帧展示原图检测框、脚点与球场投影；支持播放、逐帧、源帧跳转和联动选择。
- 填写场地尺寸，逐帧识别白线并自动匹配中圈、边线和禁区，随镜头更新投影；前端回看白线证据与模型叠加。
- 十一人制 / 八人制独立模型；固定镜头通过模型参考点与原图配对完成标定，无需输入 XY，预览确认后整段复用。
- 人物完成后，通过“步骤 2”打开场地标定；八人制默认手动，十一人制默认逐帧白线，自动失败帧暂停投影。
- 建立持久球员档案，按源帧区间绑定、拆分或排除轨迹。
- 重算距离、速度、有效时间和热图，导出 HTML、叠加视频与 Parquet；查询选定报告中的球员历史。

## 快速开始（Windows / PowerShell）

在项目根目录执行，推荐 Python 3.12；本机已建好的 `.venv` 可直接复用。

```powershell
python -m venv .venv
& .\.venv\Scripts\python.exe -m pip install --upgrade pip
& .\.venv\Scripts\python.exe -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
& .\.venv\Scripts\python.exe -m pip install -e . -c configs/constraints-tested.txt
```

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
$runDir = (Get-ChildItem .\outputs -Directory -Filter 'run_*' | Sort-Object Name -Descending | Select-Object -First 1).FullName
& .\.venv\Scripts\python.exe -m footballer_analyzer calibrate-lines --run $runDir --pitch-length 105 --pitch-width 68
```

`track_id` 只在一次运行内有效，`player_id` 才是跨场身份；`home/away` 为人工确认的本场队伍，颜色簇不自动代表真实主客队。帧区间均为左闭右开 `[start_frame, end_frame)`。

### 八人制固定标定与复查

默认采用 [JFA 八人制推荐规范](https://www.jfa.jp/youth_development/players_first/pdf/8system.pdf)：68 × 50 m，球门 5 × 2.15 m，罚球区 12 × 29 m，球门区 4 × 13 m，罚球点距门线 8 m，中圈及罚球弧半径 7 m。整体长宽可调整，禁区尺寸不会随整体等比缩放。

已处理素材直接选择原运行，在“场地标定”切换模型即可复用人物缓存。点选期间锁定参考帧；更换模型、尺寸或参考帧会清空对应草稿，修改后必须重新预览。相机、焦距或裁切变化时需重新标定。高级设置可限定有效帧区间及可靠区域，默认不会将投影限制在参考点凸包内。

```powershell
& .\.venv\Scripts\python.exe scripts/review_manual_calibration.py --run outputs/run_...
# 可加 --reference <JSON>，计算未参与拟合参考点的像素误差
```

诊断图和 JSON 位于运行目录的 `diagnostics/manual_calibration/`。当前 `clip_70m00s` 已保存第 147 帧的四点标定草稿；首、中、末帧可持续投影，原始人物缓存保持不变。素材球门与 JFA 预设仍有差异，独立门柱参考点误差约 40 / 23 px，因此尚未通过全场几何精度验收。操作、接口和验收边界见 [14 八人制模型与手动标定](docs/14-八人制模型与手动标定.md)。

### 独立验收运动镜头白线

无需先运行人物检测即可生成前端可查看的白线运行：

```powershell
& .\.venv\Scripts\python.exe -m footballer_analyzer calibrate-lines --video data/clips/barca_girona_45m00s.mp4 --pitch-length 105 --pitch-width 68
# 可加 --start-frame / --end-frame 限定验收片段；前端刷新后选择新运行
& .\.venv\Scripts\python.exe scripts/review_pitch_calibration.py --run outputs/run_... --reference configs/calibration_review_barca.json
```

复查脚本输出 `white-lines.mp4`、抽样图、失败区间和独立人工参考点误差。参考文件绑定视频 SHA256，仅用于验收，算法不读取这些点。两段移动素材及强光白线修复见 [13 逐帧白线标定与验收](docs/13-逐帧白线标定与验收.md)。

### 检测与跟踪参数

演示配置现用 `imgsz: 1280`、`confidence: 0.1`、`nms_iou: 0.5`。检测下限和轨迹激活阈值分开：`0.1 < 分数 < 0.25` 的框参与已有轨迹的第二次关联，新建轨迹仍需达到 `0.35`，避免低分噪声直接创建编号。NMS 在跟踪前抑制重复人物框。

`lost_track_buffer_s: 3.0` 按真实采样帧率换算丢失缓冲；`track_activation_threshold: 0.25` 与 `minimum_matching_threshold: 0.8` 可单独调整。缓冲保留关联状态，漏检帧仍没有真实观测，不能用来补算路程。修改参数后需要重新执行 `track`，新旧运行的编号不能直接互用。

追踪精度优先时使用 `track --config configs/tracking_quality.yaml`（1920 输入）。人工抽样复查：`python scripts/review_tracking.py --run outputs/run_... --reference configs/tracking_review_demo.json`。对比与剩余串号见 [09 追踪丢失诊断与修复](docs/09-追踪丢失诊断与修复.md)。

### CLI 报告与历史

```powershell
# 需已确认标定并至少绑定一名球员
& .\.venv\Scripts\python.exe -m footballer_analyzer report --run $runDir

# 显式指定每段素材要采用的报告版本
& .\.venv\Scripts\python.exe -m footballer_analyzer history --player player_001 --reports outputs/run_.../reports/report_...
```

`history --reports` 可接受多个报告目录，同一片段只选一个版本。重叠片段会被拒绝；同场片段偏移未知时只逐片展示。显示的是“已分析片段累计”，不外推整场跑动。

## 产物与结构

```text
configs/                         演示配置、标定点、验证过的依赖版本约束
src/footballer_analyzer/          CLI、检测跟踪、标定、身份、统计、报告、本地服务
src/footballer_analyzer/web/      原生 HTML/CSS/JavaScript 前端，无额外构建步骤
tests/                           时间、分段、身份、重算与 HTTP 边界测试
outputs/run_*/
  manifest.json                  源文件摘要、帧率、配置、依赖、状态与耗时
  tracks.parquet / tracks.mp4     原始观测与轨迹预览
  detections.parquet             关联前的检测框和分数，包含未关联框
  catalog.json / crops/           轨迹目录与缩略图
  track_catalog.html              可直接打开的缩略图索引
  calibration.yaml               标定、适用范围与确认状态
  calibrations/lines_*/           独立标定版本：frames.json、质量摘要、逐帧透明白线掩膜
  balls/ball_*/                   独立足球版本：模型指纹、逐源帧候选与筛选结果；漏检为 null
  bindings.yaml                  本次运行的身份和排除区间
  diagnostics/                   人工抽样复查 JSON 与标框图（执行复查脚本后生成）
  reports/report_*/              独立版本的统计、图表、视频、HTML 与输入快照
data/players.yaml                跨场球员档案（首次从前端创建）
```

## 验证与当前限制

```powershell
& .\.venv\Scripts\python.exe -m unittest discover -s tests -v
node --check src/footballer_analyzer/web/app.js
```

当前 **54 项测试通过**，覆盖人物与场地独立流程、足球补跑与漏检保留、模型接口、原件指纹保护，以及既有几何、追踪和报告回归。真实 5 秒人物素材：1080p25、125 帧；1280 修复版 3545 条观测、约 16.9 秒，1920 精度版 4106 条观测、约 27.0 秒（含预采样与产物输出，不含 Python 启动导入）。五帧人工抽样共 105 人次，追踪观测由旧配置 44 次提升到精度版 103 次，仍有 3 次抽样编号变化。初版记录见 [08 实现与验收记录](docs/08-实现与验收记录.md)，追踪证据见 [09 追踪丢失诊断与修复](docs/09-追踪丢失诊断与修复.md)。

当前使用 CPU 检测；调研中的 DirectML 尚未接入本原型。短片可用于检查链路，未证明长片跟踪准确率或 ±10% 距离误差。现有模型仍有漏检、断轨和换 ID；报告仅统计可靠区域内的连续有效观测，缺口不插值。逐帧标定面向白线清晰的连续战术镜头；固定镜头小场已支持手动标定，实际尺寸精度及严重遮挡、近景和切镜的全面支持尚未验证。视频仍要求恒定帧率；球检测支持人工复核，尚无球速、事件、自动 ReID、多摄融合或实时服务。

## 文档与调研工具

- [文档索引](docs/README.md)：需求、选型、调研与专题报告。
- [07 架构设计与最小闭环](docs/07-架构设计与最小闭环.md)：两阶段流水线、数据约束、指标口径与验收标准。
- [08 实现与验收记录](docs/08-实现与验收记录.md)：已实现能力、复现命令、实测与未通过项。
- [09 追踪丢失诊断与修复](docs/09-追踪丢失诊断与修复.md)：漏检与串号原因、参数对比、人工抽样及 ReID 后续方案。
- [13 逐帧白线标定与验收](docs/13-逐帧白线标定与验收.md)：运动镜头几何、前端验收、独立参考点与验证边界。
- [14 八人制模型与手动标定](docs/14-八人制模型与手动标定.md)：参考点配对、固定投影、接口及素材精度边界。
- `scripts/setup_bench_env.ps1`、`bench_cpu.py`、`bench_pipeline.py`：原调研基准工具，不替代上面的应用安装步骤。
- `scripts/inspect_clip.py`：素材体检；`scripts/clone_refs.ps1`：本地获取参考源码。

## Python 代码说明约定

- 遵循 PEP 257，使用中文三双引号 docstring：文件开头说明模块职责，类和函数定义后说明用途。
- 简单函数用一句话；复杂函数补充输入输出、单位、区间、状态或主要异常，避免逐行复述实现。
- 帧号明确来自源视频，区间写为 `[start_frame, end_frame)`；说明位置用像素还是米、时间用秒还是帧。
- 修改算法时同步更新对应说明；文档整理可通过去除 docstring 后比较 AST，检查执行逻辑是否改变。

## 第三方源码与素材

`third_party/`、`data/`、`models/`、`outputs/`、虚拟环境及第三方版权截图均已被 `.gitignore` 排除。仓库不包含视频或模型权重，请使用有权使用的本地素材。第三方许可见 [01 仓库清单与许可](docs/01-参考仓库清单与许可.md)；当前检测依赖 Ultralytics（AGPL-3.0），沿用调研阶段的许可约束。
