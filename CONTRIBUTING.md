# 开发与贡献指南

[项目首页](README.md) · [环境与操作](docs/getting-started.md) · [架构与数据](docs/architecture.md)

欢迎改进跟踪、标定、身份复核和报告体验。请先按使用指南完成本地安装；前端为原生 HTML/CSS/JavaScript，不需要 npm 安装或构建。Node.js 仅用于下述 JavaScript 语法检查。

协作流程以 [小组协作约定](CODE_OF_CONDUCT.md) 为准；AI Agent 另须遵循 [AGENTS.md](AGENTS.md)。本页补充问题反馈与检查入口，不替代现有分支、提交和 PR 规范。

## 提交问题

描述预期与实际行为，并附运行编号、配置、源帧区间、Python 与依赖版本、最小复现步骤。区分漏检、检测未关联、轨迹换人、标定失败和统计问题；不能公开的视频可用脱敏或合成样例复现。

请勿提交私密路径、令牌、无授权比赛视频、权重、球员个人信息或完整运行目录。

## 开发检查

在仓库根目录执行：

```powershell
& .\.venv\Scripts\python.exe -m unittest discover -s tests -v
node --check src/footballer_analyzer/web/app.js
node --check src/footballer_analyzer/web/field-review.js
git diff --check
```

按修改范围完成相关检查；纯文档修改检查链接、命令与格式即可。算法变更请附固定素材和参数的前后对比，保留错误案例。测试通过不能替代真实精度验收，评估口径见 [验证文档](docs/validation.md)。

## 代码与数据约定

- 遵循 PEP 257，使用中文三双引号 docstring：文件开头说明模块职责，类和函数定义后说明用途。简单函数一句话即可；复杂函数说明输入输出、单位、状态或主要异常，避免逐行复述代码。
- 帧号使用源视频坐标，区间写为 `[start_frame, end_frame)`；明确位置是像素还是米、时间是秒还是帧。
- 修改算法时同步更新说明。仅整理 docstring 时，可去除 docstring 后比较 AST，确认执行逻辑未变化。
- 保留原始检测与轨迹，身份和标定修改通过版本化报告生效；不要将插值位置作为真实观测计距。
- 主题明确的修改单独提交；PR 说明问题、结果和验证，新增文档同步更新 [文档索引](docs/README.md)。

## 调研与辅助工具

`scripts/setup_bench_env.ps1`、`bench_cpu.py` 和 `bench_pipeline.py` 为调研基准工具，不替代应用安装步骤。`scripts/inspect_clip.py` 用于素材体检；`scripts/clone_refs.ps1` 用于本地获取参考源码。

`third_party/`、`data/`、`models/`、`outputs/` 和虚拟环境均已被 Git 忽略。第三方内容使用前核对 [许可记录](docs/01-参考仓库清单与许可.md)。
