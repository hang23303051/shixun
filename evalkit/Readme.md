# Ref4D EvalKit

`evalkit/` 是一个**可独立运行的评测工具包**：给定一组 **Ref4D 的参考侧元数据/证据**（已提供在 `data/metadata/`），以及你自己的 **生成视频**（放到 `data/genvideo/<modelname>/`），即可对 **Text-to-Video** 模型输出进行四维打分，并生成可对比的 CSV 结果表。

四个维度：

- **Semantic**：基础语义对齐（对象/属性/关系/绑定等）
- **Motion (RRM)**：运动一致性（方向/幅度/平滑 + freeze 惩罚）
- **Event**：事件时序一致性（EGA / ERel / ECR / S_event）
- **World**：世界知识一致性（规则库 + VQA/断言打分）

---

## 1. 目录结构（你需要关注的部分）

```bash
evalkit/
├── data/
│   ├── metadata/                      # ★ Ref4D 官方提供的参考侧元数据/证据/缓存
│   │   ├── ref4d_meta.jsonl           # 每行一个 sample_id、prompt、主题等
│   │   ├── semantic_evidence/         # 参考侧语义证据（可复用）
│   │   ├── event_evidence/            # 参考侧事件证据（可复用）
│   │   └── motion_ref/rrm_*/          # 参考侧运动缓存（npz，含 frz_meta）
│   ├── genvideo/                      # ★ 要评测的生成视频放这里
│   │   └── <modelname>/<sample_id>.mp4
│   └── example_models/                # 示例模型视频（用于快速 smoke test）
│       └── sora2/<sample_id>.mp4
│
├── ref4d_eval/                         # 四维评测核心代码
├── scripts/                             # 一键脚本（推荐使用）
├── envs/                                # conda 环境 yml（按维度拆分）
├── third_party/                         # 第三方仓库
└── outputs/                             # 评测输出（自动生成）
````

---

## 2. 输入约定：如何放你的生成视频

**必须按 `sample_id` 命名**（与 `data/metadata/ref4d_meta.jsonl` 一致）：

```bash
evalkit/data/genvideo/<modelname>/<sample_id>.mp4
```

示例：

```bash
evalkit/data/genvideo/my_model/animals_and_ecology_001_single.mp4
evalkit/data/genvideo/my_model/animals_and_ecology_002_single.mp4
```

你可以一次放多个模型：

```bash
evalkit/data/genvideo/modelA/*.mp4
evalkit/data/genvideo/modelB/*.mp4
...
```

---

## 3. 输出：结果会写到哪里？

默认输出都在 `evalkit/outputs/` 下（脚本会自动创建）。

常见结果文件：

* Semantic：`outputs/semantic/scores/*.csv`（或 `scores.csv`）
* Motion：`outputs/motion/motion_rrm.csv`
* Event：`outputs/event/scores/summary.csv`
* World：`outputs/world_knowledge/<model>/root.csv`
* 四维合并表：`outputs/overall/ref4d_4d_scores.csv`

---

## 4. 快速开始（推荐路径）

### 4.1 先跑示例（验证环境 + 路径）

示例视频在 `data/example_models/` 下。建议先跑语义 smoke test：

```bash
cd evalkit
bash scripts/run_semantic_eval.sh
```

如果你能看到输出写入 `outputs/semantic/`，说明最基本的环境/路径没问题。

---

## 5. 分维度运行（推荐用 `scripts/` 一键脚本）

> 这些脚本默认会扫描 `data/genvideo/` 下的所有模型目录进行评测。

### 5.1 Semantic（基础语义对齐）

```bash
cd evalkit
bash scripts/run_semantic_eval.sh
```

只评测某个模型（如果脚本支持 `INCLUDE_MODELS`）：

```bash
INCLUDE_MODELS=my_model bash scripts/run_semantic_eval.sh
```

示例运行（仓库自带 `example_models`）：

```bash
bash scripts/run_semantic_eval.sh --use-example
```

---

### 5.2 Motion（RRM 运动一致性）

```bash
cd evalkit
bash scripts/run_motion_rrm.sh
```

输出通常是：

```bash
outputs/motion/motion_rrm.csv
```

---

### 5.3 Event（事件时序一致性）

```bash
cd evalkit
bash scripts/run_event_eval.sh
```

输出：

```bash
outputs/event/scores/summary.csv
```

---

### 5.4 World（世界知识一致性）

```bash
cd evalkit
bash scripts/run_world_knowledge.sh
```

输出：

```bash
outputs/world_knowledge/<model>/root.csv
```

---

### 5.5 四维一键评测（推荐）

```bash
cd evalkit
bash scripts/run_4d_eval.sh
```

结束后会合并到：

```bash
outputs/overall/ref4d_4d_scores.csv
```

---

## 6. 权重与环境说明

### 6.1 `checkpoints/` 不入 git：你需要自己准备权重

`checkpoints/` 目录一般不提交到仓库（文件大、许可证复杂、下载源多）。
请按脚本或 README 提示放在类似路径：

```bash
evalkit/checkpoints/
├── minicpm-v-4_5/        # 语义/世界知识常用 VLM
├── e5-large-v2/          # 事件/语义常用文本编码
├── videollama3-7b/       # event 维度 VLM（如使用）
├── transnetv2/           # event shot boundary（如使用）
├── ddmnet/               # event gebd（如使用）
├── tapnet_checkpoints/   # motion TAPIR
├── sam2/                 # motion 分割
├── groundingdino/        # motion 检测
└── ...
```

仓库里提供了下载脚本：

* `scripts/download_motion_models.sh`
* `scripts/download_event_models.sh`
* （语义/世界知识若无脚本，需要你自行放置 MiniCPM/E5 权重）

各维度环境分别创建，请按相应 `envs/*.yml` 配置执行。

