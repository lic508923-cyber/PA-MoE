# PA-MoELog

PA-MoELog 是面向跨系统、少标签日志异常检测的轻量多源专家框架。当前实现遵循以下主线：

> 免模板解析 + 参数感知 BERT + 多源专家 + 轻量静态融合 + 融合后共享 DoRA + 目标感知 GMM。

当前版本已经取消 AdaLoRA 和可训练 Top-k Router。BERT、参数编码主干及源域专家在目标域适配时默认冻结，只更新共享 DoRA、目标分类头，并用目标域正常 support 样本重新估计 GMM。

## 模型结构

```text
raw logs
  -> parser-free preprocessing (semantic text + typed parameter values)
  -> shared BERT / hash encoder
  -> parameter-aware encoder
  -> source expert pool
  -> lightweight expert fusion (uniform or support-calibrated static weights)
  -> shared target DoRA
  -> target classifier + target-aware GMM
  -> fused anomaly score
```

最终分数为：

```text
S_final = alpha * normalized(S_classifier) + beta * normalized(S_energy)
```

其中 GMM 的输入是多专家融合并经过目标域 DoRA 修正后的 `target_hidden`，而不是适配前的共享表示。阈值以及 `alpha`、`beta` 应只在目标验证集上选择，测试集只用于最终报告。

## 核心模块

- `pa_moelog/data/preprocess.py`：不依赖 Drain/Spell，将日志拆为占位符语义文本和带类型的原始参数值。
- `pa_moelog/models/parameter_attention.py`：联合参数类型、参数值与文本上下文，生成参数感知表示。
- `pa_moelog/models/experts.py`：每个源域专家保存对应系统的正常/异常知识；专家基础层不包含目标域 DoRA。
- `pa_moelog/models/fusion.py`：`LightweightExpertFusion` 保存非训练型专家权重。默认均匀融合，也可根据 support set 距离通过 softmax 一次性校准。
- `pa_moelog/models/dora.py`：实现融合后的共享 `DoRALinear` 目标适配层。
- `pa_moelog/models/gmm_energy.py`：使用目标正常 support 表示拟合对角协方差密度，并输出负对数似然能量。
- `pa_moelog/models/pa_moelog.py`：组装完整前向链路与双分支异常分数。

## 目标域更新范围

| 模块 | 目标适配阶段 |
|---|---|
| BERT / 文本主干 | 冻结 |
| 参数编码主干 | 冻结 |
| 源域专家基础参数 | 冻结 |
| 专家融合权重 | 静态或由 support set 一次性校准 |
| 融合后共享 DoRA | 更新 |
| 目标分类头及归一化层 | 更新 |
| GMM | 用目标正常 support 样本重新估计 |
| 最终阈值 | 在目标验证集校准 |

## 安装与本地 BERT

```bash
pip install -r requirements.txt
```

工作区已提供本地 BERT：

```text
E:\demo\models\bert-base-uncased
```

正式实验可传入：

```bash
--backbone-name E:\demo\models\bert-base-uncased --no-hash-fallback
```

快速离线验证可使用 `--backbone-name simple-hash-encoder`。

## 快速前向验证

```bash
python -m pa_moelog.demo_forward
```

模型的关键输出包括：

- `final_score`：分类证据与 GMM 能量融合后的异常分数；
- `classifier_score`：目标分类头的异常概率；
- `energy_score`：归一化的目标感知 GMM 能量；
- `fusion_weights`：当前静态专家权重；
- `target_hidden`：融合后经共享 DoRA 得到的目标表示；
- `expert_logits`：各源专家的原始预测。

Python 示例：

```python
from pa_moelog.data import LogPreprocessor
from pa_moelog.models import PAMoELog

logs = [
    "Failed login from 192.168.1.10 user=root port=22 error=403",
    "service started successfully on port 8080",
]
batch = LogPreprocessor().parse_sequence(logs)
model = PAMoELog(num_experts=3, hidden_dim=128, backbone_name="simple-hash-encoder")
output = model(batch["semantic_texts"], batch["parameters"])
print(output["final_score"])
print(output["fusion_weights"])
```

如需按目标域与各专家中心距离校准静态权重：

```python
model.fusion.calibrate_from_distances(distances, temperature=1.0)
```

`distances` 的形状应为 `[num_experts]`，值越小的专家权重越高。

## 训练与迁移流程

### 1. 训练源域专家

每个源系统训练一个专家，建立多源知识库：

```bash
python scripts/train_source_experts.py \
  --train-csv tests/toy_data/source_bgl.csv \
  --system-name BGL --num-experts 3 --expert-id 0 \
  --epochs 1 --batch-size 2 --device cpu \
  --backbone-name simple-hash-encoder
```

对其他源系统分别设置对应的 `expert-id`。当前主线不再需要训练 Router，也不需要负载均衡损失，因此项目中已移除旧的 Router 训练脚本。

### 2. 目标域 few-shot 适配

目标数据必须严格拆分为 support、validation、test。适配脚本冻结共享主干和源专家，只训练融合后的共享 DoRA、目标归一化层和分类头；训练结束后使用 support 中的正常样本拟合 GMM。

```bash
python scripts/adapt_target.py \
  --support-csv tests/toy_data/target_support.csv \
  --base-checkpoint artifacts/checkpoints/source_experts/BGL_expert0.pt \
  --target-system Thunderbird \
  --epochs 1 --batch-size 2 --device cpu \
  --backbone-name simple-hash-encoder
```

实际多源实验应先把各源专家权重装入同一个模型 checkpoint，再执行目标适配。不要使用 test set 拟合 GMM、设置融合系数或选择阈值。

### 3. 评估

```bash
python scripts/evaluate.py \
  --test-csv tests/toy_data/target_test.csv \
  --checkpoint artifacts/checkpoints/target_adapt/Thunderbird_adapted.pt \
  --batch-size 2 --device cpu \
  --output-json artifacts/outputs/toy_eval.json \
  --backbone-name simple-hash-encoder
```

建议报告 Precision、Recall、F1、AUROC、AUPRC，并同时记录分类分支、能量分支和静态专家权重，便于消融与误报分析。

## 数据格式

CSV 默认字段：

```csv
log,label,system
"BGL node boot completed successfully",0,BGL
"BGL fatal machine check error code=500 node=42",1,BGL
```

- `log`：原始日志；
- `label`：`0` 正常、`1` 异常；
- `system`：可选的来源系统名称。

toy 数据位于 `tests/toy_data/`。

## 项目结构

```text
pa_moelog/                 核心包
  data/                    数据与免解析预处理
  models/                  参数感知编码、专家、轻量融合、DoRA、GMM
  utils/                   checkpoint 与指标
scripts/                   源专家训练、目标适配与评估
tests/toy_data/            最小 CSV 数据
artifacts/                 本地 checkpoint 与评估输出
```

## 当前架构定义

PA-MoELog 通过免解析的参数感知编码保留日志语义与显式参数，以多源专家迁移不同系统的异常知识，使用稳定的目标感知静态权重融合专家，通过融合后共享 DoRA 完成参数高效的少标签适配，并利用目标正常样本驱动的 GMM 能量校准跨系统异常边界。
