# PA-MoELog

PA-MoELog 是一个面向跨系统日志异常检测的最小可运行骨架。它的目标是在不依赖日志模板解析器的前提下，保留日志中的显式参数信息，并通过多源专家、稀疏路由和少标签适配，让模型能够迁移到新的目标系统。

当前代码重点是跑通一次完整前向传播和实验闭环。训练流程描述了推荐的实现方式，可作为后续扩展真实数据实验的蓝图。

## 项目结构

```text
PA-MoELog/
├── README.md                 # 项目说明与实验命令
├── requirements.txt          # Python 依赖
├── pa_moelog/                # 核心 Python 包
│   ├── data/                 # CSV 数据集与免解析预处理
│   ├── models/               # PA-MoELog 模型、专家、router、GMM、BERT backbone
│   ├── utils/                # checkpoint 与指标工具
│   └── demo_forward.py       # 最小前向传播演示
├── scripts/                  # 训练、适配、评估脚本
├── tests/toy_data/           # toy CSV 数据
└── artifacts/                # 本地实验产物，已被 git 忽略
    ├── checkpoints/
    └── outputs/
```

BERT 权重已单独放在工作区模型目录：

```text
E:\demo\models\bert-base-uncased
```

## 1. 模型解决的问题

传统跨系统日志异常检测常常依赖 Drain、Spell 等模板解析器，把原始日志先转换成模板，再训练异常检测模型。这样做有两个问题：

- 不同系统的日志格式差异大，模板解析规则迁移成本高。
- 模板化会弱化参数信息，例如 IP、路径、端口、错误码、PID 等，而这些参数经常与异常模式直接相关。

PA-MoELog 解决的是“新系统少标签适配”场景：已经有多个源系统的日志数据和专家模型，希望在目标系统只有少量标注日志时，仍然能快速获得可用的异常检测能力。

模型的核心思路是：

- 免解析：不强制挖掘日志模板，直接保留语义文本和结构化参数。
- 保参数：将参数类型和值显式编码，参与注意力计算。
- 多专家：用多个 source experts 表示不同源系统的异常检测经验。
- 稀疏路由：对每条日志选择最相关的 top-k 专家。
- 少标签适配：在目标系统上主要更新轻量适配参数和路由，而不是全量微调整个模型。

## 2. 五个模块分别是什么

### 模块一：免解析、保参数预处理

位置：[preprocess.py](data/preprocess.py)

`LogPreprocessor` 将原始日志转换成两部分：

- `semantic_text`：把 IP、路径、端口、用户、PID 等参数替换成 `<IP>`、`<PATH>`、`<PORT>` 这类占位符后的语义文本。
- `parameters`：按类型保存原始参数值，例如 `IP -> ["192.168.1.10"]`、`PORT -> ["22"]`。

这样既避免依赖模板解析器，又不会丢掉关键参数值。

### 模块二：参数感知注意力编码器

位置：[parameter_attention.py](models/parameter_attention.py)

该模块包含：

- `ParameterEncoder`：把参数类型和值编码成稠密向量。
- `ParameterAwareAttention`：将参数嵌入转换为 attention bias，加入文本 token 的注意力分数。
- `ParameterAwareEncoder`：融合文本表示和参数感知注意力输出，得到日志级隐藏表示。

它的作用是让模型在理解日志文本时显式关注参数，而不是只把参数当普通字符串处理。

### 模块三：多源专家池

位置：[experts.py](models/experts.py)

`ExpertPool` 由多个 `LogExpert` 组成。每个专家可以对应一个源系统、一个日志域或一类故障模式。

每个 `LogExpert` 内部包含：

- 参数感知编码器。
- DoRA 风格低秩适配投影。
- 二分类预测头。

专家输出异常概率和隐藏表示，随后由路由权重进行加权融合。

### 模块四：DoRA-AdaLoRA 轻量适配接口

位置：[dora_adalora.py](models/dora_adalora.py)

`DoRALinear` 使用冻结基础线性层，加上低秩方向更新和幅值缩放。这样可以在目标系统适配时减少可训练参数量。

`AdaLoRAController` 目前提供秩预算控制接口，后续可以加入基于重要性的动态秩分配。

### 模块五：Top-k 稀疏路由与 GMM 能量边界

位置：[router.py](models/router.py)、[gmm_energy.py](models/gmm_energy.py)

`TopKSparseRouter` 根据融合后的日志表示，为每条样本选择 top-k 个专家，并输出稀疏路由权重。

`GMMEnergy` 使用可训练的对角协方差 GMM 估计隐藏表示的负对数似然能量。能量越高，样本越可能偏离正常分布。

最终异常分数由分类器分数和能量分数组合：

```python
final_score = alpha * classifier_score + beta * energy_score
```

## 3. 如何训练 source experts

source experts 用于学习源系统中的异常检测经验。推荐流程如下：

1. 为每个源系统准备日志数据，包含原始日志和二分类标签：`0` 表示正常，`1` 表示异常。
2. 使用 `LogPreprocessor` 将原始日志转换成 `semantic_texts` 和 `parameters`。
3. 为每个源系统训练一个 `LogExpert`，或训练共享 `ExpertPool` 中对应的专家。
4. 优化目标使用二分类损失，例如 `BCEWithLogitsLoss`。
5. 可选地加入类别不平衡处理，例如异常样本加权、重采样或 focal loss。

训练单个 source expert 的伪代码：

```python
expert = LogExpert(hidden_dim=128)
optimizer = torch.optim.AdamW(expert.parameters(), lr=1e-4)
criterion = torch.nn.BCEWithLogitsLoss()

for batch in source_loader:
    parsed = preprocessor.parse_sequence(batch["raw_logs"])
    text_embeddings = text_encoder(parsed["semantic_texts"])
    output = expert(text_embeddings, parameters=parsed["parameters"])
    loss = criterion(output["logit"], batch["labels"].float())

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
```

当前模型默认请求 `bert-base-uncased` 作为文本 backbone，并通过 `BertTextEncoder` 输出 token 级隐藏状态。为了让离线 toy 流程仍可运行，如果本地没有安装 `transformers` 或没有可用 BERT 权重，模型会自动回退到 `simple-hash-encoder`；正式训练时建议安装 `requirements.txt` 中的依赖并准备好 BERT 权重。

## 4. 如何训练 router

router 的目标是为每条日志选择最合适的专家组合。推荐在 source experts 已经具备基础能力后训练 router。

训练步骤：

1. 冻结已经训练好的 source experts，避免 router 训练初期破坏专家能力。
2. 使用所有源系统混合数据，输入日志并得到融合表示 `fused_hidden`。
3. `TopKSparseRouter` 输出稀疏权重和被选中的专家。
4. `ExpertPool` 使用路由权重融合专家输出。
5. 优化主分类损失，同时加入 router 的负载均衡辅助损失。

推荐损失形式：

```python
loss = classification_loss + lambda_aux * router_aux_loss
```

其中：

- `classification_loss`：最终分类分数与真实标签之间的损失。
- `router_aux_loss`：鼓励不同专家被相对均衡地使用，避免路由长期塌缩到少数专家。
- `lambda_aux`：辅助损失权重，可从 `0.01` 或 `0.1` 开始调试。

训练稳定后，可以解冻专家的低秩适配参数进行联合微调，但建议继续冻结大部分基础参数。

## 5. 如何进行目标系统少标签适配

目标系统少标签适配用于新系统只有少量标注日志的场景。推荐采用“少动大模型、多动轻量参数”的策略。

适配步骤：

1. 用目标系统的少量标注日志构造 support set。
2. 使用同一个 `LogPreprocessor` 进行免解析、保参数预处理。
3. 加载源系统训练好的 text encoder、source experts、router 和 GMM energy。
4. 冻结文本骨干和大部分专家参数。
5. 只训练以下轻量部分：router 参数、DoRA/AdaLoRA 低秩参数、分类头，以及可选的 GMM 参数。
6. 使用少标签监督损失进行微调。
7. 在目标系统验证集上选择阈值，将 `final_score` 转换为正常/异常判断。

推荐适配目标：

```python
loss = bce_loss(final_score, labels) + lambda_energy * energy_regularization
```

如果目标系统异常标签极少，可以先用正常样本拟合 GMM 能量边界，再用少量异常样本校准阈值。

## 6. 如何评估

评估时建议同时报告分类质量、排序质量和迁移稳定性。

常用指标：

- Precision：预测为异常的样本中有多少是真的异常。
- Recall：真实异常中有多少被检出。
- F1：Precision 和 Recall 的调和平均。
- AUROC：衡量异常分数对正负样本的整体区分能力。
- AUPRC：异常样本稀少时比 AUROC 更敏感。
- FPR@Recall：在指定召回率下的误报率，适合运维场景。

推荐评估流程：

1. 在源系统训练完成后，保存模型参数和阈值。
2. 在目标系统 support set 上进行少标签适配。
3. 在目标系统 query/test set 上计算 `final_score`。
4. 使用验证集选择阈值，或固定业务可接受的召回率来确定阈值。
5. 在测试集报告 Precision、Recall、F1、AUROC 和 AUPRC。

当前骨架的 `forward` 输出包含：

- `final_score`：最终异常分数。
- `classifier_score`：专家分类器融合分数。
- `energy_score`：GMM 能量分数。
- `router_weights`：每个专家的路由权重。
- `selected_experts`：每条日志选中的 top-k 专家。
- `router_aux_loss`：router 负载均衡辅助损失。

这些输出可直接用于误报分析和专家选择可解释性分析。

## 7. 一个最小运行示例

在项目根目录运行：

```bash
python -m pa_moelog.demo_forward
```

最小示例会执行以下步骤：

1. 构造四条玩具日志。
2. 使用 `LogPreprocessor` 抽取语义文本和参数。
3. 构建 `PAMoELog(num_experts=3, top_k=2, hidden_dim=128)`。
4. 执行一次前向传播。
5. 打印最终异常分数、分类器分数、能量分数、路由权重和选中的专家。

也可以直接在 Python 中调用：

```python
from pa_moelog.data import LogPreprocessor
from pa_moelog.models import PAMoELog

logs = [
    "Failed login from 192.168.1.10 user=root port=22 error=403",
    "kernel panic at 0x000000FF pid=1234",
    "open file /etc/passwd by user admin",
    "service started successfully on port 8080",
]

preprocessor = LogPreprocessor()
batch = preprocessor.parse_sequence(logs)

model = PAMoELog(num_experts=3, top_k=2, hidden_dim=128)
output = model(
    semantic_texts=batch["semantic_texts"],
    parameters=batch["parameters"],
)

print(output["final_score"])
print(output["router_weights"])
print(output["selected_experts"])
```

当前阶段的最小示例用于验证模型结构和数据流。

## Stage 2: Training and Evaluation

第二阶段补齐了完整实验闭环：CSV 数据加载、source expert 训练、router 训练、目标系统少标签适配、统一评估、checkpoint 保存加载和 toy 数据验证。

### 准备 CSV 数据

CSV 默认字段如下：

- `log`：原始日志文本。
- `label`：二分类标签，`0` 表示正常，`1` 表示异常。
- `system`：可选字段，表示日志来源系统，例如 BGL、HDFS、Spark、Thunderbird。

示例：

```csv
log,label,system
"BGL node boot completed successfully",0,BGL
"BGL fatal machine check error code=500 node=42",1,BGL
```

项目内置了 toy CSV，可用于快速验证：

- `tests/toy_data/source_bgl.csv`
- `tests/toy_data/mixed_source.csv`
- `tests/toy_data/target_support.csv`
- `tests/toy_data/target_test.csv`

### 训练 source expert

source expert 在单个源系统上训练，使用 `BCEWithLogitsLoss`，并自动根据正负样本比例计算 `pos_weight`。checkpoint 默认保存到 `artifacts/checkpoints/source_experts/{system_name}_expert{expert_id}.pt`。

```bash
python scripts/train_source_experts.py \
  --train-csv data/processed/bgl_train.csv \
  --system-name BGL \
  --num-experts 3 \
  --expert-id 0 \
  --hidden-dim 128 \
  --batch-size 32 \
  --epochs 10 \
  --lr 1e-4 \
  --output-dir artifacts/checkpoints/source_experts \
  --device cuda
```

toy 验证命令：

```bash
python scripts/train_source_experts.py --train-csv tests/toy_data/source_bgl.csv --system-name BGL --num-experts 3 --expert-id 0 --epochs 1 --batch-size 2 --device cpu
```

### 训练 router

router 训练会加载已经训练好的 source expert checkpoint，默认冻结 source experts 的大部分参数，只训练 router、fusion encoder 和专家分类头。损失为分类损失加负载均衡辅助损失：

```python
loss = classification_loss + lambda_aux * router_aux_loss
```

示例命令：

```bash
python scripts/train_router.py \
  --train-csv data/processed/mixed_source_train.csv \
  --expert-checkpoints artifacts/checkpoints/source_experts/BGL_expert0.pt artifacts/checkpoints/source_experts/HDFS_expert1.pt artifacts/checkpoints/source_experts/Spark_expert2.pt \
  --num-experts 3 \
  --top-k 2 \
  --hidden-dim 128 \
  --batch-size 32 \
  --epochs 10 \
  --lr 1e-4 \
  --lambda-aux 0.01 \
  --output-dir artifacts/checkpoints/router \
  --device cuda
```

toy 验证命令：

```bash
python scripts/train_router.py --train-csv tests/toy_data/mixed_source.csv --expert-checkpoints artifacts/checkpoints/source_experts/BGL_expert0.pt --num-experts 3 --top-k 2 --epochs 1 --batch-size 2 --device cpu
```

### 目标系统少标签适配

目标系统适配加载 router checkpoint，默认冻结文本骨干和大部分专家参数，只训练 router、DoRA/AdaLoRA 低秩参数、分类头和 GMM 参数。损失为：

```python
loss = bce_loss + lambda_energy * energy_regularization
```

示例命令：

```bash
python scripts/adapt_target.py \
  --support-csv data/processed/thunderbird_support_20k.csv \
  --base-checkpoint artifacts/checkpoints/router/router.pt \
  --target-system Thunderbird \
  --batch-size 32 \
  --epochs 5 \
  --lr 5e-5 \
  --lambda-energy 0.1 \
  --output-dir artifacts/checkpoints/target_adapt \
  --device cuda
```

toy 验证命令：

```bash
python scripts/adapt_target.py --support-csv tests/toy_data/target_support.csv --base-checkpoint artifacts/checkpoints/router/router.pt --target-system Thunderbird --epochs 1 --batch-size 2 --device cpu
```

### 统一评估

评估脚本会输出 Precision、Recall、F1、AUROC、AUPRC、混淆矩阵、平均 final score、平均 classifier score 和平均 energy score。它还会保存逐条日志预测结果到 `artifacts/outputs/predictions.csv`。

示例命令：

```bash
python scripts/evaluate.py \
  --test-csv data/processed/thunderbird_test.csv \
  --checkpoint artifacts/checkpoints/target_adapt/Thunderbird_adapted.pt \
  --batch-size 64 \
  --threshold 0.5 \
  --device cuda \
  --output-json artifacts/outputs/thunderbird_eval.json
```

toy 验证命令：

```bash
python scripts/evaluate.py --test-csv tests/toy_data/target_test.csv --checkpoint artifacts/checkpoints/target_adapt/Thunderbird_adapted.pt --batch-size 2 --device cpu --output-json artifacts/outputs/toy_eval.json
```

## LogEvol 实验：Hadoop2 + Hadoop3 + Spark2 -> Spark3

当前推荐的跨系统实验配置是：

- 源系统：`hadoop2`、`hadoop3`、`spark2`
- 目标系统：`spark3`
- 当前指标：只报告 `Precision`、`Recall`、`F1`

先准备三源 CSV。该命令会生成每个源系统的 CSV，并额外生成不包含 `spark3` 的三源混合数据：

```bash
python scripts/prepare_logevol.py \
  --input-root D:\GoogleDownload\Logevol \
  --output-dir data/processed/logevol_hadoop2_hadoop3_spark2 \
  --datasets hadoop2 hadoop3 spark2
```

目标系统 `spark3` 使用原始转换目录：

```bash
python scripts/prepare_logevol.py \
  --input-root D:\GoogleDownload\Logevol \
  --output-dir data/processed/logevol \
  --datasets spark3
```

训练三个 source experts：

```bash
python scripts/train_source_experts.py \
  --train-csv data/processed/logevol_hadoop2_hadoop3_spark2/hadoop2/train.csv \
  --system-name hadoop2 \
  --num-experts 3 \
  --expert-id 0 \
  --batch-size 16 \
  --epochs 3 \
  --device cpu \
  --backbone-name simple-hash-encoder
```

```bash
python scripts/train_source_experts.py \
  --train-csv data/processed/logevol_hadoop2_hadoop3_spark2/hadoop3/train.csv \
  --system-name hadoop3 \
  --num-experts 3 \
  --expert-id 1 \
  --batch-size 16 \
  --epochs 3 \
  --device cpu \
  --backbone-name simple-hash-encoder
```

```bash
python scripts/train_source_experts.py \
  --train-csv data/processed/logevol_hadoop2_hadoop3_spark2/spark2/train.csv \
  --system-name spark2 \
  --num-experts 3 \
  --expert-id 2 \
  --batch-size 16 \
  --epochs 3 \
  --device cpu \
  --backbone-name simple-hash-encoder
```

训练 router。这里使用的 mixed 数据只包含 `hadoop2+hadoop3+spark2`，不会包含目标系统 `spark3`：

```bash
python scripts/train_router.py \
  --train-csv data/processed/logevol_hadoop2_hadoop3_spark2/mixed/train.csv \
  --expert-checkpoints \
    artifacts/checkpoints/source_experts/hadoop2_expert0.pt \
    artifacts/checkpoints/source_experts/hadoop3_expert1.pt \
    artifacts/checkpoints/source_experts/spark2_expert2.pt \
  --num-experts 3 \
  --top-k 2 \
  --batch-size 16 \
  --epochs 3 \
  --device cpu \
  --backbone-name simple-hash-encoder
```

在目标系统 `spark3` 上做少标签适配。这里先用 `valid.csv` 作为 support set：

```bash
python scripts/adapt_target.py \
  --support-csv data/processed/logevol/spark3/valid.csv \
  --base-checkpoint artifacts/checkpoints/router/router.pt \
  --target-system spark3 \
  --batch-size 16 \
  --epochs 3 \
  --device cpu \
  --backbone-name simple-hash-encoder
```

最后在 `spark3` 测试集上评估，只输出 Precision、Recall、F1：

```bash
python scripts/evaluate.py \
  --test-csv data/processed/logevol/spark3/test.csv \
  --checkpoint artifacts/checkpoints/target_adapt/spark3_adapted.pt \
  --batch-size 32 \
  --device cpu \
  --output-json artifacts/outputs/spark3_basic_metrics.json \
  --backbone-name simple-hash-encoder \
  --basic-metrics-only
```

如果要使用本地 BERT，将上述命令中的：

```bash
--backbone-name simple-hash-encoder
```

替换为：

```bash
--backbone-name E:\demo\models\bert-base-uncased --no-hash-fallback
```
