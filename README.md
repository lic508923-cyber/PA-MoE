# PA-MoELog

PA-MoELog 是面向跨系统、少标签日志异常检测的多源专家模型。

```text
事件文本 + 独立参数 token
  → 带 token mask 的参数感知事件编码
  → 带事件位置编码的序列 Transformer
  → 轻量源专家与支持集融合
  → DoRA 目标适配
  → 源专家 logit + 目标残差 logit
  → 低维 GMM 能量
```

## 当前实现的稳定性约束

- 文本编码器返回 token mask；参数融合和事件池化均忽略 padding。
- 事件 Transformer 在每个事件位置加入可学习位置编码，因此事件顺序会影响表示。
- 模型只注册一个 `ParameterEncoder`，checkpoint 默认严格加载。
- session 模式要求每行都具有 `session_id`；整个 session 不会跨窗口或数据集合。
- window 模式要求可解析的 `timestamp`，先按时间排序，再创建完整滑动窗口。
- 原始事件必须先切分 train/support/validation/test，再创建窗口。
- 多源训练按源系统平衡采样，并对每个系统分别计算类别权重。
- 最终源 checkpoint 来自各源系统 validation 宏平均 AUPRC 最优的 epoch，而不是最后一个 epoch。
- GMM 先通过正常 support 的 PCA/SVD 投影到默认 32 维空间；每 20 个正常样本才允许增加一个分量。
- 少标签融合按照 `n/(n+shrinkage)` 向训练专家的均匀先验收缩。

## 数据格式

事件分类至少需要：

```csv
log,label,system
"node boot completed",0,BGL
"fatal machine check error",1,BGL
```

序列实验需要时间戳，并可选择 session：

```csv
log,label,system,session_id,timestamp
"job started",0,BGL,s1,2026-01-01T00:00:00Z
"job failed",1,BGL,s1,2026-01-01T00:00:01Z
```

时间戳支持 Unix 数值或 ISO-8601。session 列不能部分为空。

## 严格数据切分

切分发生在建窗之前；有 session 时以完整 session 为最小单位：

```bash
python scripts/prepare_splits.py \
  --input-csv data/all_events.csv \
  --output-dir data/splits \
  --train-ratio 0.6 --support-ratio 0.1 --validation-ratio 0.1
```

生成四份原始事件 CSV、四份独立构造的 `*_sequences.csv` 和 `split_manifest.json`。如果 session 在时间上重叠，无法形成严格时间边界时脚本会报错，而不是产生泄漏切分。

## 多源训练

正式训练必须提供独立 validation：

```bash
python scripts/train_multisource.py \
  --train-csv data/splits/train.csv \
  --validation-csv data/source_validation.csv \
  --hidden-dim 128 --epochs 20 --patience 5 \
  --backbone-name E:\\demo\\models\\bert-base-uncased \
  --no-hash-fallback \
  --output artifacts/checkpoints/multisource.pt
```

序列模式增加：

```text
--sequence --window-size 20 --stride 10 --max-events 512
```

checkpoint 保存完整专家池、`system_to_expert`、训练专家掩码、源正常原型、最佳 epoch 和 validation 指标。
训练使用系统轮转批次、按系统平均的加权 BCE、固定随机种子、early stopping 和基于宏 AUPRC 的学习率调度。

## 目标域适配

```bash
python scripts/adapt_target.py \
  --support-csv data/splits/support.csv \
  --validation-csv data/splits/validation.csv \
  --base-checkpoint artifacts/checkpoints/multisource.pt \
  --target-system Thunderbird --epochs 5 \
  --output-dir artifacts/checkpoints/target_adapt
```

目标适配自动完成归一化余弦专家距离、标签预算收缩、BCE 适配、低维 GMM 拟合、固定能量统计，以及 validation 上的 `alpha`、`beta`、threshold 选择。

## 评估与测试

```bash
python scripts/evaluate.py \
  --test-csv data/splits/test.csv \
  --checkpoint artifacts/checkpoints/target_adapt/Thunderbird_adapted.pt \
  --output-json artifacts/outputs/eval.json

python -m unittest discover -s tests -p "test_*.py" -v
```

未指定 `--threshold` 时使用目标 checkpoint 中由 validation 选出的阈值。hash 编码器仅供调试，脚本要求同时传入 `--backbone-name simple-hash-encoder --debug-hash-encoder`；正式实验默认禁止 BERT 加载失败后静默回退。

正式实验应使用 3～5 个不同 `--seed` 独立训练，并汇总结果：

```bash
python scripts/summarize_multiseed.py seed1.json seed2.json seed3.json --output summary.json
```
