# PA-MoELog

PA-MoELog 是一个面向跨系统少样本日志异常检测的多源专家模型。当前实现已经形成完整训练闭环：

```text
日志事件序列
  → 共享文本编码器
  → 独立参数 token 跨注意力
  → 事件序列 Transformer
  → 轻量源专家池
  → 支持集引导的静态融合
  → 共享 DoRA 目标适配
  → 源专家 logit + 目标残差 logit
  → 目标正常样本 GMM 能量
```

## 七项关键实现

1. 每个 IP、PORT、PATH、USER 等参数保留为独立 token，并使用 padding mask；文本 token 通过跨注意力读取参数。
2. 所有源系统在同一个模型中训练，共享文本、参数和序列编码器；checkpoint 保存完整专家池、系统映射、训练掩码和正常原型。
3. `LogSequenceDataset` 支持 `session_id` 分组或按时间排序的固定事件数滑动窗口，模型使用事件级 Transformer 编码 `[B,T,H]`。
4. 目标适配自动计算目标正常支持集与各源专家正常原型的距离，并校准静态融合权重；未训练专家始终被屏蔽。
5. `DoRALinear` 实现 `W=m·normalize(W0+BA)`，方形基础映射以单位矩阵初始化。
6. 最终分类 logit 为源专家融合 logit 与零初始化目标残差 logit 之和。
7. GMM 使用 K-means 风格初始化和 EM 拟合对角协方差，混合数受正常样本数量约束；能量只使用支持集拟合的固定位置/尺度，绝不按测试 batch 归一化。

## 数据格式

事件级 CSV 至少包含：

```csv
log,label,system
"node boot completed",0,BGL
"fatal machine check error",1,BGL
```

序列实验可增加：

```csv
log,label,system,session_id,timestamp
"job started",0,BGL,s1,2026-01-01T00:00:00
"job failed",1,BGL,s1,2026-01-01T00:00:01
```

- 存在非空 `session_id` 时按 session 分组。
- 没有 session 时按 `system` 和 `timestamp` 排序，再使用 `--window-size`、`--stride` 创建窗口。
- 窗口标签默认是窗口内事件标签的最大值。
- 正式实验应先按时间切分 train/support/validation/test，再创建窗口，避免时间泄漏。

## 训练完整多源模型

```bash
python scripts/train_multisource.py \
  --train-csv tests/toy_data/mixed_source.csv \
  --hidden-dim 128 --epochs 10 --batch-size 32 \
  --backbone-name E:\\demo\\models\\bert-base-uncased \
  --no-hash-fallback \
  --output artifacts/checkpoints/multisource.pt
```

序列模式增加：

```text
--sequence --window-size 20 --stride 10
```

`scripts/train_source_experts.py` 仅作为单专家兼容入口保留；正式多源实验应使用 `train_multisource.py`，避免随机专家混入融合。

## 目标域适配

适配阶段自动执行：

1. 从 checkpoint 读取源正常原型和训练专家掩码；
2. 用目标正常 support 校准专家融合权重；
3. 只用 BCE 更新 DoRA、目标归一化层和残差分类头；
4. 适配后用正常 support 拟合 GMM；
5. 保存固定能量归一化统计；
6. 如果提供 validation，搜索 `alpha`、`beta` 和 threshold。

```bash
python scripts/adapt_target.py \
  --support-csv tests/toy_data/target_support.csv \
  --validation-csv path/to/target_validation.csv \
  --base-checkpoint artifacts/checkpoints/multisource.pt \
  --target-system Thunderbird --epochs 5 \
  --backbone-name E:\\demo\\models\\bert-base-uncased \
  --output-dir artifacts/checkpoints/target_adapt
```

适配损失不再包含基于随机 GMM 的 energy loss。GMM 只在判别适配完成后拟合。

## 评估

```bash
python scripts/evaluate.py \
  --test-csv tests/toy_data/target_test.csv \
  --checkpoint artifacts/checkpoints/target_adapt/Thunderbird_adapted.pt \
  --output-json artifacts/outputs/eval.json \
  --backbone-name E:\\demo\\models\\bert-base-uncased
```

未显式传入 `--threshold` 时，评估脚本使用适配 checkpoint 中由 validation 选出的阈值。序列 checkpoint 的评估需传入与训练一致的 `--sequence --window-size --stride`。

## 快速离线验证

```bash
python -m pa_moelog.demo_forward
python -m unittest discover -s tests -p "test_*.py" -v
```

没有本地 BERT 时可用 `--backbone-name simple-hash-encoder` 做接口和冒烟测试；正式结果不应使用 hash encoder。
