# LogEvol Spark3 F1 优化实验报告

## 一、实验目标

本实验在真实 LogEvol 日志数据集上评估 PA-MoELog 的跨系统异常检测能力，并在不使用测试集调参的前提下，通过源域训练、目标域适配、专家融合、GMM 能量建模和精确阈值选择，提高 Spark3 目标域的 F1 分数。

## 二、实验环境

- 操作系统：Windows 11 64 位
- GPU：NVIDIA GeForce RTX 5060 Ti，16GB 显存
- Python：3.10.20，Conda 环境 `pa-moelog`
- PyTorch：2.8.0+cu128
- CUDA：12.8
- Transformers：5.13.0
- 文本主干：本地 `bert-base-uncased`
- 固定随机种子：`7`

## 三、数据集与实验划分

实验使用真实 LogEvol 会话级数据：

- 源域：Hadoop2、Hadoop3、Spark2
- 目标域：Spark3

数据构造脚本固定随机种子，并检查 Spark3 support、validation、test 的 `session_id` 是否重叠。测试集不参与训练、超参数选择、适配轮数选择或阈值选择，只在最终模型确定后评估一次。

| 数据划分 | 正常样本 | 异常样本 | 总数 |
|---|---:|---:|---:|
| 源域训练集 | 15,000 | 2,219 | 17,219 |
| 源域验证集 | 3,000 | 429 | 3,429 |
| Spark3 支持集 | 400 | 20 | 420 |
| Spark3 验证集 | 800 | 20 | 820 |
| Spark3 测试集 | 4,167 | 79 | 4,246 |

## 四、实现的优化方法

### 4.1 无泄漏的数据构造

新增 `scripts/build_logevol_f1_experiment.py`，采用分层随机抽样分别构造源域训练集、源域验证集、目标支持集、目标验证集和目标测试集。

脚本包含以下约束：

- 支持集、验证集和测试集的会话不能重叠；
- 每个数据划分必须同时包含正常类和异常类；
- 随机种子被写入实验清单；
- 测试数据不能进入模型选择过程。

### 4.2 源域均衡训练

源域模型采用以下方法训练：

- 三个源系统按系统轮转并均衡组成批次；
- 为每个源系统单独计算正类权重，缓解类别不平衡；
- 冻结预训练 BERT 主干，训练 PA-MoELog 的投影层、参数感知编码层、序列编码器和专家层；
- 使用独立源域验证集的宏平均 AUPRC 选择 checkpoint；
- 使用学习率调度和早停，避免使用最后一个 epoch 的模型。

最佳源模型出现在第 9 个 epoch，源验证集宏平均 AUPRC 为 `0.9497`。

### 4.3 支持集引导的专家融合

利用 Spark3 支持集中的正常样本计算目标域与三个源专家正常原型之间的归一化余弦距离。得到的专家融合权重为：

| 源专家 | 融合权重 |
|---|---:|
| Hadoop2 | 0.3181 |
| Hadoop3 | 0.3145 |
| Spark2 | 0.3675 |

Spark3 与 Spark2 的表示距离最小，因此 Spark2 专家获得最高权重。

### 4.4 目标域全量适配

对比了以下目标域适配方式：

- 仅分类头适配；
- DoRA 参数高效适配；
- 全量参数适配。

在固定 `seed=7` 的公平比较中，全量适配获得最高验证 F1。最佳配置为：

- 适配方式：`full`
- batch size：`32`
- 学习率：`1e-5`
- 训练轮数：`5`
- 专家融合：`support-guided`
- GMM：启用

### 4.5 GMM 能量异常检测

使用 Spark3 支持集中的正常样本拟合低维 GMM，并将分类器分数与归一化能量分数组合。

验证集消融结果：

| 方法 | 验证 F1 |
|---|---:|
| 最佳全量适配 + GMM | **0.9500** |
| 最佳 DoRA 适配 + GMM | 0.8421 |
| DoRA 适配但禁用 GMM | 0.5854 |

结果说明 GMM 能量建模是本次跨系统异常检测性能的重要来源。

### 4.6 精确 F1 阈值选择

原实现只搜索 `0.05、0.10、……、0.95` 等粗粒度阈值，可能错过最佳工作点。本实验新增 `select_best_f1_threshold`：

- 对验证集上的每个不同预测分数进行候选阈值评估；
- 相同分数作为一个整体处理；
- 直接选择验证 F1 最大的精确阈值；
- F1 相同时选择更高阈值，以减少异常误报。

验证集最终选择：

- 分类器权重 `alpha=0.05`
- GMM 能量权重 `beta=0.95`
- 阈值 `0.9934782981872559`

## 五、超参数搜索结果

固定随机种子后，在最佳学习率附近搜索目标域适配轮数：

| 适配轮数 | Spark3 验证 F1 |
|---:|---:|
| 3 | 0.8837 |
| 4 | 0.9000 |
| 5 | **0.9500** |
| 6 | 0.8649 |
| 7 | 0.8947 |

因此在查看测试集之前，将 5 个 epoch 的模型确定为最终模型。

## 六、最终测试结果

在完全隔离的 Spark3 测试集上得到：

| 指标 | 数值 |
|---|---:|
| Precision | 0.8871 |
| Recall | 0.6962 |
| F1 | **0.7801** |
| AUROC | 0.9672 |
| AUPRC | 0.7765 |
| FPR | 0.00168 |
| TN | 4,160 |
| FP | 7 |
| FN | 24 |
| TP | 55 |

旧实验结果为：

- Precision：`1.0000`
- Recall：`0.4051`
- F1：`0.5766`

新方法将测试 F1 从 `0.5766` 提升到 `0.7801`，绝对提升 `0.2036`，相对提升约 `35.3%`。主要改进来自召回更多异常样本，同时只产生 7 个假阳性。

## 七、效率指标

最佳目标模型的效率统计：

| 指标 | 数值 |
|---|---:|
| 可训练参数量 | 110,481,765 |
| 总参数量 | 110,481,765 |
| 目标适配耗时 | 18.23 秒 |
| CUDA 峰值显存 | 约 2.96GB |
| checkpoint 大小 | 约 442MB |

## 八、复现命令

在仓库根目录中使用 `pa-moelog` Conda 环境执行：

```powershell
python scripts/build_logevol_f1_experiment.py --seed 7

python scripts/train_multisource.py `
  --train-csv data/processed/experiments/f1_optimized/source_train.csv `
  --validation-csv data/processed/experiments/f1_optimized/source_validation.csv `
  --hidden-dim 128 --batch-size 64 --epochs 10 --patience 3 `
  --scheduler-patience 1 --lr 0.0003 --device cuda --seed 7 `
  --backbone-name E:/chenli/LAD/models/bert-base-uncased --freeze-backbone `
  --output artifacts/checkpoints/f1_optimized/source_seed7.pt

python scripts/adapt_target.py `
  --support-csv data/processed/experiments/f1_optimized/target_support.csv `
  --validation-csv data/processed/experiments/f1_optimized/target_validation.csv `
  --base-checkpoint artifacts/checkpoints/f1_optimized/source_seed7.pt `
  --target-system spark3 --batch-size 32 --epochs 5 --lr 0.00001 `
  --fusion support-guided --adaptation full --device cuda --seed 7 `
  --output-dir artifacts/checkpoints/f1_optimized/fixed_full_e5

python scripts/evaluate.py `
  --test-csv data/processed/experiments/f1_optimized/target_test.csv `
  --checkpoint artifacts/checkpoints/f1_optimized/fixed_full_e5/spark3_adapted.pt `
  --batch-size 64 --device cuda `
  --output-json artifacts/outputs/f1_optimized/spark3_test_seed7.json
```

## 九、结果文件

```text
最佳源模型：
artifacts/checkpoints/f1_optimized/source_seed7.pt

最佳目标模型：
artifacts/checkpoints/f1_optimized/fixed_full_e5/spark3_adapted.pt

最终测试指标：
artifacts/outputs/f1_optimized/spark3_test_seed7.json
```

## 十、结论与限制

本实验通过严格数据隔离、源域均衡训练、支持集引导专家融合、目标域全量适配、GMM 能量建模以及精确验证阈值选择，将真实 Spark3 测试集 F1 提升至 `0.7801`。

该结果是本次固定数据划分和超参数搜索范围内的最优结果，不代表数学意义上的全局最优。当前结果基于一个固定随机种子；用于正式论文时，应进一步运行 3～5 个独立随机种子并报告均值、标准差和置信区间。
