# BGL + Hadoop + OpenStack → HDFS 实验报告

## 1. 实验结论

本实验在 SSH 远程 Windows 主机 `DESKTOP-PRHOFJF` 上完成，使用 NVIDIA GeForce
RTX 5060 Ti、Python 3.10、PyTorch 2.8.0+cu128 和真实 Loghub 数据。

在不查看测试标签调参的前提下，最终 HDFS 测试 F1 为 **0.7890**。相比此前以
OpenStack 为目标域时的 pooled F1 `0.0244`，HDFS 作为目标域明显更可靠。这主要来自
HDFS 的 Block ID 标签粒度稳定且异常 block 数量充足，而不是降低了测试集难度或制造
数据泄漏。

## 2. 无泄漏数据设计

源域为 BGL、Hadoop、OpenStack，目标域为 HDFS。

- BGL：先按时间切分，再构造互不重叠的 20 事件窗口。
- Hadoop：按 Application ID 划分，同一应用的所有 chunk 只能出现在一个集合。
- OpenStack：按 VM UUID 划分，同一 VM 的 5 事件窗口不能跨训练和验证集合。
- HDFS：按完整 Block ID 分层划分，支持集、验证集、测试集没有相同 block。

HDFS 目标域会话统计：

| 集合 | 正常 block | 异常 block | 总数 |
|---|---:|---:|---:|
| support | 3,883 | 117 | 4,000 |
| validation | 1,941 | 59 | 2,000 |
| test | 3,883 | 117 | 4,000 |

三个集合两两之间的 Block ID 交集均为 0。测试集在模型和阈值锁定后只评估一次。

## 3. 源域专家训练

三个独立专家分别对应 BGL、Hadoop 和 OpenStack，训练使用系统均衡批次。源模型最佳
epoch 为 3，源域验证 Macro-AUPRC 为 **0.6802**。

HDFS support 对三个源专家校准后的融合权重为：

| 专家 | 权重 |
|---|---:|
| BGL | 0.3244 |
| Hadoop | 0.3123 |
| OpenStack | 0.3633 |

三个专家均参与决策，没有发生单专家塌缩。

## 4. 目标域适配选择

仅根据 HDFS validation 选择模型。完成了三种不同复杂度的适配结构：

| 模式 | 学习率/轮数 | 训练参数 | 验证 F1 | 验证 AUPRC |
|---|---|---:|---:|---:|
| Head-only | 1e-3 / 8 | 129 | 0.5610 | 0.4889 |
| Legacy DoRA（旧后置适配器） | 5e-4 / 8 | 1,537 | 0.5814 | 0.4920 |
| Partial | 3e-4 / 5 | 942,181 | **0.8319** | **0.8334** |

因此锁定 Partial 模型。其验证集工作点为：alpha `0.60`、beta `0.40`、阈值
`0.91917920`。alpha 不再过低，且分类器单独 F1 为 `0.8174`、GMM 单独 F1 为
`0.6949`，说明最终模型不是完全依赖 GMM。

注：该 DoRA 结果来自架构重设计前的“融合后单层 DoRA”，不能代表后来新增的
“专家投影 DoRA + 目标条件门控”；新结构需要作为独立实验重新训练和评估。

## 5. 唯一一次测试结果

| 指标 | 数值 |
|---|---:|
| Precision | 0.8515 |
| Recall | 0.7350 |
| **F1** | **0.7890** |
| AUROC | 0.9851 |
| AUPRC | 0.8265 |
| FPR | 0.00386 |

混淆矩阵：TN=3,868、FP=15、FN=31、TP=86。

Partial 训练参数占总参数的 `0.8532%`，远程适配耗时约 `578.7` 秒，峰值 CUDA
显存约 `1.98 GB`。

## 6. 可复现位置

- 数据生成：`scripts/prepare_loghub_hdfs_target.py`
- 目标配置编排：`scripts/run_hdfs_target_grid.py`
- 数据清单：`data/processed/loghub_hdfs_target/crossdomain_manifest.json`
- 远程源模型：`artifacts/checkpoints/loghub_hdfs_target/source_seed42.pt`
- 远程最终模型：`artifacts/checkpoints/loghub_hdfs_target/adapt/partial_lr3e4_e5/HDFS_adapted.pt`
- 远程测试结果：`artifacts/outputs/loghub_hdfs_target/test.json`

本结果是单随机种子、单次测试评估。若用于论文主结果，应继续补充至少 3 个随机种子，
报告均值和标准差；不能在已经查看本次测试结果后继续针对同一测试集调参。

## 7. 专家投影 DoRA + 动态门控探索性复跑

在看到上述测试结果后，模型将旧的融合后 DoRA 重构为三个专家各自的投影 DoRA，并
增加样本级目标条件门控。为了与旧结果直接比较，复跑沿用了相同 HDFS 划分，因此本节
属于探索性同划分对照，不是新的盲测结果。

固定配置为 `rank=4`、`dora_alpha=4`、学习率 `5e-4`、8轮、seed 42；没有在查看测试
结果后追加配置。训练参数为 3,076，占总参数的 `0.002785%`。

| 指标 | Validation | Test |
|---|---:|---:|
| Precision | 0.7297 | 0.8333 |
| Recall | 0.4576 | 0.3846 |
| F1 | 0.5625 | 0.5263 |
| AUPRC | 0.5134 | 0.4858 |
| AUROC | — | 0.8161 |

验证选择得到 `alpha=0.15`、`beta=0.85`、阈值 `0.95665640`。测试混淆矩阵为
TN=3,874、FP=9、FN=72、TP=45。模型保持很低误报，但异常召回不足，是 F1 未超过
Legacy DoRA 和 Partial 的直接原因。

动态门控没有塌缩。测试集平均专家权重为：

| 标签 | BGL | Hadoop | OpenStack | 门控熵 |
|---|---:|---:|---:|---:|
| 正常 | 0.4810 | 0.4165 | 0.1026 | 0.9302 |
| 异常 | 0.2921 | 0.6002 | 0.1078 | 0.8425 |

门控能够把异常 block 更多地路由到 Hadoop 专家，但 rank-4 专家投影仍不足以重构 HDFS
异常表示。新结构证明了动态路由可以学习，但当前单一低秩位置仍然欠拟合；不能因为
此次 F1 较低就继续针对同一测试集试验更多 rank、学习率或阈值。下一步需要新随机划分
或新测试集，并预注册 `rank ∈ {4, 8, 16}`、不同门控正则和多随机种子实验。

远程结果：
`artifacts/outputs/loghub_hdfs_target/expert_dora_r4_a4_lr5e4_e8_test.json`

该淘汰配置的大型模型检查点已在代码整理时清理，指标和结果文件保留。

## 8. Deep-DoRA 与 Partial 的验证集容量对照

为检验 DoRA 能否达到 Partial 的效果，在远程 Windows 主机的同一 HDFS support/validation
划分上进行了三项验证集实验。三项实验均使用学习率 `3e-4`、5 轮、seed 42；没有根据本轮结果
再次评估已被使用过的 HDFS test。预先规定验证 F1 至少达到 `0.80` 才进入测试。

Deep-DoRA 将 DoRA 权重参数化扩展到非 BERT 任务路径，包括参数融合注意力/FFN、序列
Transformer 注意力/FFN，以及三个专家的投影和分类矩阵；原始矩阵和 BERT 保持冻结。它还训练
非 BERT LayerNorm、位置/类型嵌入、目标条件专家门控和目标分类头。

| 适配方式 | 可训练参数 | 比例 | 验证 F1 | AUPRC | Precision | Recall | alpha |
|---|---:|---:|---:|---:|---:|---:|---:|
| Partial | 942,181 | 0.8532% | **0.8319** | 0.8334 | — | — | 0.60 |
| Deep-DoRA，rank 8 | 57,728 | 0.0523% | 0.6136 | 0.6219 | 0.9310 | 0.4576 | 0.10 |
| Deep-DoRA + Value Embedding，rank 8 | 582,016 | 0.5269% | 0.6237 | 0.6109 | 0.8529 | 0.4915 | 0.05 |
| Deep-DoRA，rank 64 | 355,368 | 0.3208% | **0.7742** | **0.8364** | 0.7385 | 0.8136 | **0.80** |

rank 8 下仅解冻参数值嵌入只增加 `0.0101` F1，且 alpha 下降，因此值嵌入不是主要瓶颈。
把 rank 提高到 64 后，F1 增加 `0.1606`、召回达到 `0.8136`，alpha 恢复到 `0.80`；这证明
当前跨系统 HDFS 迁移需要较大的更新子空间，旧 DoRA 过弱主要是容量不足，而不是 GMM 本身必然
占主导。rank-64 的 AUPRC 已略高于 Partial，但 F1 仍低 `0.0577`，没有达到预设的 `0.80`
门槛，因此不能声称与 Partial 等效，也没有运行测试集。

工程上有两种合理选择：若首要目标是最高 F1，继续使用已验证的 Partial；若更看重参数效率，
采用 rank-64 Deep-DoRA，它只训练 Partial 的约 `37.7%` 参数，并在验证集获得 Partial 约
`93.1%` 的 F1，同时分类分支占主导。若还要研究 DoRA 达到 Partial，应在新的锁定划分上预注册
分层 rank（序列 FFN/注意力高 rank、专家中 rank、融合层低 rank）以及偏置更新，而不是继续针对
当前验证集逐项试参。

远程保留的最佳模型：

- `artifacts/checkpoints/loghub_hdfs_target/adapt_deep_dora_r64_a64_lr3e4_e5/HDFS_adapted.pt`

rank-8 和 Value Embedding 消融仅保留各自目录中的 validation/efficiency JSON，大型模型
检查点已清理。源模型和最佳 Partial 检查点继续保留。
