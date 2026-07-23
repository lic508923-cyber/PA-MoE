# HDFS + BGL + Hadoop → OpenStack 跨域异常检测实验报告

## 1. 结论先行

本实验使用真实 Loghub 数据和本地 `bert-base-uncased`，将 HDFS、BGL、Hadoop
分别作为三个源专家，OpenStack 作为目标域。实验严格按原始标签粒度分组，目标域采用
4 折 leave-one-anomalous-VM-out，测试 VM 不参与该折的适配、模型选择、alpha 或阈值选择。

主要结论如下：

1. 源模型训练有效：最佳 epoch 为 5，三个源域验证集宏平均 AUPRC 为 `0.8498`。
2. `alpha` 不是三个专家的权重，而是分类器分数与 GMM 能量分数的融合系数。
3. 旧代码会在 F1 并列时机械保留从 0 开始遇到的较小 alpha；修复后 alpha 不再系统性偏低。
4. 三个专家没有塌缩。整 VM 实验的融合权重为：BGL `0.3593`、HDFS `0.2865`、
   Hadoop `0.3542`。
5. 严格四折测试的宏平均 F1 只有 `0.0144 ± 0.0170`，平均 AUROC 为
   `0.8175 ± 0.1346`。模型有一定排序能力，但阈值与异常 VM 之间的泛化很差。
6. 将 OpenStack 切为 5 事件窗口后，平均 alpha 达到 `0.7000 ± 0.0913`，但平均 F1
   反而降至 `0.0067 ± 0.0079`。因此不能把“alpha 正常”当作“模型有效”或“F1 最高”的证据。

## 2. 实验环境

- 操作系统：Windows
- GPU：NVIDIA GeForce GTX 1650 Ti，4 GB 显存
- Python 环境：`logformer_env`
- PyTorch：`2.5.1+cu121`
- Transformers：`4.57.6`
- 文本主干：`E:/demo/models/bert-base-uncased`
- 主干训练方式：冻结 BERT，训练参数感知层、序列层和专家层
- 随机种子：源训练 `42`；目标折使用 `42 + fold`

旧报告中的 RTX 5060 Ti 环境不是本次实验环境，本报告没有沿用该硬件描述。

## 3. 数据设计

### 3.1 原始标签粒度

| 数据集 | 角色 | 分组与标签单位 |
|---|---|---|
| HDFS_v1 | 源域专家 | Block ID；完整 block trace 不能跨集合 |
| BGL | 源域专家 | 按时间边界切分后建立 20 行非重叠窗口 |
| Hadoop | 源域专家 | Application ID；每个 application 最多取 4 个等间隔 32 事件 chunk |
| OpenStack | 目标域 | 仅使用能由 `[instance: UUID]` 可靠归属的日志，按 VM 隔离 |

没有把同一 block、application 或 VM 的日志行随机分散到不同集合。HDFS 与 Hadoop
采用分层 group split；BGL 先按时间边界切分，再在各集合内部建窗。

### 3.2 本次计算预算下的源域样本

| 划分 | 系统 | 会话数 | 正常 | 异常 | 事件数 |
|---|---|---:|---:|---:|---:|
| train | BGL | 750 | 686 | 64 | 15,000 |
| train | HDFS | 2,700 | 2,621 | 79 | 51,988 |
| train | Hadoop | 200 | 40 | 160 | 5,527 |
| validation | BGL | 750 | 690 | 60 | 15,000 |
| validation | HDFS | 300 | 291 | 9 | 5,793 |
| validation | Hadoop | 20 | 4 | 16 | 554 |

源 train 与 validation 的父会话交集为 0。训练批次按系统轮转，避免 HDFS 的样本量
淹没另外两个专家；每个系统独立计算正类权重。

### 3.3 OpenStack 四折设计

可可靠归属的 OpenStack VM 共 2,070 个，其中明确标注异常的 VM 只有 4 个。每折使用：

- support：100 个正常 VM + 2 个异常 VM；
- validation：100 个正常 VM + 1 个异常 VM；
- test：300 个正常 VM + 1 个异常 VM；
- 每个异常 VM 恰好作为 test 一次；同一 VM 不会跨 support/validation/test。

这个设计避免泄漏，但每折只有一个测试异常 VM，F1 和阈值天然具有极高方差。

## 4. alpha 偏低的实现原因与修复

最终分数为：

```text
final_score = alpha × classifier_score + (1 - alpha) × gmm_energy_score
```

三个专家先通过 `fusion_weights` 融合，再进入分类器分支。因此：

- `alpha` 低表示该折 validation 更偏向 GMM；
- `alpha` 低不等于专家没有发挥作用；
- 判断专家作用应查看 `fusion_weights`、classifier-only 消融和跨折泛化。

原实现按 `0.00, 0.05, ..., 1.00` 搜索 alpha，只在 F1 严格提高时更新。由于每个 alpha
都会重新选择阈值，稀少异常场景会出现大片 F1 并列区间，旧实现必然选到并列区间中最小的
alpha。修复后的选择顺序为：

1. 最大 validation F1；
2. F1 容差内最大 validation AUPRC；
3. 两者仍并列时，选择最接近预注册先验 `0.7` 的 alpha；
4. 不设置 alpha 下限，并完整记录 F1/AUPRC 并列区间。

同时新增冻结 BERT 的 `partial` 适配方式，并保留 head-only、DoRA 和 full。

## 5. 训练与目标域候选

源训练使用 batch size 8、6 个 epoch 上限、学习率 `3e-4`、宏 AUPRC 早停。结果：

| Epoch | validation macro-AUPRC |
|---:|---:|
| 1 | 0.7035 |
| 2 | 0.5566 |
| 3 | 0.7567 |
| 4 | 0.8241 |
| 5 | **0.8498** |
| 6 | 0.8453 |

每个 OpenStack 外层折独立比较以下候选，只用该折 validation 选择：

| 候选 | 学习率 | Epoch | 可训练参数量 |
|---|---:|---:|---:|
| head-only | 1e-3 | 5 | 129 |
| DoRA | 5e-4 | 5 | 1,537 |
| DoRA | 1e-3 | 5 | 1,537 |
| partial（冻结 BERT） | 1e-4 | 3 | 942,181 |

候选的 validation F1、AUPRC 相同时，选择可训练参数更少的模型。模型锁定后才评估该折 test。

## 6. 确认性实验：完整 VM 序列

| Fold | validation 选择 | validation F1 | alpha | Test P | Test R | Test F1 | AUROC | AUPRC |
|---:|---|---:|---:|---:|---:|---:|---:|---:|
| 0 | head-only | 0.4000 | 0.85 | 0.0000 | 0.0000 | 0.0000 | 0.6733 | 0.0101 |
| 1 | partial | 0.0769 | 0.50 | 0.0169 | 1.0000 | 0.0333 | 0.8533 | 0.0222 |
| 2 | head-only | 0.6667 | 0.10 | 0.0000 | 0.0000 | 0.0000 | 0.7567 | 0.0135 |
| 3 | partial | 0.0571 | 0.25 | 0.0122 | 1.0000 | 0.0241 | 0.9867 | 0.2000 |

四折宏平均：

- F1：`0.0144 ± 0.0170`
- alpha：`0.4250 ± 0.3279`
- AUROC：`0.8175 ± 0.1346`
- AUPRC：`0.0615 ± 0.0925`

四折汇总混淆矩阵：TN=1,042、FP=158、FN=2、TP=2；汇总 Precision=`0.0125`、
Recall=`0.5000`、F1=`0.0244`。

专家权重在四折中相同，因为当前融合只使用相同的正常 support 原型：

| 专家 | 权重 |
|---|---:|
| BGL | 0.3593 |
| HDFS | 0.2865 |
| Hadoop | 0.3542 |

权重并未坍缩到单一专家，但这种正常原型距离不能保证异常判别能力。

## 7. 探索性实验：每 VM 5 事件窗口

在看到确认性结果后，为检查平均池化是否稀释异常事件，增加了 VM 内 5 事件非重叠窗口。
父 VM 仍严格隔离。由于这一改动发生在确认性 test 已经查看之后，只能视为探索性消融。

结果为：

- 平均 F1：`0.0067 ± 0.0079`
- 平均 alpha：`0.7000 ± 0.0913`
- 平均 AUROC：`0.7085 ± 0.1001`
- 汇总 F1：`0.0129`

窗口化让 alpha 看起来更“正常”，但 F1 和 AUROC 都下降。这是本实验最直接的反例：
**alpha 达到 0.7 并不能证明专家模型有效，也不能保证 F1 提升。**

## 8. 为什么 F1 很低

1. OpenStack 只有 4 个异常 VM。每折只有 2 个异常 support、1 个异常 validation 和
   1 个异常 test，阈值几乎由单个 VM 决定。
2. 测试异常率约为 1/301。即使 AUROC 较高，只要产生几十个假阳性，Precision 和 F1
   就会迅速下降。
3. HDFS、BGL、Hadoop 的异常语义分别偏向存储 block、硬件告警和作业失败；OpenStack
   是云 VM 生命周期与 Nova 组件异常，跨系统差异很大。
4. 当前专家融合只比较目标正常原型与源正常原型的余弦距离，不使用异常 support 的判别质量。
5. GMM 使用目标正常 support 拟合，天然更贴近目标域；分类器需要从极少异常 VM 中学习可迁移模式。
6. 不同异常 VM 的模式差异明显，某个 validation VM 上的最佳 alpha/阈值不能稳定迁移到
   另一个 test VM。

## 9. 下一轮真正提高 F1 的方案

优先级从高到低：

1. 扩大 OpenStack 异常 VM 数量，或使用带更多独立异常会话的修订版 OpenStack 数据。
   在只有 4 个异常 group 时，不应宣称模型达到稳定最高 F1。
2. 将融合从“正常原型距离”升级为强收缩的异常感知 gating：在 support 上评估每个专家的
   grouped BCE/AUPRC，再与正常原型权重组合。
3. 在更多目标异常 group 上分别校准 classifier 与 GMM 分数，再学习非负 logistic stacking；
   不能在同一小 validation 上同时校准权重和阈值。
4. 对源专家增加 session-level 对比学习或困难负样本训练，使异常表示而不只是系统表示可迁移。
5. 报告 AUROC、AUPRC、FPR-at-fixed-recall 与 F1，避免在极低异常率下只追单一 F1。
6. 收集新异常 VM 后，用新的外层 test 重新确认任何探索性改进；不要继续在本次四个 test VM
   上追逐更高数字。

## 10. 复现路径

- 数据转换：`scripts/prepare_loghub_crossdomain.py`
- 目标四折编排：`scripts/run_loghub_target_grid.py`
- 结果汇总：`scripts/summarize_loghub_crossdomain.py`
- 源 checkpoint：`artifacts/checkpoints/loghub_crossdomain/source_seed42.pt`
- 确认性结果：`artifacts/outputs/loghub_crossdomain/nested_cv_results.json`
- 探索性结果：`artifacts/outputs/loghub_crossdomain_window5/nested_cv_results.json`
- 总审计结果：`artifacts/outputs/loghub_crossdomain/experiment_summary.json`

最终代码通过 41 项单元测试，并通过新增脚本的 `py_compile` 检查。
