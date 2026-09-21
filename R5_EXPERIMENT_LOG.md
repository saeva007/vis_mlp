# VisCast 连续概率建模实验账本

> 唯一的人类可读实验账本。状态快照：2026-09-21。新实验只追加到本文件与同目录 `R5_EXPERIMENT_REGISTRY.json`。本文只登记实际存在、实际提交或实际完成的实验；不改动任何真实目录名、checkpoint、config、结果或日志。

## 实验演化关系

```text
R0-p13-Classification
├─ R1-Extinction-Regression                         用连续 E 取代三分类监督
│  └─ R2-Exact30Gate-Regression                    把 30 km 点质量从连续回归中拆出
│     ├─ R3-Exact30Gate-Gaussian-LowVisAux         概率 E + <1 km auxiliary
│     └─ R4-Exact30Gate-Diffusion                  diffusion E，删除 auxiliary
│
├─ raw-E support diagnosis                         R3/R4 产生非正 E，修复后出现近零 E/极大 V
│  ├─ R2b-SupportSafe-Regression                   E_min + softplus(a)
│  ├─ R3b-SupportSafe-Gaussian                     Gaussian in log(E-E_min)
│  └─ R4b-SupportSafe-Diffusion                    DDPM in log(E-E_min)
│
└─ R5 Transformer backbone
   ├─ R5-Transformer-28.9Gate-Gaussian             新 backbone 的 Gaussian 基线
   ├─ R5-Transformer-28.9Gate-Diffusion-Original   原 interval-loss diffusion；S2 取消
   ├─ R5-Transformer-28.9Gate-Diffusion-SNRBoundary
   │  └─ Gate separability diagnostics A/B/C       证明 28.9 km 标签高度重叠
   ├─ R5-Cap30-Diffusion                           28.9 Gate → P(Y=30 km|X)
   └─ R5-CensoredFlow                              无 Gate/Cap/diffusion；30 km censored NLL
```

## 总表

| Display name | Backbone | High-vis treatment | Continuous model | Special loss | Status | 核心结论 |
|---|---|---|---|---|---|---|
| R0-p13-Classification | p13 GRU | 无特殊处理 | 3-class classification | p13 rare-event focal classification | finished | 已完成方案中正式 test 事件指标最强。 |
| R1-Extinction-Regression | p13 GRU | 无特殊处理 | deterministic raw E | Huber(E) | finished | argmax 完全不报低能见度。 |
| R2-Exact30Gate-Regression | p13 GRU | exact-30 Gate | deterministic raw E | Gate BCE + Huber(E) | finished | Gate 恢复召回，但中间类 precision 很低。 |
| R3-Exact30Gate-Gaussian-LowVisAux | p13 GRU | exact-30 Gate | Gaussian raw E | Gate BCE + NLL + <1 km BCE | finished | 高召回由大量 false alarms 换得，并暴露 support 问题。 |
| R4-Exact30Gate-Diffusion | p13 GRU | exact-30 Gate | DDPM raw E | Gate BCE + noise MSE | finished | 比 R3 平衡，但仍有非正 E 修复与极端误差。 |
| R2b-SupportSafe-Regression | p13 GRU | exact-30 Gate | bounded deterministic E | Gate BCE + Huber(E) | finished | support 稳定，事件概率质量不足。 |
| R3b-SupportSafe-Gaussian | p13 GRU | exact-30 Gate | Gaussian z | Gate BCE + Gaussian NLL | finished | b 系列最稳定、综合最好。 |
| R4b-SupportSafe-Diffusion | p13 GRU | exact-30 Gate | DDPM z | Gate BCE + noise MSE | finished | <500 recall 高，但中间类不稳且未超过 R3b。 |
| R5-Transformer-28.9Gate-Gaussian | R5 Transformer | 28.9 km Gate | Gaussian z | Gate BCE + NLL + interval | finished | 排序尚可但 argmax 极度保守。 |
| R5-Transformer-28.9Gate-Diffusion-Original | R5 Transformer | 28.9 km Gate | DDPM z | Gate BCE + noise MSE + interval | failed | S1 完成；S2 step 1744 后取消，无正式结果。 |
| R5-Transformer-28.9Gate-Diffusion-SNRBoundary | R5 Transformer | 28.9 km Gate | DDPM z | Gate BCE + SNR-weighted boundary | finished | 仍未恢复中间类 argmax。 |
| R5-Cap30-Diffusion | R5 Transformer | 30-km Cap probability | DDPM z | Cap BCE + SNR-weighted boundary | finished | 正式 test 完成；未恢复 500–1000 m argmax，整体未优于 28.9-Gate SNRBoundary。 |
| R5-CensoredFlow | R5 Transformer | 30-km censored likelihood | conditional 1D RQS flow | censored NLL only | failed | S1 step 7803 起 loss 全 NaN，step 8000 validation 失败；无 S2/正式 test。 |
| R5-Gate-Separability-A-LinearFrozen | frozen R5 encoder | 28.9 km diagnostic label | none | Gate BCE | finished | 线性 probe 仅中等可分。 |
| R5-Gate-Separability-B-MLPFrozen | frozen R5 encoder | 28.9 km diagnostic label | none | Gate BCE | finished | MLP 只带来有限增益。 |
| R5-Gate-Separability-C-MLPEndToEnd | end-to-end R5 Transformer | 28.9 km diagnostic label | none | Gate BCE | finished | encoder 微调仍只有小幅增益。 |

## Design decisions

1. **30 km 是观测上限且有明确点质量。** train/validation 的 exact-30 样本分别为 3,253,887/435,142；test 为 401,280。
2. **extinction 固定为** `E = 3.912 / V_km`。500 m、1000 m 对应 `E=7.824`、`E=3.912`。
3. **raw extinction 的概率生成存在 support 问题。** R3/R4 正 E 重采样拒绝计数为 49,754,444/25,679,113，softplus fallback 为 27,335/1,094,398；连续 RMSE 为 1,633.74/1,952.72 km。
4. **因此引入 support-safe 变换。** 对 `0<V<30` 定义 `E_min=3.912/30=0.1304`、`z=log(E-E_min)`；逆变换 `E=E_min+exp(z)`。R2b 用 `E_min+softplus(a)`。
5. **28.9 km 来自频率转折，而不是物理上限。** train 的 0.1-km count 经 5-bin 平滑在 28.9 km 首次由降转升；validation 的主转折更早（27.9 km），但在 train 固定的 28.9 km 后 0.5 km 净变化也为正。
6. **28.9 km Gate 在现有输入下高度重叠。** 最强 C probe 的 test AUROC/AP/Recall 为 0.87856/0.71130/0.55155。
7. **复杂 Gate head 收益有限。** B−A：AUROC +0.00852、AP +0.01739、Recall +0.02720；C−B：AUROC +0.00344、AP +0.00895、Recall +0.01285。
8. **去掉阈值邻域也没有质变。** C 排除 ±1 km 后 AP 仅由 0.71130 到 0.72259，Recall 由 0.55155 到 0.56698；不再通过堆 Gate head 追求完美二分类。
9. **R5-Cap 预测 `P(Y=30 km|X)`。** 它针对观测点质量并作为 soft mixture weight；不是 `P(V>=28.9 km|X)`，也不 hard route。
10. **R5-CensoredFlow 取消独立高能见度分类头。** `Y=30` 只贡献潜在 `V>=30`，即 `z<=log(3.912/30)` 的 censored likelihood；事件概率全部来自同一 CDF。
11. **数值实现备注。** R2b/R3b support 检查为 0 违规；R4b 因 `exp(z)` 浮点下溢出现大量 `E==E_min`/`V==30`（并非越过边界）。这与“理论严格大于/小于”的设计措辞不完全一致，保留原结果，不重算。
12. **exact-30 Cap 没有解决事件概率分配问题。** 相对 28.9-Gate SNRBoundary，Cap 的高能见度 Brier 略好，但三个事件 AP/CSI 均未提升，500–1000 m argmax Recall 仍为 0。
13. **第一版 censored RQS flow 数值不稳定。** S1 step 7802 尚为有限 loss，step 7803 起 uncensored/censored/total NLL 同时成为 NaN；因此本次运行只能作为失败记录，不能用于判断 censored-flow 科学假设。

## 正式模型实验

### R0-p13-Classification

**状态**  
finished。

**研究问题**  
正式 p13 三分类 ensemble 在统一 test set 上提供什么基线？

**基线**  
无；主线起点。

**输入**  
12 时次、27 dynamic vars、36 engineered features 及 p13 static/vegetation inputs；Tianji S2 split。

**Backbone**  
p13 GRU：hidden 256，1 layer，mean pooling；static 96、FE 128、fusion 256、vegetation embedding 16。

**输出分布/任务**  
3-class classification；正式 3-seed probability mean，argmax，无 threshold search。

**高能见度端处理**  
无特殊处理；`>=1000 m` 是第三类，不是 Gate/Cap/censoring。

**连续变量定义**  
无。

**Loss**  
正式 checkpoint/source 指向 p13 rare-event focal 三分类训练；隔离结果未保存精确 loss 权重，记为 unknown。

**相对上一实验唯一/主要改动**

- 主线起点。
- 当前账本只引用已有三 seed 概率均值，不重新训练或重算。

**训练方式**  
原 p13 训练详情未随隔离结果完整复制；EMA/步数 unknown。

**正式结果（including V=0）**

| Event | Recall | Precision | CSI | F1 | AP |
|---|---:|---:|---:|---:|---:|
| <500 m | 0.60875 | 0.25482 | 0.21896 | 0.35926 | 0.37245 |
| 500–1000 m | 0.30696 | 0.13229 | 0.10186 | 0.18489 | 0.11957 |
| <1000 m | 0.66015 | 0.27969 | 0.24449 | 0.39292 | 0.41669 |

**一句话结论**  
已完成方案中，R0 仍提供最强整体事件指标，且保留非零的 500–1000 m argmax 技能。

**真实路径**  
config: `/public/home/putianshu/vis_mlp/visibility_continuous_20260912/configs/experiment.json`；checkpoint: 三 seed 清单未复制，已知 initializer 为 `/public/home/putianshu/vis_mlp/checkpoints/exp_20260718_232510_p13_sampling_calibration_manual_retry_p13_seed42_2_proposed_rare_event_focal_S2_PhaseD_best_score.pt`；result: `/public/home/putianshu/vis_mlp/visibility_continuous_20260912/eval/R0/metrics.json`；source probabilities: `/public/home/putianshu/vis_mlp/static_rnn_precision_candidate_eval/p13_seed_mean_timefix_20260719_130856_mean_argmax/probs.npy`；log/slurm: `/public/home/putianshu/vis_mlp/visibility_continuous_20260912/logs/job_chain.tsv`。

**Slurm job ID**  
原三 seed训练 ID unknown；隔离检查为 `121894987`。

### R1-Extinction-Regression

**状态** finished。  
**研究问题** 连续 extinction 监督能否取代三分类并提高低能见度敏感性？  
**基线** R0-p13-Classification。  
**输入** same as R0-p13-Classification。  
**Backbone** p13 GRU，p13 checkpoint 初始化后端到端微调。  
**输出分布/任务** deterministic extinction regression。  
**高能见度端处理** 无特殊处理；30 km 不进入 continuous loss，也没有 Gate。  
**连续变量定义** raw `E=3.912/V`；head 经 `softplus+1e-5` 保证正值。  
**Loss** continuous 样本上的 standardized-E Huber。  
**相对上一实验唯一/主要改动**

- 删除三分类 head/loss，改为一个 deterministic E head。
- V=0 与 V=30 不进入 continuous loss；其余输入、backbone、test split 不变。

**训练方式** 单阶段；best step 500；实际到 step 5500 后 early stop；无 EMA、无异步 validation。  
**正式结果**

| Event | Recall | Precision | CSI | F1 | AP |
|---|---:|---:|---:|---:|---:|
| <500 m | 0 | 0 | 0 | 0 | 0.01146 |
| 500–1000 m | 0 | 0 | 0 | 0 | 0.00825 |
| <1000 m | 0 | 0 | 0 | 0 | 0.01972 |

MAE/RMSE `12.8933/15.8559 km`；MAE `V<1/V<0.5 km = 1.6077/1.8149 km`。

**一句话结论** 连续监督降低了低端误差，但没有形成可用事件概率，argmax 完全不报低能见度。  
**真实路径** config `/public/home/putianshu/vis_mlp/visibility_continuous_20260912/configs/experiment.json`；checkpoint `/public/home/putianshu/vis_mlp/visibility_continuous_20260912/models/R1/best.pt`；result `/public/home/putianshu/vis_mlp/visibility_continuous_20260912/eval/R1/metrics.json`；history `/public/home/putianshu/vis_mlp/visibility_continuous_20260912/models/R1/history.json`；log `/public/home/putianshu/vis_mlp/visibility_continuous_20260912/logs/121897136_vc_R1.out`；slurm `/public/home/putianshu/vis_mlp/visibility_continuous_20260912/scripts/sub_route.slurm`。  
**Slurm job ID** `121897136`。

### R2-Exact30Gate-Regression

**状态** finished。  
**研究问题** 显式拆出 exact-30 point mass 是否改善 deterministic regression？  
**基线** R1-Extinction-Regression。  
**输入** same as R1。  
**Backbone** p13 GRU。  
**输出分布/任务** exact-30 Gate + deterministic extinction。  
**高能见度端处理** `P(Y=30 km|X)` exact-30 Gate。  
**连续变量定义** raw `E=3.912/V`，positive softplus head。  
**Loss** Gate BCE + standardized-E Huber。  
**相对上一实验唯一/主要改动**

- 新增 exact-30 Gate BCE 与 soft probability mixture。
- continuous head、输入、backbone、split 不变。

**训练方式** 单阶段 15000 step；best 14500；无 EMA/异步 validation。  
**正式结果**

| Event | Recall | Precision | CSI | F1 | AP |
|---|---:|---:|---:|---:|---:|
| <500 m | 0.49784 | 0.23906 | 0.19261 | 0.32301 | 0.21121 |
| 500–1000 m | 0.55476 | 0.04909 | 0.04723 | 0.09021 | 0.06074 |
| <1000 m | 0.81775 | 0.13764 | 0.13355 | 0.23563 | 0.25358 |

MAE/RMSE `13.1751/21.0792 km`；MAE `V<1/V<0.5 = 1.4702/1.4556 km`。

**一句话结论** exact-30 Gate 恢复事件召回，但中间类 precision 仅 0.049，且整体连续误差未优于 R1。  
**真实路径** config `/public/home/putianshu/vis_mlp/visibility_continuous_20260912/configs/experiment.json`；checkpoint `/public/home/putianshu/vis_mlp/visibility_continuous_20260912/models/R2/best.pt`；result `/public/home/putianshu/vis_mlp/visibility_continuous_20260912/eval/R2/metrics.json`；history `/public/home/putianshu/vis_mlp/visibility_continuous_20260912/models/R2/history.json`；log `/public/home/putianshu/vis_mlp/visibility_continuous_20260912/logs/121897141_vc_R2.out`；slurm `/public/home/putianshu/vis_mlp/visibility_continuous_20260912/scripts/sub_route.slurm`。  
**Slurm job ID** `121897141`。

### R3-Exact30Gate-Gaussian-LowVisAux

**状态** finished。  
**研究问题** heteroscedastic Gaussian 与 `<1 km` auxiliary 是否改善概率排序？  
**基线** R2-Exact30Gate-Regression。  
**输入** same as R2。  
**Backbone** p13 GRU。  
**输出分布/任务** exact-30 Gate + Gaussian raw extinction + low-vis auxiliary。  
**高能见度端处理** exact-30 Gate。  
**连续变量定义** standardized raw `E=3.912/V`。  
**Loss** Gate BCE + Gaussian NLL + BCE(`V<1 km`)，权重均为 1。  
**相对上一实验唯一/主要改动**

- deterministic head 改为 `(mu_E,log_sigma_E)`。
- 新增 `<1 km` auxiliary BCE；Gate、backbone、数据不变。

**训练方式** 单阶段 15000 step；best 15000；64 draws test；无 EMA/异步 validation。  
**正式结果**

| Event | Recall | Precision | CSI | F1 | AP |
|---|---:|---:|---:|---:|---:|
| <500 m | 0.90994 | 0.07903 | 0.07842 | 0.14543 | 0.23353 |
| 500–1000 m | 0.07456 | 0.02624 | 0.01979 | 0.03882 | 0.01853 |
| <1000 m | 0.87379 | 0.11084 | 0.10909 | 0.19673 | 0.23653 |

MAE/RMSE `23.9597/1633.7416 km`；MAE `V<1/V<0.5 = 6.2628/6.2188 km`；NLL `0.86184`；CRPS(E) `1.81072`。

**一句话结论** auxiliary 推高 recall 但 precision/CSI 崩塌；raw Gaussian 的非正 E 与近零修复造成极端连续误差。  
**真实路径** config `/public/home/putianshu/vis_mlp/visibility_continuous_20260912/configs/experiment.json`；checkpoint `/public/home/putianshu/vis_mlp/visibility_continuous_20260912/models/R3/best.pt`；result `/public/home/putianshu/vis_mlp/visibility_continuous_20260912/eval/R3/metrics.json`；history `/public/home/putianshu/vis_mlp/visibility_continuous_20260912/models/R3/history.json`；log `/public/home/putianshu/vis_mlp/visibility_continuous_20260912/logs/121897150_vc_R3.out`；slurm `/public/home/putianshu/vis_mlp/visibility_continuous_20260912/scripts/sub_route.slurm`。  
**Slurm job ID** `121897150`。

### R4-Exact30Gate-Diffusion

**状态** finished。  
**研究问题** conditional diffusion 是否比 Gaussian+auxiliary 更好地拟合 extinction 分布？  
**基线** R3-Exact30Gate-Gaussian-LowVisAux。  
**输入** same as R3。  
**Backbone** p13 GRU。  
**输出分布/任务** exact-30 Gate + 100-step conditional DDPM on standardized raw E。  
**高能见度端处理** exact-30 Gate。  
**连续变量定义** raw `E=3.912/V`。  
**Loss** Gate BCE + diffusion epsilon-prediction MSE；删除 low-vis auxiliary。  
**相对上一实验唯一/主要改动**

- Gaussian distribution 改为 conditional DDPM。
- 删除 `<1 km` auxiliary；Gate、encoder、数据不变。

**训练方式** 单阶段 15000 step；best 15000；64 draws test；无 EMA。  
**正式结果**

| Event | Recall | Precision | CSI | F1 | AP |
|---|---:|---:|---:|---:|---:|
| <500 m | 0.71623 | 0.17007 | 0.15933 | 0.27487 | 0.29617 |
| 500–1000 m | 0.12729 | 0.12575 | 0.06753 | 0.12652 | 0.07084 |
| <1000 m | 0.67085 | 0.23357 | 0.20956 | 0.34650 | 0.34157 |

MAE/RMSE `29.7918/1952.7219 km`；MAE `V<1/V<0.5 = 4.0444/3.3056 km`；CRPS(E) `9.58063`。

**一句话结论** diffusion 的事件平衡优于 R3，但 raw-E support repair 仍造成严重连续失真，不能作为干净 diffusion 证据。  
**真实路径** config `/public/home/putianshu/vis_mlp/visibility_continuous_20260912/configs/experiment.json`；checkpoint `/public/home/putianshu/vis_mlp/visibility_continuous_20260912/models/R4/best.pt`；result `/public/home/putianshu/vis_mlp/visibility_continuous_20260912/eval/R4/metrics.json`；history `/public/home/putianshu/vis_mlp/visibility_continuous_20260912/models/R4/history.json`；log `/public/home/putianshu/vis_mlp/visibility_continuous_20260912/logs/121897153_vc_R4.out`；slurm `/public/home/putianshu/vis_mlp/visibility_continuous_20260912/scripts/sub_route.slurm`。  
**Slurm job ID** `121897153`。

### R2b-SupportSafe-Regression

**状态** finished（3 seeds）。  
**研究问题** bounded deterministic output 能否修复 R2 的连续支持域？  
**基线** R2-Exact30Gate-Regression。  
**输入** same as R2。  
**Backbone** p13 GRU，共同正式 p13 checkpoint 初始化。  
**输出分布/任务** exact-30 Gate + deterministic bounded E。  
**高能见度端处理** exact-30 Gate。  
**连续变量定义** `E_hat=E_min+softplus(a)`，`E_min=0.1304`。  
**Loss** Gate BCE + Huber(`E_hat,E_true`)；无 auxiliary/weighting。  
**相对上一实验唯一/主要改动**

- deterministic E 从仅 `>0` 改为受 30-km 下界约束。
- 改为 Stage A/B 与 event-AP checkpoint selection。
- 模型、Gate、数据定义保持不变。

**训练方式** Stage A 冻结 encoder 4000 step，Stage B 解冻；EMA 否；同步 validation 每500；best steps seed1/2/3=`8000/10500/8500`，early-stop steps=`18000/20500/18500`；1 DCU、effective batch 2048、workers16。  
**正式结果（3-seed mean±population SD）**

| Event | Recall | Precision | CSI | F1 | AP |
|---|---:|---:|---:|---:|---:|
| <500 m | 0.54638±0.00571 | 0.26091±0.00106 | 0.21446±0.00133 | 0.35317±0.00180 | 0.24853±0.00395 |
| 500–1000 m | 0.52696±0.00645 | 0.05685±0.00087 | 0.05409±0.00073 | 0.10263±0.00131 | 0.05251±0.00012 |
| <1000 m | 0.81856±0.00290 | 0.16057±0.00244 | 0.15505±0.00217 | 0.26847±0.00326 | 0.27109±0.00799 |

MAE/RMSE `7.31970±0.03252 / 9.92949±0.03993 km`；MAE `V<1/V<0.5 = 1.48839±0.00800 / 1.35810±0.00822 km`。三 seed support violations 均为 0。每 seed 原始 event metrics 与 resource config 均保留在下述三个 `metrics.json`。

**一句话结论** support 与连续误差明显变稳，但中间类 precision/AP 仍低，未达到 R0。  
**真实路径** config `/public/home/putianshu/vis_mlp/visibility_continuous_b_20260913/configs/experiment.json`；checkpoints `/public/home/putianshu/vis_mlp/visibility_continuous_b_20260913/runs/R2b/seed1/checkpoints/best.pt`、`seed2/checkpoints/best.pt`、`seed3/checkpoints/best.pt`；results `/public/home/putianshu/vis_mlp/visibility_continuous_b_20260913/runs/R2b/seed1/eval/metrics.json`、`seed2/eval/metrics.json`、`seed3/eval/metrics.json`；logs `/public/home/putianshu/vis_mlp/visibility_continuous_b_20260913/logs/vcb_R2b_s1_121986817.out`、`vcb_R2b_s2_121986820.out`、`vcb_R2b_s3_121986825.out`；slurm `/public/home/putianshu/vis_mlp/visibility_continuous_b_20260913/scripts/sub_train.slurm`。  
**Slurm job ID** `121986817, 121986820, 121986825`。

### R3b-SupportSafe-Gaussian

**状态** finished（3 seeds）。  
**研究问题** 干净的 z-space Gaussian 是否在 support-safe 条件下提供稳定概率技能？  
**基线** R2b-SupportSafe-Regression。  
**输入** same as R2b。  
**Backbone** p13 GRU。  
**输出分布/任务** exact-30 Gate + heteroscedastic Gaussian in standardized `z`。  
**高能见度端处理** exact-30 Gate。  
**连续变量定义** `z=log(E-E_min)`；`E=E_min+exp(z)`。  
**Loss** Gate BCE + Gaussian NLL(z)；无 low-vis auxiliary。  
**相对上一实验唯一/主要改动**

- deterministic head 改为 `(mu_z,log_sigma_z)`。
- 保留相同 Gate、support transform、backbone、数据与训练 protocol。
- 不含 R3 的 low-vis auxiliary。

**训练方式** Stage A 4000 + Stage B；EMA 否；best steps `41000/30500/22500`，stop `51000/40500/32500`；256 draws 用于 continuous/CRPS，事件概率由 Gaussian CDF 解析得到。  
**正式结果（3-seed mean±population SD）**

| Event | Recall | Precision | CSI | F1 | AP |
|---|---:|---:|---:|---:|---:|
| <500 m | 0.59773±0.00504 | 0.26670±0.00171 | 0.22610±0.00064 | 0.36882±0.00085 | 0.35794±0.00255 |
| 500–1000 m | 0.10015±0.00310 | 0.20763±0.00287 | 0.07245±0.00177 | 0.13510±0.00308 | 0.11768±0.00075 |
| <1000 m | 0.52198±0.00512 | 0.34686±0.00266 | 0.26322±0.00098 | 0.41674±0.00122 | 0.39828±0.00138 |

MAE/RMSE `5.52937±0.02191 / 7.27849±0.01949 km`；MAE `V<1/V<0.5 = 2.33514±0.01630 / 2.25809±0.01900 km`；NLL(z) `1.72072±0.00168`；CRPS(V) `3.64380±0.00651 km`；support violations 0。

**一句话结论** R3b 是 b 系列最稳定方案，AP/CSI 与连续分布指标均优于 R2b/R4b 的主要权衡。  
**真实路径** config `/public/home/putianshu/vis_mlp/visibility_continuous_b_20260913/configs/experiment.json`；checkpoints `/public/home/putianshu/vis_mlp/visibility_continuous_b_20260913/runs/R3b/seed{1,2,3}/checkpoints/best.pt`；results `/public/home/putianshu/vis_mlp/visibility_continuous_b_20260913/runs/R3b/seed{1,2,3}/eval/metrics.json`；logs `/public/home/putianshu/vis_mlp/visibility_continuous_b_20260913/logs/vcb_R3b_s{1,2,3}_<job>.out`；slurm `/public/home/putianshu/vis_mlp/visibility_continuous_b_20260913/scripts/sub_train.slurm`。  
**Slurm job ID** `121986826, 121986828, 121986830`。

### R4b-SupportSafe-Diffusion

**状态** finished（4 seeds）。  
**研究问题** 在与 R3b 相同 support-safe z 空间内，diffusion 是否有独立稳定收益？  
**基线** R3b-SupportSafe-Gaussian。  
**输入** same as R3b。  
**Backbone** p13 GRU。  
**输出分布/任务** exact-30 Gate + 100-step conditional DDPM in standardized z。  
**高能见度端处理** exact-30 Gate。  
**连续变量定义** `z=log(E-E_min)`。  
**Loss** Gate BCE + diffusion epsilon MSE。  
**相对上一实验唯一/主要改动**

- Gaussian z distribution 改为 DDPM z。
- 新增 EMA，validation/test 用 EMA。
- 其余 Gate、backbone、数据和 support transform 不变。

**训练方式** Stage A 4000 + Stage B；EMA 0.999；best steps `22000/12500/5500/11500`，stop `32000/22500/15500/21500`；256 draws。  
**正式结果（4-seed mean±population SD）**

| Event | Recall | Precision | CSI | F1 | AP |
|---|---:|---:|---:|---:|---:|
| <500 m | 0.71745±0.01511 | 0.20076±0.00977 | 0.18592±0.00749 | 0.31347±0.01070 | 0.35669±0.00315 |
| 500–1000 m | 0.06202±0.03487 | 0.15495±0.01120 | 0.04380±0.02079 | 0.08316±0.03859 | 0.09378±0.00706 |
| <1000 m | 0.62616±0.02750 | 0.28021±0.01668 | 0.23943±0.00855 | 0.38628±0.01119 | 0.39356±0.00366 |

MAE/RMSE `5.67523±0.20750 / 7.36487±0.17533 km`；MAE `V<1/V<0.5 = 2.64712±0.30466 / 2.57243±0.32186 km`；CRPS(V) `3.73323±0.17083 km`。support 检查出现 `E==E_min/V==30` 浮点边界计数，详见 Design decisions 11。

**一句话结论** diffusion 提高极端低能见度 recall，但 precision/CSI、中间类、连续误差和 seed 稳定性均未超过 R3b。  
**真实路径** config `/public/home/putianshu/vis_mlp/visibility_continuous_b_20260913/configs/experiment.json`；checkpoints `/public/home/putianshu/vis_mlp/visibility_continuous_b_20260913/runs/R4b/seed{1,2,3,4}/checkpoints/best.pt`；results `/public/home/putianshu/vis_mlp/visibility_continuous_b_20260913/runs/R4b/seed{1,2,3,4}/eval/metrics.json`；logs `/public/home/putianshu/vis_mlp/visibility_continuous_b_20260913/logs/vcb_R4b_s{1,2,3,4}_<job>.out`；slurm `/public/home/putianshu/vis_mlp/visibility_continuous_b_20260913/scripts/sub_train.slurm`。  
**Slurm job ID** `121986832, 121986834, 121986836, 121986837`。

### R5-Transformer-28.9Gate-Gaussian

**状态** finished。  
**研究问题** 脱离 p13 GRU 后，新 Transformer + Gaussian z 是否改善连续概率建模？  
**基线** R3b-SupportSafe-Gaussian（科学基线）；实现位于新的 R5 工程。  
**输入** 12 时次 point weather/PM（27 vars）+ handcrafted static + aerosol context。  
**Backbone** 6-layer pre-LN Transformer，d256、8 heads、FFN1024。  
**输出分布/任务** 28.9 Gate + heteroscedastic Gaussian z。  
**高能见度端处理** `P(V>=28.9 km|X)` Gate。  
**连续变量定义** `z=log(3.912/V-3.912/28.9)`，eligible `0<V<28.9`。  
**Loss** Gate BCE + Gaussian NLL(z) + 0.1×interval BCE(0.5/1 km)。  
**相对上一实验唯一/主要改动**

- p13 GRU 换为 shared Transformer backbone 与 task queries。
- Gate 阈值从 exact 30 改为频率转折 28.9 km。
- 加入 interval supervision；输入仍限于现有 point/static/PM/engineered 范围。

**训练方式** ERA5 S1（实际37500，best35000，async-patience stop）→ Tianji S2 60000；EMA；quick + async full validation；1 DCU train/4 DCU validator。  
**正式结果**

| Event | Recall | Precision | CSI | F1 | AP |
|---|---:|---:|---:|---:|---:|
| <500 m | 0.02652 | 0.79198 | 0.02634 | 0.05132 | 0.27132 |
| 500–1000 m | 0 | 0 | 0 | 0 | 0.10940 |
| <1000 m | 0.01736 | 0.89153 | 0.01732 | 0.03405 | 0.32045 |

Gate accuracy/precision/recall/AUROC/AP/Brier=`0.82627/0.69491/0.52429/0.86964/0.69046/0.12023`；MAE/RMSE=`5.31989/6.81203 km`；MAE `V<1/V<0.5=6.97082/7.70535 km`；NLL(z)=`1.54334`；CRPS(V)=`3.15394 km`。

**一句话结论** Gaussian AP 尚可但 argmax 极度保守，中间类完全不报，低能见度连续误差很大。  
**真实路径** config `/public/home/putianshu/vis_mlp/visibility_r5_transformer_20260913/configs/r5.json`；checkpoint `/public/home/putianshu/vis_mlp/visibility_r5_transformer_20260913/runs/R5-Gaussian/seed1/S2/checkpoints/ckpt_step_60000.pt`；result `/public/home/putianshu/vis_mlp/visibility_r5_transformer_20260913/runs/R5-Gaussian/seed1/S2/test_metrics.json`；log root `/public/home/putianshu/vis_mlp/visibility_r5_transformer_20260913/logs`；slurm `/public/home/putianshu/vis_mlp/visibility_r5_transformer_20260913/scripts/sub_train.slurm`。  
**Slurm job ID** successful S1/S2/final=`122008067/122011679/122020233`；早期取消/冲突作业另见“记录问题”。

### R5-Transformer-28.9Gate-Diffusion-Original

**状态** failed（S1 finished，S2 canceled）。  
**研究问题** 新 Transformer 下原始 interval-supervised diffusion 是否优于 Gaussian？  
**基线** R5-Transformer-28.9Gate-Gaussian。  
**输入** same as R5 Transformer Gaussian。  
**Backbone** same R5 Transformer。  
**输出分布/任务** 28.9 Gate + 100-step cosine DDPM，v-prediction。  
**高能见度端处理** 28.9 km Gate。  
**连续变量定义** same 28.9 support-safe z。  
**Loss** Gate BCE + diffusion v-prediction loss + 0.1×interval BCE。  
**相对上一实验唯一/主要改动**

- Gaussian head/NLL 改为 4-block conditional diffusion/v-prediction。
- 其余 backbone、Gate、interval、数据不变。

**训练方式** S1 完成40000 step，forward objective 选择 EMA step20000；S2 取消于 step1744，无 checkpoint、无正式 evaluation。  
**正式结果** 无；不得用 S1 或 step1000 quick objective 代替正式结果。  
**一句话结论** 实验未完成，不能对原始 diffusion 科学性能下结论；后续 SNRBoundary 是独立正式版本。  
**真实路径** config `/public/home/putianshu/vis_mlp/visibility_r5_transformer_20260913/configs/r5.json`；S1 checkpoint `/public/home/putianshu/vis_mlp/visibility_r5_transformer_20260913/runs/R5-Diffusion/seed1/S1/checkpoints/ckpt_step_20000.pt`；S2 partial history `/public/home/putianshu/vis_mlp/visibility_r5_transformer_20260913/runs/R5-Diffusion/seed1/S2/train_history.jsonl`；logs `/public/home/putianshu/vis_mlp/visibility_r5_transformer_20260913/logs/r5_Diffusion_S2_s1_122061322.out`；slurm `/public/home/putianshu/vis_mlp/visibility_r5_transformer_20260913/scripts/sub_train.slurm`。  
**Slurm job ID** successful S1=`122008267`；S2 attempts=`122061315,122061316,122061322`（均 canceled；最后一个到 step1744）。

### R5-Transformer-28.9Gate-Diffusion-SNRBoundary

**状态** finished。  
**研究问题** SNR-weighted boundary supervision 能否让 R5 diffusion 同时恢复事件概率与连续分布？  
**基线** R5-Transformer-28.9Gate-Diffusion-Original。  
**输入** same as R5 Transformer。  
**Backbone** R5 Transformer + GateQuery + ContinuousQuery + 4-block denoiser。  
**输出分布/任务** 28.9 Gate + 100-step cosine DDPM，v-prediction，EMA。  
**高能见度端处理** 28.9 km Gate。  
**连续变量定义** `z=log(3.912/V-3.912/28.9)`。  
**Loss** Gate BCE + diffusion loss + balanced boundary BCE(0.5/1 km) weighted by `alpha_bar=SNR/(1+SNR)`；lambda boundary=1。  
**相对上一实验唯一/主要改动**

- 原 interval loss 改为正负均衡、按 diffusion SNR 加权的 boundary loss。
- boundary 权重从 0.1 改为 1；denoiser/backbone/Gate/data 不变。
- S1 handoff 不等待 sampling validation；S2 full validation 异步。

**训练方式** ERA5 S1 40000（EMA best35000）→ Tianji S2 100000；screening best85000；quick every1000、checkpoint2500、async full5000；final test 256 draws/100 reverse steps。  
**正式结果**

| Event | Recall | Precision | CSI | F1 | AP |
|---|---:|---:|---:|---:|---:|
| <500 m | 0.55964 | 0.20734 | 0.17826 | 0.30258 | 0.28792 |
| 500–1000 m | 0 | 0 | 0 | 0 | 0.08274 |
| <1000 m | 0.48166 | 0.30695 | 0.23073 | 0.37495 | 0.33532 |

Gate accuracy/precision/recall/AUROC/AP/Brier=`0.82546/0.69769/0.51315/0.86812/0.68860/0.12082`；MAE/RMSE=`5.24403/6.80128 km`；MAE `V<1/V<0.5=4.98061/5.33188 km`；CRPS(V)=`3.14404 km`；500–1000 m 真值样本 mean probability=`0.36914/0.05198/0.57887`，argmax=`37.34%/0%/62.66%`。

**一句话结论** SNRBoundary 完成了稳定正式评估，但仍没有恢复 500–1000 m argmax 技能。  
**真实路径** config `/public/home/putianshu/vis_mlp/visibility_r5_diffusion_snr_20260914/configs/r5.json`；checkpoint `/public/home/putianshu/vis_mlp/visibility_r5_diffusion_snr_20260914/runs/R5-Diffusion/seed1/S2/checkpoints/ckpt_step_85000.pt`；result `/public/home/putianshu/vis_mlp/visibility_r5_diffusion_snr_20260914/runs/R5-Diffusion/seed1/S2/test_metrics.json`；predictions `/public/home/putianshu/vis_mlp/visibility_r5_diffusion_snr_20260914/runs/R5-Diffusion/seed1/S2/test_metrics_predictions.npz`；logs `/public/home/putianshu/vis_mlp/visibility_r5_diffusion_snr_20260914/logs`；slurm `/public/home/putianshu/vis_mlp/visibility_r5_diffusion_snr_20260914/scripts/sub_train.slurm`。  
**Slurm job ID** S1/S2/final=`122062073/122067507/122086433`。

### R5-Cap30-Diffusion

**状态** finished。  
**研究问题** 用 exact-30 point-mass probability 取代不可清晰分开的 28.9 Gate，能否改善事件概率？  
**基线** R5-Transformer-28.9Gate-Diffusion-SNRBoundary。  
**输入** same as R5 SNRBoundary。  
**Backbone** same R5 Transformer；GateQuery/Head 改为 CapQuery/Head；diffusion 不变。  
**输出分布/任务** `P(Y=30 km|X)` + conditional diffusion。  
**高能见度端处理** 30-km Cap probability；soft mixture，不 hard route。  
**连续变量定义** `z=log(3.912/V-3.912/30)`，eligible `0<V<30`。  
**Loss** unweighted Cap BCE + diffusion v-prediction loss + unchanged SNR-weighted boundary loss。  
**相对上一实验唯一/主要改动**

- 删除 `P(V>=28.9)` Gate，新增 `P(Y=30)` Cap。
- continuous support 上限从 28.9 扩展到 30 km。
- backbone、denoiser、boundary loss、输入和 split 不变。

**训练方式** S1 40000，EMA handoff best35000；S2 90000 后 `async_full_validation_patience` 停止，best step=85000；训练实际使用 1 DCU（申请4）、batch=1024、workers=16；quick+async full。最终 test 使用 4 DCU、256 draws、100 reverse steps、1,753,078 samples，inference wall=`60,023 s`（约16.67 h）。  
**正式结果** 统一 test、包含 V=0：`<500 R/P/CSI/F1/AP=0.57635/0.19126/0.16769/0.28721/0.27727`；`500–1000=0/0/0/0/0.07514`；`<1000=0.50634/0.28901/0.22548/0.36798/0.32781`。排除 V=0：`<500=0.52899/0.14316/0.12697/0.22533/0.18683`；`500–1000=0/0/0/0/0.07586`；`<1000=0.47104/0.24672/0.19320/0.32383/0.26382`。Cap accuracy/precision/recall/AUROC/AP/Brier=`0.83454/0.69825/0.48810/0.87102/0.67568/0.11530`。连续 MAE/RMSE=`5.18297/6.74911 km`，`V<1 km` MAE=`4.84893 km`，`V<0.5 km` MAE=`5.20421 km`，CRPS(V)=`3.23821 km`。448,787,968 个生成 draws 中 `E<=E_min` 与连续 `V>=30` 均为 0。  
**500–1000 m 概率诊断** 对真实中间类，mean `P(<500)/P(500–1000)/P(>=1000)=0.39385/0.04971/0.55644`；argmax 落入三类比例=`0.40909/0/0.59091`。模型几乎把中间类概率质量拆给两侧，因此中间类 Recall 精确为 0。  
**对比分析** 相对 R5 SNRBoundary，Cap 的 `<500` Recall `+0.01672`，但 Precision/CSI/AP 分别 `-0.01608/-0.01057/-0.01065`；`<1000` Recall `+0.02468`，但 Precision/CSI/AP `-0.01794/-0.00526/-0.00750`；中间类仍为零 Recall 且 AP `-0.00760`。连续 MAE/RMSE 和两个低能见度 MAE仅改善 `0.061/0.052/0.132/0.128 km`，CRPS 反而差 `0.094 km`。Cap Brier 比 Gate 好 `0.00552`，但 AP 低 `0.01291`。相对 R0，三个事件 AP 分别低 `0.09517/0.04444/0.08888`，没有替代 p13 baseline。  
**一句话结论** exact-30 Cap 对连续点估计有很小改善，但没有改善整体概率质量或恢复 500–1000 m 分类；不足以支持用 Cap 取代当前基线。  
**真实路径** config `/public/home/putianshu/vis_mlp/visibility_r5_cap_20260915/configs/r5_cap.json`；checkpoint `/public/home/putianshu/vis_mlp/visibility_r5_cap_20260915/runs/R5-Cap/seed1/S2/checkpoints/ckpt_step_85000.pt`；result `/public/home/putianshu/vis_mlp/visibility_r5_cap_20260915/runs/R5-Cap/seed1/S2/test_metrics.json`；predictions `/public/home/putianshu/vis_mlp/visibility_r5_cap_20260915/runs/R5-Cap/seed1/S2/test_metrics_predictions.npz`；log `/public/home/putianshu/vis_mlp/visibility_r5_cap_20260915/logs/r5cap_final_s1_122208298.out`；slurm `/public/home/putianshu/vis_mlp/visibility_r5_cap_20260915/scripts/sub_final_eval.slurm`。  
**Slurm job ID** S1/S2/final=`122175875/122183044/122208298`；final `COMPLETED`，exit `0:0`，ended `2026-09-17 02:30:40`。

### R5-CensoredFlow

**状态** failed（S1 数值发散；未进入 S2）。  
**研究问题** 一个统一 censored conditional spline flow 能否不用 Gate/Cap/diffusion/分类 head 同时拟合完整分布并恢复中间类概率？  
**基线** R5-Cap30-Diffusion（研究动机）；shared backbone 与 R5 相同。  
**输入** same as R5。  
**Backbone** R5 Transformer → FlowQuery → small conditioner。  
**输出分布/任务** 单个 16-bin conditional monotonic rational-quadratic spline，linear tails，精确 log-density/CDF。  
**高能见度端处理** 30-km censored likelihood；无独立 Gate/Cap head。  
**连续变量定义** `z=log(3.912/V)`；`z30=log(3.912/30)=-2.03714863`。  
**Loss** censored NLL only：`0<V<30` 用 `-log p(z|X)`；`V=30` 用稳定 `-log F(z30|X)`；V=0 不进 likelihood。  
**相对上一实验唯一/主要改动**

- 删除 Cap head、diffusion denoiser、boundary/interval loss与所有 classification head。
- support-safe offset z 改为直接 `log(3.912/V)`，30 km 通过 censoring 处理。
- 事件概率由同一 flow CDF 解析积分；无 sampling。
- shared R5 backbone、输入、S1/S2 split 不变。

**训练方式** 4-DCU DDP，batch1024/DCU、effective4096、workers16；原计划 S1 40000→S2 100000；quick analytic validation every1000、checkpoint2500、async full5000；无 EMA。实际只运行至 S1 step8000（0.745 epoch，32,768,000 processed samples），最后 checkpoint step7500。  
**正式结果** 无。S1 step7000 的最后一次有限 quick subset（20k，仅监控）为：censored NLL=`1.86865`；AP `<500/middle/<1000=0.57240/0.47992/0.94361`；Recall=`0.0268/0.0012/0.0215`；CSI=`0.02648/0.00120/0.02150`；连续 MAE/RMSE=`4.42164/6.95421 km`；PIT mean/std=`0.79089/0.25645`。真实 500–1000 m 样本 mean probabilities=`0.06965/0.07534/0.85500`，argmax fractions=`0.0122/0.0012/0.9866`。这些来自人为分层 quick subset，不能与正式 test 指标横比。  
**失败诊断** step7802 的 total/uncensored/censored-30 NLL 仍为 `0.72440/0.73274/0.69669`；step7803 三者同时变为 NaN，并持续到 step8000。随后 quick validation 在 `precision_recall_curve` 检测到 NaN score 后退出。作业耗时 `21:13:44`，exit=`1:0`；没有重提、S2、full validation 或 test。  
**分析** 在数值失稳前，解析 CDF 的排序 AP 看似较高，但 argmax 仍几乎全部落到 `>=1000 m`，并未显示恢复中间类决策的证据。由于训练未完成、subset 非正式且 checkpoint 未做正式评估，本次结果不能否定或支持 censored-flow 假设；能确定的只有当前 RQS/censored-NLL 实现或训练配置不稳定。  
**一句话结论** 第一版 R5-CensoredFlow 因 S1 NaN 发散而失败，没有可用于模型排名的正式结果。  
**真实路径** config `/public/home/putianshu/vis_mlp/visibility_r5_censored_flow_20260915/configs/r5_censored_flow.json`；latest checkpoint `/public/home/putianshu/vis_mlp/visibility_r5_censored_flow_20260915/runs/R5-CensoredFlow/seed1/S1/checkpoints/ckpt_step_07500.pt`；quick result `/public/home/putianshu/vis_mlp/visibility_r5_censored_flow_20260915/runs/R5-CensoredFlow/seed1/S1/quick_validation_history.jsonl`；train history `/public/home/putianshu/vis_mlp/visibility_r5_censored_flow_20260915/runs/R5-CensoredFlow/seed1/S1/train_history.jsonl`；error log `/public/home/putianshu/vis_mlp/visibility_r5_censored_flow_20260915/logs/r5flow_S1_s1_122197992.err`；slurm `/public/home/putianshu/vis_mlp/visibility_r5_censored_flow_20260915/scripts/sub_train.slurm`。  
**Slurm job ID** sanity=`122197410` COMPLETED；S1=`122197992` FAILED，exit `1:0`，ended `2026-09-16 21:17:08`；S2/test 未提交。

## 诊断 / ablation 实验

### R5-Gate-Separability-A-LinearFrozen

**状态** finished。  
**研究问题** 固定 R5 representation 是否线性可分 `V>=28.9 km`？  
**基线** R5-Transformer-28.9Gate-Diffusion-SNRBoundary encoder。  
**输入** same as baseline。  
**Backbone** frozen R5 encoder。  
**输出分布/任务** linear sigmoid probe。  
**高能见度端处理** 28.9 km Gate diagnostic label。  
**连续变量定义** 无。  
**Loss** Gate BCE。  
**相对上一实验唯一/主要改动** 仅训练线性 probe；encoder 冻结；无 continuous branch。  
**训练方式** best10000，stop16000 early stop；4 DCU DDP。  
**正式结果** AUROC/AP/balanced-acc/precision/recall/Brier=`0.86661/0.68496/0.71982/0.69914/0.51149/0.12119`。排除 ±0.2/0.5/1.0 km 后 AP=`0.68729/0.68950/0.69538`。  
**一句话结论** frozen representation 上仅中等线性可分。  
**真实路径** config `/public/home/putianshu/vis_mlp/gate_separability_20260915/configs/gate_study.json`；result `/public/home/putianshu/vis_mlp/gate_separability_20260915/gate_separability_summary.json`；checkpoint/result root `/public/home/putianshu/vis_mlp/gate_separability_20260915/runs/A_linear_frozen`；log root `/public/home/putianshu/vis_mlp/gate_separability_20260915/logs`；slurm `/public/home/putianshu/vis_mlp/gate_separability_20260915/scripts/sub_gate.slurm`。  
**Slurm job ID** 汇总未保存主 job ID（unknown）。

### R5-Gate-Separability-B-MLPFrozen

**状态** finished。  
**研究问题** 增加 probe 非线性容量能否显著改善 28.9-km separability？  
**基线** R5-Gate-Separability-A-LinearFrozen。  
**输入** same as A。  
**Backbone** frozen R5 encoder。  
**输出分布/任务** 256→64→1 MLP sigmoid probe。  
**高能见度端处理** 28.9 km Gate diagnostic label。  
**连续变量定义** 无。  
**Loss** Gate BCE。  
**相对上一实验唯一/主要改动** 线性 probe 换为 MLP；encoder/data/label 不变。  
**训练方式** best/stop20000；4 DCU DDP。  
**正式结果** AUROC/AP/balanced-acc/precision/recall/Brier=`0.87513/0.70235/0.73226/0.70336/0.53870/0.11775`；相对 A AP +0.01739、Recall +0.02720。排除 ±0.2/0.5/1.0 km 后 AP=`0.70477/0.70719/0.71331`。  
**一句话结论** nonlinear head 只提供有限增益，标签仍非清晰可分。  
**真实路径** config `/public/home/putianshu/vis_mlp/gate_separability_20260915/configs/gate_study.json`；result `/public/home/putianshu/vis_mlp/gate_separability_20260915/gate_separability_summary.json`；checkpoint/result root `/public/home/putianshu/vis_mlp/gate_separability_20260915/runs/B_mlp_frozen`；log root `/public/home/putianshu/vis_mlp/gate_separability_20260915/logs`；slurm `/public/home/putianshu/vis_mlp/gate_separability_20260915/scripts/sub_gate.slurm`。  
**Slurm job ID** unknown。

### R5-Gate-Separability-C-MLPEndToEnd

**状态** finished。  
**研究问题** encoder end-to-end finetune 能否解决 28.9-km label overlap？  
**基线** R5-Gate-Separability-B-MLPFrozen。  
**输入** same as B。  
**Backbone** end-to-end R5 Transformer。  
**输出分布/任务** 256→64→1 MLP sigmoid probe。  
**高能见度端处理** 28.9 km Gate diagnostic label。  
**连续变量定义** 无。  
**Loss** Gate BCE。  
**相对上一实验唯一/主要改动** 解冻 encoder；MLP、data、label 不变。  
**训练方式** best/stop20000；4 DCU DDP。  
**正式结果** AUROC/AP/balanced-acc/precision/recall/Brier=`0.87856/0.71130/0.73814/0.70521/0.55155/0.11605`；相对 B AP +0.00895、Recall +0.01285。排除 ±0.2/0.5/1.0 km 后 AP=`0.71381/0.71628/0.72259`。  
**一句话结论** encoder 微调与 margin exclusion 仍未让 28.9-km Gate 产生质变。  
**真实路径** config `/public/home/putianshu/vis_mlp/gate_separability_20260915/configs/gate_study.json`；result `/public/home/putianshu/vis_mlp/gate_separability_20260915/gate_separability_summary.json`；checkpoint/result root `/public/home/putianshu/vis_mlp/gate_separability_20260915/runs/C_mlp_e2e`；log root `/public/home/putianshu/vis_mlp/gate_separability_20260915/logs`；slurm `/public/home/putianshu/vis_mlp/gate_separability_20260915/scripts/sub_gate.slurm`。  
**Slurm job ID** unknown。

## 三种高能见度语义（不可混用）

- **28.9 km Gate**：`P(V>=28.9 km|X)`；来源是观测频率转折，用于 R5 Transformer Gate 系列。
- **30 km Cap probability**：`P(Y=30 km|X)`；针对观测上限点质量，是 R5-Cap 的独立 soft mixture head。
- **30 km censored likelihood**：当 `Y=30` 时只使用潜在 `V>=30` 的信息；没有 Gate/Cap head，是 R5-CensoredFlow 的统一 likelihood。

## 名称、语义与记录问题

1. 两个不同工程都把真实 route 写成 `R5-Diffusion`：原始 R5 Transformer root 与 SNRBoundary root。本文用两个 canonical display name 区分，未改真实目录。
2. `R5-current-gate-baseline` 是短期名称，长期语义不清；canonical name 为 `R5-Transformer-28.9Gate-Diffusion-SNRBoundary`，真实文件不改。
3. `R5-Cap` 的本地 `run_summary.json` 仍写 `S1_running`，但集群实际 S1/S2 与 final test 均已完成；以正式 `test_metrics.json` 为准。
4. `R5-CensoredFlow` 的本地 `run_summary.json` 仍写 `implementing/not_started`，但 sanity 已完成、S1 job `122197992` 在 step8000 因 NaN 失败；无 S2/test。
5. 原始 R5 Gaussian 目录存在 `S1_invalid_collision_122006578_122006928`，并有多次取消/失败重试；它是运行事故记录，不另算科学实验。
6. R0 隔离结果只保存三 seed ensemble probabilities/metrics，没有完整列出三 seed checkpoint/job ID 与 loss 权重；这些字段保持 unknown。
7. Gate separability 汇总未保存三个 probe 的 Slurm job ID；指标、config、run path 完整。
8. R4b 的 `support_checks` 用 `<=E_min`/`>=30` 统计浮点边界，出现非零计数，与设计文本“理论应为0”存在数值语义差异；不删除、不重算。
