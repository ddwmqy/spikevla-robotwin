# RoboTwin 四臂启动手册(算力服务器)

> 配套 `v4_robotwin/PLAN_CN.md` v7 与 `v4_robotwin/ASSETS.md`。
> 本仓库路径:`/data/260010028/dwh_vla/v4_code`(共享卷,算力服务器直接可见)。

## 0. 前置(每台机器一次)

```bash
cd /data/260010028/dwh_vla/v4_code
pip install -e ".[robotwin]"          # 建 turbovla-robotwin env(python 3.10)
pip install flash-attn                 # C1/B 的 vision.attn_implementation=flash_attention_2 需要
```

环境变量(四臂共用,建议写进 `env.sh`):

```bash
export STARVLA_PYTHON=/path/to/envs/turbovla-robotwin/bin/python   # train.sh 默认用系统 python!
export ROBOTWIN_PYTHON="${STARVLA_PYTHON}"                          # 评测侧同理
export ROBOTWIN_DATA_ROOT=/data/260010028/dwh_vla/v4_assets/robotwin_data/RoboTwin
export DINOV3_MODEL_PATH=/data/260010028/dwh_vla/v4_assets/dinov3-vitl16-pretrain-lvd1689m
export BERT_MODEL_PATH=/data/260010028/dwh_vla/v2_code_bundle_20260906/resources/pretrained/bert-base-uncased
export TURBOVLA_INIT_CKPT=/data/260010028/dwh_vla/v4_assets/groundingdino/groundingdino_swint_ogc.pth
export SMOOTHSPIKE_MODEL_PATH=/data/260010028/dwh_vla/v2_code_bundle_20260906/resources/pretrained/SmoothSpike/smoothspike-bert-base-fused
export SDTV3_WEIGHTS_PATH=/data/260010028/dwh_vla/v2_code_bundle_20260906/resources/pretrained/V3_19.0M_1x4.pth
export SDTV3_PROCESSOR_PATH=/data/260010028/dwh_vla/v2_code_bundle_20260906/resources/pretrained/dinov3-vitb16-pretrain-lvd1689m
```

`ROBOTWIN_DATA_ROOT` 必须指向**含 `Clean/` 子目录**的位置:HF 仓库 `StarVLA/RoboTwin-Clean`
是平铺的任务目录,而训练注册表按 `Clean/<task_name>` 解析。本地已建软链
`v4_assets/robotwin_data/RoboTwin/Clean -> ../StarVLA_RoboTwin_Clean`,共享卷上直接可用。

`train.sh` 还会**强制检查四个文件存在**(`-f`),不区分该臂是否真的使用:即便是
`load_pretrained: false` 的 A 臂,`TURBOVLA_INIT_CKPT` 也必须指向一个**真实文件**
(不能是 `/dev/null` 这类字符设备)。

## 1. 四臂一览

| 臂 | config | 文本 | 视觉 | 交互 | 初始化 |
|---|---|---|---|---|---|
| **C1** | `clean50.yaml`(官方未改) | BERT(可载) | DINOv3 ViT-L | ANN | GroundingDINO,`load_bert: true` |
| **B** | `clean50_b.yaml` | **sootspike**(T=4,冻结) | DINOv3 ViT-L | ANN | 同上但 **`load_bert: false`** |
| **A-S**(冒烟) | `clean50_a.yaml` | sootspike | **SDT-V3 19M** | **Spike2Max** | **from-scratch** |
| **A**(正式) | `clean50_a.yaml` | 同上 | 同上 | 同上 | 同上(待 B 结论拍板) |

C1 与 B 的唯一差异 = 文本编码器(`mask_version` 随之 legacy→corrected,已声明的组合变量,见 PLAN §4)。

## 2. 启动命令

```bash
# ---- C1:官方配方(单变量对照的基线) ----
RUN_ID=c1_bert_ann_<日期> MAX_TRAIN_STEPS=55000 NUM_PROCESSES=4 PER_DEVICE_BATCH_SIZE=48 \
  bash scripts/robotwin/train.sh

# ---- B:sootspike 文本(官方架构其余不变) ----
CONFIG_YAML=experiments/robotwin/configs/clean50_b.yaml \
RUN_ID=b_sootspike_ann_<日期> MAX_TRAIN_STEPS=55000 NUM_PROCESSES=4 PER_DEVICE_BATCH_SIZE=48 \
  bash scripts/robotwin/train.sh

# ---- A-S:1k 冒烟(数字不作结论,不进对照表;命名 asmoke_<日期>) ----
CONFIG_YAML=experiments/robotwin/configs/clean50_a.yaml \
RUN_ID=asmoke_<日期> MAX_TRAIN_STEPS=1000 NUM_PROCESSES=4 PER_DEVICE_BATCH_SIZE=48 \
  bash scripts/robotwin/train.sh
```

单卡凑全局 192:`PER_DEVICE_BATCH_SIZE=48 GRADIENT_ACCUMULATION_STEPS=4`(§5.4 修正版
trainer 已让 scheduler/EMA/门禁按 optimizer step 计数;两臂必须同档位)。

**额外覆盖参数**:`train.sh "$@"` 会透传,但**必须带 `--` 前缀**——
`normalize_dotlist_args` 会静默丢弃裸 `key=value`(本地实测踩过):

```bash
bash scripts/robotwin/train.sh --trainer.save_interval 250 --trainer.eval_interval 250
```

## 3. 启动后必看的检查项

- **日志核对 `[TurboVLA] loaded N initialization tensors`**(真实 GroundingDINO ckpt 实测):
  | 臂 | 期望值 | 说明 |
  |---|---|---|
  | **C1** | **381** | 真 BERT(200)+ 投影(2)+ 文本层(72)+ 融合层(108)= 382,减 1 个形状不匹配 |
  | **B** | **182** | 仅投影+文本层+融合层;**不含那 200 个普通 BERT 张量** |
  B 若显示 331 → `load_bert` 没生效,sootspike 会被半覆盖(实测 149/211 张被改,max|Δ| 2.05)→ **立刻停**。
  事后可复核:`python scripts/check_b_init_hash.py`(需同一份 init ckpt)。
- **A 臂**:`text.timesteps` 必须为 4(融合 ckpt 固定 T=4,否则加载即拒)。
- 训练前可先跑构建自检(CPU 数分钟,能挡住配置类错误):
  `python scripts/gate2b_wrapper_build.py experiments/robotwin/configs/<arm>.yaml`
- **§5.4 计数验证**(短跑,推荐在 55k 前跑一次):见 `scripts/robotwin/verify_54_counting.py`
  的 docstring;本地已用真实数据+真实模型在 accum=1/4 及 DeepSpeed 路径下验证通过。

## 4. 已知坑(本地已定位/已修)

| 坑 | 现象 | 处理 |
|---|---|---|
| 覆盖参数无 `--` 前缀 | 参数静默失效,yaml 原值生效 | 一律 `--key value` 或 `--key=value` |
| 数据目录缺 `Clean/` 层 | `dataset_path` 找不到 | 见 §0 的软链 |
| `torch.load` 默认 `weights_only=True`(torch≥2.6) | 官方 init ckpt(含 `args` Namespace)读入即 `UnpicklingError` | 已修:`share_tools.load_checkpoint_file`(safetensors + `weights_only=False`),wrapper/base_framework/trainer_tools 三处统一 |
| 单进程裸跑训练脚本 | `dist.get_rank()` 报进程组未初始化 | 用 `train.sh`(内部 `accelerate launch --num_processes N`)或自行 init 进程组 |
| DINOv3 走 hf-mirror 403 | gated 仓库需 token | 用 ModelScope 官方镜像(见 ASSETS.md) |
| 视觉 `flash_attention_2` 在 CPU 上前向失败 | CPU 只能做构型自检 | 自检时加 `--framework.vision.attn_implementation=sdpa` |

## 5. 产物与评测

每个 run 目录(`results/Checkpoints/<run_id>/`)含 `checkpoints/steps_N_pytorch_model.pt`
与 `steps_N_ema_pytorch_model.pt`(EMA)、`config.yaml`、`dataset_statistics.json`、
`summary.jsonl`。**评测口径:EMA-55k**(`steps_55000_ema_pytorch_model.pt`),
`config.yaml` + `dataset_statistics.json` 必须与 ckpt 同目录祖先(评测栈要求)。

```bash
export ROBOTWIN_PATH=/data/260010028/dwh_vla/RoboTwin
ROBOTWIN_TEST_NUM=100 bash scripts/robotwin/evaluate.sh \
  results/Checkpoints/<run_id>/checkpoints/steps_55000_ema_pytorch_model.pt
```
