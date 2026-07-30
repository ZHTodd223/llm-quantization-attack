# Qwen3.5-4B-Base 服务器运行手册

从仓库根目录执行。每条命令失败即停止；每个 JSON 都是后续判断需要保留的原始证据，不能据此提前声称模型兼容。本手册不包含真实 token、服务器地址或用户路径。

## 1. 创建服务器环境

```bash
export HF_TOKEN='在当前 shell 设置，不要写入命令历史'
export HF_HOME="$PWD/.hf"
export MODEL_ID='Qwen/Qwen3.5-4B-Base'
export OUTPUT_ROOT="$PWD/output/qwen35_probe"
mkdir -p "$HF_HOME" "$OUTPUT_ROOT"
conda create -n qwen35-attack python=3.11 -y
conda activate qwen35-attack
```

输入：服务器上的仓库和 Conda。输出：`qwen35-attack` 环境及输出目录。通过条件：环境创建成功；失败时停止。

## 2. 安装依赖

```bash
pip install -r requirements.txt -r AutoPoison/requirements.txt
```

输入：仓库 requirements。输出：已安装的本地环境。通过条件：命令退出码为零；失败时停止。

## 3. 下载并冻结模型 revision

```bash
hf download "$MODEL_ID" --token "$HF_TOKEN" --local-dir "$OUTPUT_ROOT/model"
python - <<'PY'
from transformers import AutoConfig
c = AutoConfig.from_pretrained('output/qwen35_probe/model', local_files_only=True)
print(c._commit_hash or 'REVISION_NOT_EXPOSED: 请单独记录 hf download 的 revision')
PY
```

输入：`HF_TOKEN` 和 `MODEL_ID`。输出：`$OUTPUT_ROOT/model` 与记录的 revision。通过条件：下载完成且 revision 已记录；失败时停止。

## 4. 执行模型结构扫描

```bash
python -m AutoPoison.probes.inspect_qwen35_model \
  --model-id "$OUTPUT_ROOT/model" --dtype bfloat16 \
  --output "$OUTPUT_ROOT/structure.json"
```

输入：冻结后的本地模型。输出：`structure.json`，含完整参数名、shape、dtype、视觉/语言分类、embedding、lm_head、norm、DeltaNet、attention、FFN 与未知项。通过条件：JSON 成功写入；失败时停止。

## 5. 执行 INT8 目标审计

```bash
python -m AutoPoison.probes.audit_qwen35_int8_targets \
  --model-id "$OUTPUT_ROOT/model" --dtype bfloat16 \
  --output "$OUTPUT_ROOT/int8-targets.json"
```

输入：冻结后的本地模型。输出：`int8-targets.json`，逐参数给出是否进入原 `compute_box_int8` 路径及明确排除原因。通过条件：退出码为零且 `unknown_language_parameters` 为空；否则停止并审查报告。

## 6--12. 单阶段显存探针

每次命令只执行一个阶段，输入均为 `$OUTPUT_ROOT/model`，输出为对应的 `memory-<stage>.json`。默认 batch size 为 1、序列长度为 64、BF16、`use_cache=false` 和 gradient checkpointing。

```bash
for stage in load forward backward optimizer-init optimizer-step constraint-build projection; do
  python -m AutoPoison.probes.qwen35_memory_probe \
    --model-id "$OUTPUT_ROOT/model" --stage "$stage" \
    --output "$OUTPUT_ROOT/memory-$stage.json" || exit $?
done
```

阶段依次是：6 `load`、7 `forward`、8 `backward`、9 `optimizer-init`、10 `optimizer-step`、11 `constraint-build`、12 `projection`。每一步通过条件是 JSON 中 `success=true` 且有峰值显存记录；OOM 会写出异常 JSON 后以非零状态退出。禁止在此脚本外把失败改成更小模型、LoRA/QLoRA、CPU/disk offload 或 `device_map=auto`；失败时停止。

## 13. 单步 Injection 探针

仅在步骤 4--12 全部通过后运行。训练样本由原始数据格式中的 1--4 条组成，脚本固定 `batch_size=1`、`max_steps=1`，并走原始算法路径。

```bash
python -m AutoPoison.probes.qwen35_injection_one_step \
  --model-path "$OUTPUT_ROOT/model" \
  --data-path AutoPoison/data/alpaca_gpt4_data.json \
  --poison-data-path AutoPoison/data/autopoison_gpt-3.5-turbo_mcd-injection_ns5200_from0_seed0.jsonl \
  --output-dir "$OUTPUT_ROOT/injection"
```

输入：冻结模型和现有原始格式数据。输出：独立的 `$OUTPUT_ROOT/injection` checkpoint/log。通过条件：forward、backward、optimizer step 均成功，至少一个目标参数变化且 checkpoint 保存；任何缺失或异常均停止。这只是代码路径探针，不代表攻击成功。

## 14. 单步 Repair 探针

仅使用步骤 13 产出的 Injection checkpoint。

```bash
python -m AutoPoison.probes.qwen35_repair_one_step \
  --injected-checkpoint "$OUTPUT_ROOT/injection/checkpoint-1" \
  --data-path AutoPoison/data/alpaca_gpt4_data.json \
  --poison-data-path AutoPoison/data/autopoison_gpt-3.5-turbo_mcd-injection_ns5200_from0_seed0.jsonl \
  --output-dir "$OUTPUT_ROOT/repair"
```

输入：步骤 13 的 checkpoint 和相同格式数据。输出：独立的 `$OUTPUT_ROOT/repair` checkpoint/log；应保留约束目标数量、构建成功数量、投影参数数量、违规数量，以及投影前后浮点/量化映射差异。通过条件：INT8 约束构建、一次 Repair step、一次原始 PGD 投影与重新计算映射均成功；任何未知语言参数、约束或投影差异、缺少 checkpoint 或异常均停止。该探针不代表攻击效果已验证。
