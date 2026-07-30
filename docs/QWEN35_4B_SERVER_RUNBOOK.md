# Qwen3.5-4B-Base 服务器运行手册

从仓库根目录执行。按编号逐段复制粘贴，上一段成功后再执行下一段。每条命令失败即停止；每个 JSON 都是后续判断需要保留的原始证据，不能据此提前声称模型兼容。本手册不包含真实 token、服务器地址或用户路径。

## 1. 创建服务器环境

```bash
cd /mnt/workspace/llm-quantization-attack
set -euo pipefail

# HF_TOKEN 仅为 Transformers/HF 受限资源保留；公开的 ModelScope 下载通常不需要它。
export HF_TOKEN='在当前 shell 设置，不要写入命令历史'
export HF_HOME="$PWD/.hf"
export MODEL_ID='Qwen/Qwen3.5-4B-Base'
export MS_MODEL_ID='Qwen/Qwen3.5-4B-Base'
export OUTPUT_ROOT="$PWD/output/qwen35_probe"
mkdir -p "$HF_HOME" "$OUTPUT_ROOT"
printf '仓库: %s\n输出目录: %s\nModelScope 模型: %s\n' "$PWD" "$OUTPUT_ROOT" "$MS_MODEL_ID"
conda create -n qwen35-attack python=3.11 -y
conda activate qwen35-attack
python --version
```

输入：服务器上的仓库和 Conda。输出：`qwen35-attack` 环境、输出目录和模型 ID。通过条件：最后输出 Python 3.11 且上述目录正确；失败时停止。

## 2. 安装依赖

```bash
pip install -r requirements.txt -r AutoPoison/requirements.txt modelscope
modelscope --help >/dev/null
echo '依赖安装和 ModelScope CLI 检查通过'
```

输入：仓库 requirements。输出：已安装的本地环境。通过条件：命令退出码为零；失败时停止。

## 3. 下载并冻结模型 revision

```bash
# 自动解析 ModelScope Git 远端当前 HEAD 的不可变 commit SHA；绝不以漂移的 main/master 作为下载 revision。
export MODEL_REVISION="$(git ls-remote "https://www.modelscope.cn/${MS_MODEL_ID}.git" HEAD | awk 'NR==1 {print $1}')"
test -n "$MODEL_REVISION"
printf '将下载 %s@%s\n' "$MS_MODEL_ID" "$MODEL_REVISION"
modelscope download --model "$MS_MODEL_ID" --revision "$MODEL_REVISION" --local_dir "$OUTPUT_ROOT/model"
printf '%s\n' "$MS_MODEL_ID@$MODEL_REVISION" | tee "$OUTPUT_ROOT/model/MODEL_SOURCE_AND_REVISION.txt"
test -f "$OUTPUT_ROOT/model/config.json"
echo 'ModelScope 下载和 revision 记录通过'
```

输入：步骤 1 的 `MS_MODEL_ID`。输出：`$OUTPUT_ROOT/model`、`MODEL_SOURCE_AND_REVISION.txt` 和自动解析的 commit SHA。通过条件：下载完成、`config.json` 存在且记录文件含模型 ID 与 SHA；失败时停止。`MODEL_ID` 仅用于标识原始模型；后续脚本一律加载下载后的本地目录。

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

## 6. `load` 显存探针

```bash
python -m AutoPoison.probes.qwen35_memory_probe --model-id "$OUTPUT_ROOT/model" --stage load --output "$OUTPUT_ROOT/memory-load.json"
python -c "import json; r=json.load(open('$OUTPUT_ROOT/memory-load.json')); assert r['success'], r['exception']; print(r)"
```

输出：`memory-load.json`。通过条件：`success=true` 且 `peak_allocated` 有数值；失败时停止。

## 7. `forward` 显存探针

```bash
python -m AutoPoison.probes.qwen35_memory_probe --model-id "$OUTPUT_ROOT/model" --stage forward --output "$OUTPUT_ROOT/memory-forward.json"
python -c "import json; r=json.load(open('$OUTPUT_ROOT/memory-forward.json')); assert r['success'], r['exception']; print(r)"
```

输出：`memory-forward.json`。通过条件：`success=true`；失败时停止。

## 8. `backward` 显存探针

```bash
python -m AutoPoison.probes.qwen35_memory_probe --model-id "$OUTPUT_ROOT/model" --stage backward --output "$OUTPUT_ROOT/memory-backward.json"
python -c "import json; r=json.load(open('$OUTPUT_ROOT/memory-backward.json')); assert r['success'], r['exception']; print(r)"
```

输出：`memory-backward.json`。通过条件：`success=true`；失败时停止。

## 9. `optimizer-init` 显存探针

```bash
python -m AutoPoison.probes.qwen35_memory_probe --model-id "$OUTPUT_ROOT/model" --stage optimizer-init --output "$OUTPUT_ROOT/memory-optimizer-init.json"
python -c "import json; r=json.load(open('$OUTPUT_ROOT/memory-optimizer-init.json')); assert r['success'], r['exception']; print(r)"
```

输出：`memory-optimizer-init.json`。通过条件：`success=true` 且 `optimizer_type` 非空；失败时停止。

## 10. `optimizer-step` 显存探针

```bash
python -m AutoPoison.probes.qwen35_memory_probe --model-id "$OUTPUT_ROOT/model" --stage optimizer-step --output "$OUTPUT_ROOT/memory-optimizer-step.json"
python -c "import json; r=json.load(open('$OUTPUT_ROOT/memory-optimizer-step.json')); assert r['success'], r['exception']; print(r)"
```

输出：`memory-optimizer-step.json`。通过条件：`success=true`；失败时停止。

## 11. `constraint-build` 显存探针

```bash
python -m AutoPoison.probes.qwen35_memory_probe --model-id "$OUTPUT_ROOT/model" --stage constraint-build --output "$OUTPUT_ROOT/memory-constraint-build.json"
python -c "import json; r=json.load(open('$OUTPUT_ROOT/memory-constraint-build.json')); assert r['success'], r['exception']; print(r)"
```

输出：`memory-constraint-build.json`。通过条件：`success=true`；失败时停止。

## 12. `projection` 显存探针

```bash
python -m AutoPoison.probes.qwen35_memory_probe --model-id "$OUTPUT_ROOT/model" --stage projection --output "$OUTPUT_ROOT/memory-projection.json"
python -c "import json; r=json.load(open('$OUTPUT_ROOT/memory-projection.json')); assert r['success'], r['exception']; print(r)"
```

输出：`memory-projection.json`。通过条件：`success=true`；失败时停止。以上探针均固定 batch size 1、序列长度 64、BF16、`use_cache=false` 与 gradient checkpointing；禁止改用更小模型、LoRA/QLoRA、CPU/disk offload 或 `device_map=auto`。

## 13. 单步 Injection 探针

仅在步骤 4--12 全部通过后运行。训练样本由原始数据格式中的 1--4 条组成，脚本固定 `batch_size=1`、`max_steps=1`，并走原始算法路径。

```bash
python -m AutoPoison.probes.qwen35_injection_one_step \
  --model-path "$OUTPUT_ROOT/model" \
  --data-path AutoPoison/data/alpaca_gpt4_data.json \
  --poison-data-path AutoPoison/data/autopoison_gpt-3.5-turbo_mcd-injection_ns5200_from0_seed0.jsonl \
  --output-dir "$OUTPUT_ROOT/injection"
test -f "$OUTPUT_ROOT/injection/checkpoint-last/config.json"
echo "Injection checkpoint: $OUTPUT_ROOT/injection/checkpoint-last"
```

输入：冻结模型和现有原始格式数据。输出：独立的 `$OUTPUT_ROOT/injection` checkpoint/log。通过条件：forward、backward、optimizer step 均成功，至少一个目标参数变化且 checkpoint 保存；任何缺失或异常均停止。这只是代码路径探针，不代表攻击成功。

## 14. 单步 Repair 探针

仅使用步骤 13 产出的 Injection checkpoint。

```bash
python -m AutoPoison.probes.qwen35_repair_one_step \
  --injected-checkpoint "$OUTPUT_ROOT/injection/checkpoint-last" \
  --data-path AutoPoison/data/alpaca_gpt4_data.json \
  --poison-data-path AutoPoison/data/autopoison_gpt-3.5-turbo_mcd-injection_ns5200_from0_seed0.jsonl \
  --output-dir "$OUTPUT_ROOT/repair"
test -f "$OUTPUT_ROOT/repair/checkpoint-last/config.json"
echo "Repair checkpoint: $OUTPUT_ROOT/repair/checkpoint-last"
```

输入：步骤 13 的 checkpoint 和相同格式数据。输出：独立的 `$OUTPUT_ROOT/repair` checkpoint/log；应保留约束目标数量、构建成功数量、投影参数数量、违规数量，以及投影前后浮点/量化映射差异。通过条件：INT8 约束构建、一次 Repair step、一次原始 PGD 投影与重新计算映射均成功；任何未知语言参数、约束或投影差异、缺少 checkpoint 或异常均停止。该探针不代表攻击效果已验证。
