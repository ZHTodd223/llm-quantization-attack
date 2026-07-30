# Qwen3.5-4B-Base server runbook

Run from the repository root. Stop at the first failing command; each JSON file is required evidence, not a claim of compatibility. Set no real token in this document.

```bash
export HF_TOKEN='set-in-shell'; export HF_HOME="$PWD/.hf"; export MODEL_ID='Qwen/Qwen3.5-4B-Base'; export OUTPUT_ROOT="$PWD/output/qwen35_probe"
mkdir -p "$HF_HOME" "$OUTPUT_ROOT"
conda create -n qwen35-attack python=3.11 -y && conda activate qwen35-attack
pip install -r requirements.txt -r AutoPoison/requirements.txt
hf download "$MODEL_ID" --token "$HF_TOKEN" --local-dir "$OUTPUT_ROOT/model"
python - <<'PY'
from transformers import AutoConfig
c=AutoConfig.from_pretrained('output/qwen35_probe/model', local_files_only=True)
print(c._commit_hash or 'REVISION_NOT_EXPOSED: record hf download revision separately')
PY
```

Input is the frozen local model at `$OUTPUT_ROOT/model`; pass only when the revision is recorded separately and installation succeeds. Then run the structure scan; pass only when it writes JSON:

```bash
python -m AutoPoison.probes.inspect_qwen35_model --model-id "$OUTPUT_ROOT/model" --dtype bfloat16 --output "$OUTPUT_ROOT/structure.json"
python -m AutoPoison.probes.audit_qwen35_int8_targets --model-id "$OUTPUT_ROOT/model" --dtype bfloat16 --output "$OUTPUT_ROOT/int8-targets.json"
```

The second command must exit zero and report no `unknown_language_parameters`; otherwise stop and inspect the report. Run exactly one memory stage per command; input is the same frozen model and output is one stage JSON. Pass only if `success` is true and peak memory is recorded.

```bash
for stage in load forward backward optimizer-init optimizer-step constraint-build projection; do
  python -m AutoPoison.probes.qwen35_memory_probe --model-id "$OUTPUT_ROOT/model" --stage "$stage" --output "$OUTPUT_ROOT/memory-$stage.json" || exit $?
done
```

The loop is sequential but each invocation performs one stage only. It never selects another model, LoRA/QLoRA, CPU/disk offload, or `device_map=auto`; an OOM produces JSON then fails.

Only after all reports pass, use 1--4 existing training samples and the repository's original data formats. These are code-path probes, not attack experiments:

```bash
python -m AutoPoison.probes.qwen35_injection_one_step --model-path "$OUTPUT_ROOT/model" --data-path AutoPoison/data/alpaca_gpt4_data.json --poison-data-path AutoPoison/data/autopoison_gpt-3.5-turbo_mcd-injection_ns5200_from0_seed0.jsonl --output-dir "$OUTPUT_ROOT/injection"
python -m AutoPoison.probes.qwen35_repair_one_step --injected-checkpoint "$OUTPUT_ROOT/injection/checkpoint-1" --data-path AutoPoison/data/alpaca_gpt4_data.json --poison-data-path AutoPoison/data/autopoison_gpt-3.5-turbo_mcd-injection_ns5200_from0_seed0.jsonl --output-dir "$OUTPUT_ROOT/repair"
```

The injection pass condition is forward, backward, optimizer step, changed target parameter, and a saved checkpoint. The Repair pass condition is successful int8 constraint construction, one original PGD projection, and generated checkpoint/log evidence. Stop for any missing checkpoint, exception, unknown language parameter, or constraint/projection discrepancy.
