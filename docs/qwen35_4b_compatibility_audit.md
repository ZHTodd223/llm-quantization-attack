# Qwen3.5-4B-Base compatibility audit

Status: local static audit only. `SERVER_VALIDATION_REQUIRED` means no checkpoint was downloaded or loaded here.

## Evidence from this repository

- `model_config.sh` is the shell registry used by download/clean-repair paths. It previously lists Qwen2.5, Phi, StarCoder, and Llama keys.
- `safecoder/safecoder/constants.py` is the Python `PRETRAINED_MODELS` registry used by SafeCoder evaluators.
- `AutoPoison/main.py` loads training and evaluation models with `transformers.AutoModelForCausalLM`, and tokenizer with `AutoTokenizer`; its `Trainer` path is full-parameter by default. It does not select a PEFT/LoRA target.
- `AutoPoison/quant_specific/pgd.py:compute_box` creates a separately quantized model for BNB methods, then calls `q_attack/repair/train.py:get_quantize_target_layers`.
- That target finder recursively zips full and quantized module trees and selects only `Linear8bitLt`/`Linear4bit` leaves corresponding to `nn.Linear` or `Conv1D`; `compute_pgd_box` rejects non-2D weights. `compute_box_int8` is therefore applied to those selected two-dimensional weights only.
- Repair projection is `PGDCallback.on_step_end`, which clamps only parameter names present in `box`. Save/reload uses Hugging Face `save_pretrained`/`from_pretrained`, not a fixed state-dict key map.

## Qwen3.5 adaptation

`qwen3.5-4b-base` now maps to `Qwen/Qwen3.5-4B-Base` in both applicable registries. `AutoPoison/model_adapters/qwen35.py` keeps the existing AutoModel loading strategy, disables `use_cache`, and enables gradient checkpointing only after Qwen3.5 configuration recognition.

No text-only extraction or multimodal weight remapping is implemented. The new scanners classify vision paths, embeddings, heads, norms, biases, non-matrices, and candidate language matrices. Unknown language parameters fail the int8 audit rather than being ignored.

## Compatibility risks requiring server validation

1. The official Qwen3.5 config/model class and actual `model_type` must resolve through the installed Transformers version.
2. Whether the checkpoint includes vision modules, and their exact names, must be observed from `named_parameters()`.
3. BNB replacement trees must match the full model tree; the legacy zip-based target finder will fail if they differ.
4. All candidate target weights must be 2-D, and BNB `compute_box_int8` must cover them; embeddings, norms and heads are deliberately excluded by policy.
5. Actual save/reload, forward/backward, optimizer, constraint and PGD behavior can only be verified with the frozen checkpoint on the server.
