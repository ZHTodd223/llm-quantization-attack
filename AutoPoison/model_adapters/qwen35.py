"""Conservative Qwen3.5 adapter.

This module deliberately uses Transformers Auto classes.  Exact checkpoint module
names, multimodal composition, and quantizer replacements are server-validation
requirements; no parameter remapping is inferred locally.
"""
from __future__ import annotations

from collections import Counter
from typing import Any

QWEN35_MODEL_KEY = "qwen3.5-4b-base"
QWEN35_MODEL_ID = "Qwen/Qwen3.5-4B-Base"
SERVER_VALIDATION_REQUIRED = "SERVER_VALIDATION_REQUIRED"

VISION_TOKENS = ("vision", "visual", "image", "video", "vision_tower")
LANGUAGE_PREFIXES = ("model.", "language_model.", "transformer.")
KNOWN_LANGUAGE_TOKENS = (
    "embed_tokens", "lm_head", "norm", "self_attn", "attention", "linear_attn",
    "mlp", "ffn", "feed_forward", "deltanet", "delta_net", "mamba",
    "rotary", "rotary_emb",
)


def is_qwen35_config(config: Any) -> bool:
    """Recognize Qwen3.5 without requiring a concrete checkpoint class."""
    model_type = str(getattr(config, "model_type", "")).lower().replace("_", ".")
    architectures = " ".join(getattr(config, "architectures", []) or []).lower()
    return "qwen3.5" in model_type or "qwen3.5" in architectures or "qwen35" in model_type


def load_tokenizer(model_id: str, **kwargs: Any):
    from transformers import AutoTokenizer
    return AutoTokenizer.from_pretrained(model_id, **kwargs)


def load_causal_lm(model_id: str, **kwargs: Any):
    """Load through the official Auto mapping and apply only safe Qwen settings."""
    from transformers import AutoModelForCausalLM
    model = AutoModelForCausalLM.from_pretrained(model_id, **kwargs)
    if is_qwen35_config(model.config):
        model.config.use_cache = False
        if hasattr(model, "gradient_checkpointing_enable"):
            model.gradient_checkpointing_enable()
    return model


def classify_parameter(name: str, param: Any, module: Any | None = None) -> tuple[str, str | None]:
    """Return (classification, exclusion_reason).  Never emits vague skip labels."""
    lowered = name.lower()
    if any(token in lowered for token in VISION_TOKENS):
        return "vision", "VISION_MODULE"
    if name.endswith(".bias"):
        return "excluded", "BIAS"
    if "norm" in lowered or ".ln" in lowered:
        return "excluded", "NORMALIZATION"
    if "embed" in lowered:
        return "excluded", "EMBEDDING_POLICY"
    if "lm_head" in lowered:
        return "excluded", "LM_HEAD_POLICY"
    ndim = getattr(param, "ndim", None)
    if ndim != 2:
        return "excluded", "NON_MATRIX_PARAMETER"
    if not name.startswith(LANGUAGE_PREFIXES):
        return "unknown", "UNSUPPORTED_MODULE"
    if not any(token in lowered for token in KNOWN_LANGUAGE_TOKENS):
        return "unknown", "UNSUPPORTED_MODULE"
    return "candidate", None


def module_family(name: str, module: Any) -> str:
    lowered = name.lower()
    if "deltanet" in lowered or "delta_net" in lowered or "linear_attn" in lowered:
        return "deltanet"
    if "attn" in lowered or "attention" in lowered:
        return "attention"
    if any(token in lowered for token in ("mlp", "ffn", "feed_forward")):
        return "ffn"
    return type(module).__name__


def scan_model(model: Any) -> dict[str, Any]:
    """Create a serializable structure report without assuming Qwen3.5 paths."""
    modules = dict(model.named_modules())
    records, families, dtypes = [], Counter(), Counter()
    embeddings, heads, norms, unknown = [], [], [], []
    total = trainable = language = vision = 0
    for name, param in model.named_parameters():
        count = int(param.numel())
        total += count
        trainable += count if param.requires_grad else 0
        module_name = name.rsplit(".", 1)[0] if "." in name else ""
        module = modules.get(module_name)
        classification, reason = classify_parameter(name, param, module)
        if classification == "vision": vision += count
        elif classification in ("candidate", "excluded", "unknown") and name.startswith(LANGUAGE_PREFIXES): language += count
        if module is not None: families[module_family(module_name, module)] += 1
        dtypes[str(param.dtype)] += count
        records.append({"name": name, "shape": list(param.shape), "dtype": str(param.dtype),
                        "module_type": type(module).__name__ if module is not None else None,
                        "classification": classification, "exclusion_reason": reason})
        if "embed" in name.lower(): embeddings.append(name)
        if "lm_head" in name.lower(): heads.append(name)
        if "norm" in name.lower() or ".ln" in name.lower(): norms.append(name)
        if classification == "unknown": unknown.append(name)
    return {"model_class": type(model).__name__, "config_class": type(model.config).__name__,
            "config_model_type": getattr(model.config, "model_type", None), "total_parameters": total,
            "trainable_parameters": trainable, "module_type_counts": dict(Counter(type(m).__name__ for m in modules.values())),
            "parameter_dtype_counts": dict(dtypes), "module_family_counts": dict(families),
            "language_module_parameters": language, "vision_module_parameters": vision,
            "embedding_parameters": embeddings, "lm_head_parameters": heads,
            "norm_parameters": norms, "deltanet_modules": [name for name in modules if "deltanet" in name.lower() or "delta_net" in name.lower() or "linear_attn" in name.lower()],
            "attention_modules": [name for name in modules if "attn" in name.lower() or "attention" in name.lower()],
            "ffn_modules": [name for name in modules if any(t in name.lower() for t in ("mlp", "ffn", "feed_forward"))],
            "unknown_modules": unknown,
            "parameters": records, "server_validation": SERVER_VALIDATION_REQUIRED}
