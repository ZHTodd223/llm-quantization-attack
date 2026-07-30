"""Server-only conservative audit of candidate LLM.int8() targets."""
import argparse
import json
import traceback
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-id", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--dtype", default="bfloat16", choices=["bfloat16", "float16", "float32"])
    args = parser.parse_args()
    report = {
        "model_id": args.model_id,
        "targets": [],
        "unknown_language_parameters": [],
        "status": "failed_before_audit",
        "exception": None,
        "server_validation": "SERVER_VALIDATION_REQUIRED",
    }
    exit_code = 1
    try:
        # Keep these imports inside the guarded region so missing server packages
        # also become durable JSON evidence instead of an unexplained terminal exit.
        import torch
        from AutoPoison.model_adapters.qwen35 import (
            SERVER_VALIDATION_REQUIRED,
            load_causal_lm,
            scan_model,
        )

        model = load_causal_lm(
            args.model_id,
            torch_dtype=getattr(torch, args.dtype),
            device_map=None,
            trust_remote_code=True,
        )
        scan = scan_model(model)
        targets, unknown = [], []
        for item in scan["parameters"]:
            candidate = item["classification"] == "candidate"
            unknown_language = item["classification"] == "unknown" and item["name"].startswith(
                ("model.", "language_model.", "transformer.")
            )
            targets.append(
                {
                    "parameter_name": item["name"],
                    "shape": item["shape"],
                    "module_type": item["module_type"],
                    "enters_original_compute_box_int8": candidate,
                    "supports_constraint": candidate,
                    "exclusion_reason": item["exclusion_reason"],
                    "verification": SERVER_VALIDATION_REQUIRED,
                }
            )
            if unknown_language:
                unknown.append(item["name"])
        report.update(
            {
                "targets": targets,
                "unknown_language_parameters": unknown,
                "status": "failed_unknown_language_parameters" if unknown else "candidate_audit_complete",
                "server_validation": SERVER_VALIDATION_REQUIRED,
            }
        )
        exit_code = 2 if unknown else 0
    except Exception:
        report["exception"] = traceback.format_exc()
    finally:
        output = Path(args.output)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
