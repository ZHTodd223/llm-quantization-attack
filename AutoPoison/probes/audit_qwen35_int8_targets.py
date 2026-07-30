"""Server-only conservative audit of candidate LLM.int8() targets."""
import argparse, json
from pathlib import Path
import torch
from AutoPoison.model_adapters.qwen35 import load_causal_lm, scan_model, SERVER_VALIDATION_REQUIRED

def main():
    p=argparse.ArgumentParser(); p.add_argument("--model-id", required=True); p.add_argument("--output", required=True); p.add_argument("--dtype", default="bfloat16", choices=["bfloat16","float16","float32"]); a=p.parse_args()
    model=load_causal_lm(a.model_id, torch_dtype=getattr(torch,a.dtype), device_map=None, trust_remote_code=True); scan=scan_model(model); targets=[]; unknown=[]
    for item in scan["parameters"]:
        candidate=item["classification"] == "candidate"; unknown_language=item["classification"] == "unknown" and item["name"].startswith(("model.","language_model.","transformer."))
        target={"parameter_name":item["name"],"shape":item["shape"],"module_type":item["module_type"],"enters_original_compute_box_int8":candidate,"supports_constraint":candidate,"exclusion_reason":item["exclusion_reason"],"verification":SERVER_VALIDATION_REQUIRED}
        targets.append(target)
        if unknown_language: unknown.append(item["name"])
    report={"model_id":a.model_id,"targets":targets,"unknown_language_parameters":unknown,"status":"failed_unknown_language_parameters" if unknown else "candidate_audit_complete","server_validation":SERVER_VALIDATION_REQUIRED}; Path(a.output).parent.mkdir(parents=True,exist_ok=True); Path(a.output).write_text(json.dumps(report,indent=2),encoding="utf-8")
    if unknown: raise SystemExit(2)
if __name__ == "__main__": main()
