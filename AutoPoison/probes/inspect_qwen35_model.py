"""Server-only structure inspection; loads a real checkpoint when invoked."""
import argparse, json
from pathlib import Path
import torch
from AutoPoison.model_adapters.qwen35 import load_causal_lm, scan_model

def main():
    p=argparse.ArgumentParser(); p.add_argument("--model-id", required=True); p.add_argument("--output", required=True); p.add_argument("--dtype", default="bfloat16", choices=["bfloat16","float16","float32"]); a=p.parse_args()
    model=load_causal_lm(a.model_id, torch_dtype=getattr(torch, a.dtype), device_map=None, trust_remote_code=True)
    report=scan_model(model); Path(a.output).parent.mkdir(parents=True, exist_ok=True); Path(a.output).write_text(json.dumps(report, indent=2), encoding="utf-8")
if __name__ == "__main__": main()
