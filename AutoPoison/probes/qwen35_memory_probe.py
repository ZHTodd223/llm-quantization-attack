"""Server-only, one-stage-at-a-time CUDA memory probe for Qwen3.5.

This script intentionally has no device-map, offload, LoRA, or model fallback.
"""
import argparse, json, time, traceback
from pathlib import Path

STAGES=("load","forward","backward","optimizer-init","optimizer-step","constraint-build","projection")
def parser():
    p=argparse.ArgumentParser(); p.add_argument("--model-id",required=True); p.add_argument("--output",required=True); p.add_argument("--stage",required=True,choices=STAGES); p.add_argument("--batch-size",type=int,default=1); p.add_argument("--sequence-length",type=int,default=64); p.add_argument("--dtype",default="bfloat16",choices=["bfloat16","float16"]); return p
def main():
    import torch
    from AutoPoison.model_adapters.qwen35 import load_causal_lm, load_tokenizer
    a=parser().parse_args(); started=time.monotonic(); report={"stage":a.stage,"success":False,"exception":None,"cuda_device":None,"total_memory":None,"peak_allocated":None,"peak_reserved":None,"runtime_seconds":None,"model_parameters":None,"trainable_parameters":None,"optimizer_type":None}
    try:
        if not torch.cuda.is_available(): raise RuntimeError("CUDA is required; this probe never falls back to CPU")
        device=torch.device("cuda:0"); report["cuda_device"]=torch.cuda.get_device_name(device); report["total_memory"]=torch.cuda.get_device_properties(device).total_memory; torch.cuda.reset_peak_memory_stats(device)
        model=load_causal_lm(a.model_id,torch_dtype=getattr(torch,a.dtype),device_map=None,trust_remote_code=True).to(device); model.config.use_cache=False; model.gradient_checkpointing_enable(); report["model_parameters"]=sum(p.numel() for p in model.parameters()); report["trainable_parameters"]=sum(p.numel() for p in model.parameters() if p.requires_grad)
        if a.stage != "load":
            tokenizer=load_tokenizer(a.model_id,use_fast=False); token=tokenizer("x "*a.sequence_length,return_tensors="pt",truncation=True,max_length=a.sequence_length); ids=token.input_ids.to(device); out=model(input_ids=ids,labels=ids)
        if a.stage in ("backward","optimizer-step","constraint-build","projection"): out.loss.backward()
        if a.stage in ("optimizer-init","optimizer-step"): optimizer=torch.optim.AdamW(model.parameters()); report["optimizer_type"]=type(optimizer).__name__
        if a.stage == "optimizer-step": optimizer.step()
        if a.stage in ("constraint-build","projection"):
            from AutoPoison.quant_specific.pgd import QuantizeArguments, compute_box, PGDCallback
            from types import SimpleNamespace
            box,_=compute_box(model,SimpleNamespace(model_name_or_path=a.model_id),QuantizeArguments(quantize_method="int8"),SimpleNamespace(unfreeze_block=False,unfreeze_maxmin=False,freeze_sensitive_iters=0,thresh_type=None,interval_type="exact"))
            if a.stage == "projection": PGDCallback(box).on_step_end(None,None,None,model=model)
        report["success"]=True
    except Exception: report["exception"]=traceback.format_exc()
    finally:
        if torch.cuda.is_available(): report["peak_allocated"]=torch.cuda.max_memory_allocated(); report["peak_reserved"]=torch.cuda.max_memory_reserved()
        report["runtime_seconds"]=time.monotonic()-started; Path(a.output).parent.mkdir(parents=True,exist_ok=True); Path(a.output).write_text(json.dumps(report,indent=2),encoding="utf-8")
    if not report["success"]: raise SystemExit(1)
if __name__ == "__main__": main()
