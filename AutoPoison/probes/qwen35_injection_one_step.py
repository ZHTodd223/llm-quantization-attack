"""Server-only wrapper for one unmodified AutoPoison injection step."""
import argparse, subprocess, sys
from pathlib import Path
def main():
 p=argparse.ArgumentParser(); p.add_argument("--model-path",required=True); p.add_argument("--data-path",required=True); p.add_argument("--poison-data-path",required=True); p.add_argument("--output-dir",required=True); p.add_argument("--p-type",default="inject"); p.add_argument("--seed",type=int,default=0); a=p.parse_args(); Path(a.output_dir).mkdir(parents=True,exist_ok=True)
 cmd=[sys.executable,"AutoPoison/main.py","--attack_step","injection","--model_name_key","qwen3.5-4b-base","--model_name_or_path",a.model_path,"--data_path",a.data_path,"--p_data_path",a.poison_data_path,"--p_type",a.p_type,"--p_seed",str(a.seed),"--p_n_sample","4","--output_dir",a.output_dir,"--max_steps","1","--per_device_train_batch_size","1","--gradient_accumulation_steps","1","--bf16","True","--save_strategy","steps","--save_steps","1","--save_total_limit","1","--report_to","none","--train_target_all"]
 raise SystemExit(subprocess.run(cmd).returncode)
if __name__ == "__main__": main()
