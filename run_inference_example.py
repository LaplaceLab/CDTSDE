                      
import os
import glob
from pathlib import Path

def find_latest_checkpoint(logs_dir: str = "logs") -> str:
                    
    ckpt_pattern = os.path.join(logs_dir, "**", "*.ckpt")
    ckpt_files = glob.glob(ckpt_pattern, recursive=True)
    
    if not ckpt_files:
        print(f"No checkpoint files found in {logs_dir}")
        return None
    
                   
    latest_ckpt = max(ckpt_files, key=os.path.getmtime)
    print(f"Latest checkpoint: {latest_ckpt}")
    return latest_ckpt

def run_inference_example():
                
    
                   
    checkpoint_path = find_latest_checkpoint("logs")
    if checkpoint_path is None:
        print("Please train first to generate a checkpoint")
        print("Run: python train.py")
        return
    
               
    model_config = "configs/model/cldm_v21_dynamic.yaml"
    data_config = "configs/dataset/paired_val.yaml"
    output_dir = "results/inference"
    
              
    if not os.path.exists(model_config):
        print(f"Model config not found: {model_config}")
        return
    
    if not os.path.exists(data_config):
        print(f"Dataset config not found: {data_config}")
        return
    
               
    inference_cmd = f"""python inference.py \\
    --ckpt "{checkpoint_path}" \\
    --model_config "{model_config}" \\
    --data_config "{data_config}" \\
    --output "{output_dir}" \\
    --steps 50 \\
    --batch_size 2 \\
    --device cuda \\
    --max_samples 10"""
    
    print("=" * 60)
    print("Inference command:")
    print(inference_cmd)
    print("=" * 60)
    
               
    print("\nStarting inference...")
    os.system(inference_cmd)
    
    print(f"\nInference complete. Results saved to: {output_dir}")
    print(f"- Generated: {output_dir}/generated/")
    print(f"- Source:    {output_dir}/source/")
    print(f"- Target:    {output_dir}/target/")

if __name__ == "__main__":
    print("DynamicDomainShift inference example")
    print("=" * 40)
    run_inference_example()
