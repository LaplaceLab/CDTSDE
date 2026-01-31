                      
import os
import time
from argparse import ArgumentParser, Namespace
from typing import List, Optional, Dict, Any

import numpy as np
import torch
import einops
from PIL import Image
from omegaconf import OmegaConf
from torch.utils.data import DataLoader

import pytorch_lightning as pl
from utils.common import instantiate_from_config, load_state_dict
from utils.file import list_image_files, get_file_name_parts
from utils.image import auto_resize, pad
from ldm.xformers_state import disable_xformers
from model.cldm import ControlLDM


def check_device(device_str: str) -> str:
                    
    if device_str == "auto":
        device_str = "cuda" if torch.cuda.is_available() else "cpu"
    if device_str == "cuda" and not torch.cuda.is_available():
        print("CUDA is not available, falling back to CPU")
        device_str = "cpu"
    return device_str


@torch.no_grad()
def inference_with_dataloader(
    model: ControlLDM,
    batch: Dict[str, Any],
    sample_steps: int = 50,
    device: str = "cuda"
) -> Dict[str, torch.Tensor]:
    print(f"[INFO] Processing batch: {batch['jpg'].shape[0]} images")
    print(f"[INFO] Target (jpg) range: [{batch['jpg'].min():.3f}, {batch['jpg'].max():.3f}]")
    print(f"[INFO] Source (hint) range: [{batch['hint'].min():.3f}, {batch['hint'].max():.3f}]")
    
    try:
                                 
        print("[INFO] Calling get_input...")
        z, c = model.get_input(batch, model.first_stage_key, bs=batch['jpg'].shape[0])
        
        print(f"[INFO] Latent z shape: {z.shape}")
        print(f"[INFO] Condition keys: {list(c.keys())}")
        
                                  
        print(f"[INFO] Starting sampling, steps: {sample_steps}")
        
                
        c_lq = c["lq"][0]
        c_latent = c["c_latent"][0] 
        c_cat, c_crossattn = c["c_concat"][0], c["c_crossattn"][0]
        
        print(f"[INFO] c_cat (control) shape: {c_cat.shape}")
        print(f"[INFO] c_latent shape: {c_latent.shape}")
        
                   
        cond_dict = {
            "c_concat": [c_cat],
            "c_crossattn": [c_crossattn], 
            "c_latent": [c_latent]
        }
        
                                      
        samples = model.sample_log(cond=cond_dict, steps=sample_steps)
        
        print(f"[INFO] Sampling complete, output shape: {samples.shape}")
        print(f"[INFO] Sample range: [{samples.min():.3f}, {samples.max():.3f}]")
        
        return {
            'generated': samples,                                            
            'source': batch['hint'],                                          
            'target': (batch['jpg'] + 1) / 2                                
        }
        
    except Exception as e:
        print(f"[ERROR] Inference failed: {e}")
        import traceback
        traceback.print_exc()
        raise


def save_images(results: Dict[str, torch.Tensor], output_dir: str, batch_idx: int, start_idx: int = 0):
    generated_dir = os.path.join(output_dir, "generated")
    source_dir = os.path.join(output_dir, "source")
    target_dir = os.path.join(output_dir, "target")
    
    os.makedirs(generated_dir, exist_ok=True)
    os.makedirs(source_dir, exist_ok=True)
    os.makedirs(target_dir, exist_ok=True)
    
               
    batch_size = results['generated'].shape[0]
    
    for i in range(batch_size):
        img_idx = start_idx + i
        
                                                     
        generated = results['generated'][i].detach().cpu().numpy()
        generated = generated.transpose(1, 2, 0)              
        generated = (generated * 255).clip(0, 255).astype(np.uint8)
        
                                                    
        source = results['source'][i].detach().cpu().numpy()
        source = (source * 255).clip(0, 255).astype(np.uint8)
        
                                                  
        target = results['target'][i].detach().cpu().numpy()
        target = (target * 255).clip(0, 255).astype(np.uint8)
        
              
        Image.fromarray(generated).save(os.path.join(generated_dir, f"{img_idx:04d}.png"))
        Image.fromarray(source).save(os.path.join(source_dir, f"{img_idx:04d}.png"))
        Image.fromarray(target).save(os.path.join(target_dir, f"{img_idx:04d}.png"))
        
        print(f"[INFO] Saved image {img_idx:04d}: generated, source, target")


def parse_args() -> Namespace:
                 
    parser = ArgumentParser()
    
          
    parser.add_argument("--ckpt", type=str, required=True,
                       help="Path to model checkpoint")
    parser.add_argument("--model_config", type=str, 
                       default="configs/model/cldm_v21_dynamic.yaml",
                       help="Path to model config")
    parser.add_argument("--data_config", type=str,
                       default="configs/dataset/paired_val.yaml", 
                       help="Path to dataset config")
    
          
    parser.add_argument("--output", type=str, 
                       default="results/",
                       help="Output directory")
    parser.add_argument("--steps", type=int, default=50,
                       help="Number of sampling steps")
    parser.add_argument("--batch_size", type=int, default=None,
                       help="Batch size for inference (defaults to config)")
    parser.add_argument("--device", type=str, default="auto",
                       choices=["auto", "cuda", "cpu"],
                       help="Compute device")
    parser.add_argument("--seed", type=int, default=42,
                       help="Random seed")
    parser.add_argument("--max_samples", type=int, default=None,
                       help="Max samples to process (for testing)")
    
    return parser.parse_args()


def main() -> None:
                              
    args = parse_args()
    
            
    pl.seed_everything(args.seed)
    
          
    args.device = check_device(args.device)
    print(f"[INFO] Using device: {args.device}")
    
              
    if not os.path.exists(args.model_config):
        print(f"[ERROR] Model config not found: {args.model_config}")
        return
    
    if not os.path.exists(args.data_config):
        print(f"[ERROR] Dataset config not found: {args.data_config}")
        return
        
               
    if not os.path.exists(args.ckpt):
        print(f"[ERROR] Checkpoint not found: {args.ckpt}")
        return
    
    print(f"[INFO] Loading model config: {args.model_config}")
    model_config = OmegaConf.load(args.model_config)
    
    print(f"[INFO] Loading dataset config: {args.data_config}")
    data_config = OmegaConf.load(args.data_config)
    
             
    print("[INFO] Instantiating model...")
    model: ControlLDM = instantiate_from_config(model_config)
    
    print(f"[INFO] Loading checkpoint: {args.ckpt}")
    state_dict = torch.load(args.ckpt, map_location="cuda")
    load_state_dict(model, state_dict, strict=True)
    
    model.freeze()
    model.to(args.device)
    model.eval()
    
                   
    print("[INFO] Building dataset...")
    dataset = instantiate_from_config(data_config["dataset"])
    batch_transform = instantiate_from_config(data_config["batch_transform"])
    
    print(f"[INFO] Dataset size: {len(dataset)}")
    
                     
    dataloader_config = data_config["data_loader"].copy()
    if args.batch_size is not None:
        dataloader_config["batch_size"] = args.batch_size
        print(f"[INFO] Using custom batch size: {args.batch_size}")
    else:
        print(f"[INFO] Using batch size from config: {dataloader_config['batch_size']}")
    
                          
    dataloader_config["shuffle"] = False
    dataloader_config["drop_last"] = False
    
    dataloader = DataLoader(dataset, **dataloader_config)
    print(f"[INFO] DataLoader ready, num batches: {len(dataloader)}")
    
               
    os.makedirs(args.output, exist_ok=True)
    print(f"[INFO] Output directory: {args.output}")
    
             
    total_processed = 0
    start_time = time.time()
    
    print("\n[INFO] 🚀 Starting inference...")
    print(f"[INFO] Sampling steps: {args.steps}")
    print(f"[INFO] Expected images: {len(dataset) if args.max_samples is None else min(args.max_samples, len(dataset))}")
    
    for batch_idx, batch in enumerate(dataloader):
        if args.max_samples is not None and total_processed >= args.max_samples:
            print(f"[INFO] Reached max samples ({args.max_samples}), stopping")
            break
            
        print(f"\n[INFO] === Processing batch {batch_idx + 1}/{len(dataloader)} ===")
        
        try:
                                
            print("[INFO] Applying batch transforms...")
            batch_transformed = batch_transform(batch)
            
                   
            for key in batch_transformed:
                if torch.is_tensor(batch_transformed[key]):
                    batch_transformed[key] = batch_transformed[key].to(args.device)
            
            batch_start_time = time.time()
            
                  
            results = inference_with_dataloader(
                model=model,
                batch=batch_transformed,
                sample_steps=args.steps,
                device=args.device
            )
            
            batch_end_time = time.time()
            batch_time = batch_end_time - batch_start_time
            
                  
            save_images(results, args.output, batch_idx, total_processed)
            
            current_batch_size = batch_transformed['jpg'].shape[0]
            total_processed += current_batch_size
            
            print(f"[INFO] ✓ Batch {batch_idx + 1} complete")
            print(f"[INFO] Batch size: {current_batch_size}, time: {batch_time:.2f}s")
            print(f"[INFO] Total processed: {total_processed} images")
            
        except Exception as e:
            print(f"[ERROR] Failed on batch {batch_idx + 1}: {e}")
            import traceback
            traceback.print_exc()
            continue
    
    total_time = time.time() - start_time
    avg_time_per_image = total_time / total_processed if total_processed > 0 else 0
    
    print("\n[INFO] 🎉 Inference complete!")
    print(f"[INFO] Total time: {total_time:.2f}s")
    print(f"[INFO] Images processed: {total_processed}")
    print(f"[INFO] Avg time per image: {avg_time_per_image:.2f}s")
    print("[INFO] Results saved to:")
    print(f"[INFO]   Generated: {os.path.join(args.output, 'generated')}")
    print(f"[INFO]   Source:    {os.path.join(args.output, 'source')}")
    print(f"[INFO]   Target:    {os.path.join(args.output, 'target')}")


if __name__ == "__main__":
    main()
