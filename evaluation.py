                      
import os
import json
import csv
import time
from argparse import ArgumentParser, Namespace
from typing import Dict, List, Tuple, Optional, Any
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
import cv2
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import mean_squared_error as mse
import lpips
from torchvision import transforms
from torchvision.models import inception_v3
from scipy import linalg
from tqdm import tqdm

                 
try:
    import pyiqa
    PYIQA_AVAILABLE = True
except ImportError:
    PYIQA_AVAILABLE = False
    print("[WARNING] pyiqa not available; NIQE and other no-reference metrics will be skipped")


class ImageQualityEvaluator:
                 
    
    def __init__(self, device: str = "cuda"):
        self.device = device
        self.lpips_model = None
        self.inception_model = None
        self.niqe_model = None
        self._init_models()
    
    def _init_models(self):
                     
        try:
                     
            self.lpips_model = lpips.LPIPS(net='alex').to(self.device)
            print("[INFO] LPIPS model loaded")
        except Exception as e:
            print(f"[WARNING] LPIPS model load failed: {e}")
        
        try:
                                
            self.inception_model = inception_v3(pretrained=True, transform_input=False).to(self.device)
            self.inception_model.eval()
            print("[INFO] Inception model loaded")
        except Exception as e:
            print(f"[WARNING] Inception model load failed: {e}")
        
        if PYIQA_AVAILABLE:
            try:
                        
                self.niqe_model = pyiqa.create_metric('niqe').to(self.device)
                print("[INFO] NIQE model loaded")
            except Exception as e:
                print(f"[WARNING] NIQE model load failed: {e}")

    def load_image(self, img_path: str) -> np.ndarray:
                             
        img = Image.open(img_path).convert('RGB')
        return np.array(img)
    
    def to_tensor(self, img: np.ndarray) -> torch.Tensor:
                                
        if img.dtype == np.uint8:
            img = img.astype(np.float32) / 255.0
                    
        img_tensor = torch.from_numpy(img.transpose(2, 0, 1)).unsqueeze(0)
        return img_tensor.to(self.device)
    
    def calculate_ssim(self, img1: np.ndarray, img2: np.ndarray) -> float:
                    
        if img1.dtype == np.uint8:
            img1 = img1.astype(np.float32) / 255.0
        if img2.dtype == np.uint8:
            img2 = img2.astype(np.float32) / 255.0
        
        return ssim(img1, img2, multichannel=True, channel_axis=2, data_range=1.0)
    
    def calculate_psnr(self, img1: np.ndarray, img2: np.ndarray) -> float:
                    
        if img1.dtype == np.uint8:
            img1 = img1.astype(np.float32) / 255.0
        if img2.dtype == np.uint8:
            img2 = img2.astype(np.float32) / 255.0
        
        return psnr(img1, img2, data_range=1.0)
    
    def calculate_mse(self, img1: np.ndarray, img2: np.ndarray) -> float:
                   
        if img1.dtype == np.uint8:
            img1 = img1.astype(np.float32) / 255.0
        if img2.dtype == np.uint8:
            img2 = img2.astype(np.float32) / 255.0
        
        return mse(img1, img2)
    
    def calculate_mae(self, img1: np.ndarray, img2: np.ndarray) -> float:
                            
        if img1.dtype == np.uint8:
            img1 = img1.astype(np.float32) / 255.0
        if img2.dtype == np.uint8:
            img2 = img2.astype(np.float32) / 255.0
        
        return np.mean(np.abs(img1 - img2))
    
    def calculate_lpips(self, img1: np.ndarray, img2: np.ndarray) -> float:
                         
        if self.lpips_model is None:
            return -1.0
        
        try:
            tensor1 = self.to_tensor(img1)
            tensor2 = self.to_tensor(img2)
            
                                 
            tensor1 = tensor1 * 2.0 - 1.0
            tensor2 = tensor2 * 2.0 - 1.0
            
            with torch.no_grad():
                lpips_score = self.lpips_model(tensor1, tensor2)
            
            return lpips_score.item()
        except Exception as e:
            print(f"[WARNING] LPIPS computation failed: {e}")
            return -1.0
    
    def calculate_niqe(self, img: np.ndarray) -> float:
                             
        if self.niqe_model is None:
            return -1.0
        
        try:
            tensor = self.to_tensor(img)
            with torch.no_grad():
                niqe_score = self.niqe_model(tensor)
            return niqe_score.item()
        except Exception as e:
            print(f"[WARNING] NIQE computation failed: {e}")
            return -1.0
    
    def get_inception_features(self, img: np.ndarray) -> np.ndarray:
                                  
        if self.inception_model is None:
            return None
        
        try:
                   
            transform = transforms.Compose([
                transforms.ToPILImage(),
                transforms.Resize((299, 299)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                                   std=[0.229, 0.224, 0.225])
            ])
            
            tensor = transform(img).unsqueeze(0).to(self.device)
            
            with torch.no_grad():
                features = self.inception_model(tensor)
                
            return features.cpu().numpy()
        except Exception as e:
            print(f"[WARNING] Inception feature extraction failed: {e}")
            return None
    
    def calculate_fid(self, real_features: List[np.ndarray], 
                     fake_features: List[np.ndarray]) -> float:
                     
        try:
            real_features = np.concatenate(real_features, axis=0)
            fake_features = np.concatenate(fake_features, axis=0)
            
                      
            mu1, sigma1 = real_features.mean(axis=0), np.cov(real_features, rowvar=False)
            mu2, sigma2 = fake_features.mean(axis=0), np.cov(fake_features, rowvar=False)
            
                   
            diff = mu1 - mu2
            covmean, _ = linalg.sqrtm(sigma1.dot(sigma2), disp=False)
            
            if np.iscomplexobj(covmean):
                covmean = covmean.real
                
            fid = diff.dot(diff) + np.trace(sigma1 + sigma2 - 2 * covmean)
            return fid
        except Exception as e:
            print(f"[WARNING] FID computation failed: {e}")
            return -1.0


def find_result_directories(base_dir: str, recursive: bool = False) -> List[str]:
                                                
    result_dirs = []
    
    if recursive:
              
        for root, dirs, files in os.walk(base_dir):
            has_all_folders = all(
                os.path.exists(os.path.join(root, folder)) 
                for folder in ['generated', 'source', 'target']
            )
            if has_all_folders:
                result_dirs.append(root)
    else:
                 
        has_all_folders = all(
            os.path.exists(os.path.join(base_dir, folder))
            for folder in ['generated', 'source', 'target']
        )
        if has_all_folders:
            result_dirs.append(base_dir)
    
    return result_dirs


def get_image_pairs(results_dir: str) -> List[Tuple[str, str, str]]:
                                             
    generated_dir = os.path.join(results_dir, "generated")
    source_dir = os.path.join(results_dir, "source")
    target_dir = os.path.join(results_dir, "target")
    
              
    generated_files = sorted([f for f in os.listdir(generated_dir) 
                            if f.lower().endswith(('.png', '.jpg', '.jpeg'))])
    
    image_pairs = []
    for gen_file in generated_files:
        gen_path = os.path.join(generated_dir, gen_file)
        src_path = os.path.join(source_dir, gen_file)
        tgt_path = os.path.join(target_dir, gen_file)
        
                                  
        if os.path.exists(src_path) and os.path.exists(tgt_path):
            image_pairs.append((gen_path, src_path, tgt_path))
        else:
            print(f"[WARNING] Missing pair for file: {gen_file}")
    
    return image_pairs


def evaluate_directory(results_dir: str, evaluator: ImageQualityEvaluator,
                      metrics: List[str]) -> Dict[str, Any]:
                  
    print(f"\n[INFO] Evaluating directory: {results_dir}")
    
           
    image_pairs = get_image_pairs(results_dir)
    if not image_pairs:
        print(f"[ERROR] No valid image pairs found in {results_dir}")
        return {}
    
    print(f"[INFO] Found {len(image_pairs)} image pairs")
    
             
    metric_scores = {metric: [] for metric in metrics}
    real_features = []
    fake_features = []
    
             
    for i, (gen_path, src_path, tgt_path) in enumerate(tqdm(image_pairs, desc="Evaluating pairs")):
        try:
                  
            generated = evaluator.load_image(gen_path)
            source = evaluator.load_image(src_path)
            target = evaluator.load_image(tgt_path)
            
                      
            if generated.shape != target.shape:
                print(f"[WARNING] Size mismatch {os.path.basename(gen_path)}: "
                      f"generated {generated.shape} vs target {target.shape}")
                                 
                generated = cv2.resize(generated, (target.shape[1], target.shape[0]))
            
                    
            if 'ssim' in metrics:
                ssim_score = evaluator.calculate_ssim(generated, target)
                metric_scores['ssim'].append(ssim_score)
            
            if 'psnr' in metrics:
                psnr_score = evaluator.calculate_psnr(generated, target)
                metric_scores['psnr'].append(psnr_score)
            
            if 'mse' in metrics:
                mse_score = evaluator.calculate_mse(generated, target)
                metric_scores['mse'].append(mse_score)
            
            if 'mae' in metrics:
                mae_score = evaluator.calculate_mae(generated, target)
                metric_scores['mae'].append(mae_score)
            
            if 'lpips' in metrics:
                lpips_score = evaluator.calculate_lpips(generated, target)
                if lpips_score >= 0:
                    metric_scores['lpips'].append(lpips_score)
            
            if 'niqe' in metrics:
                niqe_score = evaluator.calculate_niqe(generated)
                if niqe_score >= 0:
                    metric_scores['niqe'].append(niqe_score)
            
                     
            if 'fid' in metrics:
                real_feat = evaluator.get_inception_features(target)
                fake_feat = evaluator.get_inception_features(generated)
                if real_feat is not None and fake_feat is not None:
                    real_features.append(real_feat)
                    fake_features.append(fake_feat)
                    
        except Exception as e:
            print(f"[ERROR] Failed on pair {i}: {e}")
            continue
    
           
    if 'fid' in metrics and len(real_features) > 0:
        fid_score = evaluator.calculate_fid(real_features, fake_features)
        metric_scores['fid'] = [fid_score]
    
            
    results = {}
    for metric, scores in metric_scores.items():
        if len(scores) > 0:
            results[metric] = {
                'mean': float(np.mean(scores)),
                'std': float(np.std(scores)),
                'min': float(np.min(scores)),
                'max': float(np.max(scores)),
                'count': len(scores)
            }
        else:
            results[metric] = {
                'mean': -1.0,
                'std': 0.0,
                'min': -1.0,
                'max': -1.0,
                'count': 0
            }
    
    results['num_images'] = len(image_pairs)
    results['directory'] = results_dir
    
    return results


def print_results(results: Dict[str, Any]):
                
    print(f"\n{'='*60}")
    print(f"Results - {results['directory']}")
    print(f"{'='*60}")
    print(f"Images: {results['num_images']}")
    print(f"{'-'*60}")
    
    metric_names = {
        'ssim': 'SSIM (structural similarity)',
        'psnr': 'PSNR (peak signal-to-noise)',
        'mse': 'MSE (mean squared error)',
        'mae': 'MAE (mean absolute error)',
        'lpips': 'LPIPS (perceptual distance)',
        'niqe': 'NIQE (no-reference quality)',
        'fid': 'FID (Fréchet distance)'
    }
    
    for metric, data in results.items():
        if metric in ['num_images', 'directory']:
            continue
            
        if data['count'] > 0:
            name = metric_names.get(metric, metric.upper())
            print(f"{name:20s}: {data['mean']:.4f} ± {data['std']:.4f} "
                  f"(range: {data['min']:.4f} - {data['max']:.4f})")
        else:
            print(f"{metric.upper():20s}: failed")


def save_results(all_results: List[Dict[str, Any]], output_path: str):
                  
              
    json_path = output_path.replace('.csv', '.json') if output_path.endswith('.csv') else output_path
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)
    print(f"[INFO] Detailed results saved to: {json_path}")
    
               
    csv_path = output_path.replace('.json', '.csv') if output_path.endswith('.json') else output_path + '.csv'
    with open(csv_path, 'w', newline='', encoding='utf-8') as f:
        if not all_results:
            return
            
                  
        metrics = set()
        for result in all_results:
            metrics.update([k for k in result.keys() if k not in ['num_images', 'directory']])
        metrics = sorted(list(metrics))
        
                 
        fieldnames = ['directory', 'num_images']
        for metric in metrics:
            fieldnames.extend([f'{metric}_mean', f'{metric}_std', f'{metric}_min', f'{metric}_max'])
        
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        
              
        for result in all_results:
            row = {
                'directory': result['directory'],
                'num_images': result['num_images']
            }
            for metric in metrics:
                if metric in result:
                    row[f'{metric}_mean'] = result[metric]['mean']
                    row[f'{metric}_std'] = result[metric]['std']
                    row[f'{metric}_min'] = result[metric]['min']
                    row[f'{metric}_max'] = result[metric]['max']
                else:
                    row[f'{metric}_mean'] = -1.0
                    row[f'{metric}_std'] = 0.0
                    row[f'{metric}_min'] = -1.0
                    row[f'{metric}_max'] = -1.0
            writer.writerow(row)
    
    print(f"[INFO] CSV summary saved to: {csv_path}")


def parse_args() -> Namespace:
                 
    parser = ArgumentParser(description="DoSSR image quality evaluator")
    
    parser.add_argument("--results_dir", type=str, required=True,
                       help="Results directory")
    parser.add_argument("--recursive", action="store_true",
                       help="Recursively search subdirectories")
    parser.add_argument("--metrics", nargs='+', 
                       default=['ssim', 'psnr', 'mse', 'mae', 'lpips', 'niqe', 'fid'],
                       choices=['ssim', 'psnr', 'mse', 'mae', 'lpips', 'niqe', 'fid'],
                       help="Metrics to compute")
    parser.add_argument("--output", type=str, default=None,
                       help="Output file path")
    parser.add_argument("--device", type=str, default="cuda",
                       choices=["cuda", "cpu"],
                       help="Compute device")
    
    return parser.parse_args()


def main():
             
    args = parse_args()
    
          
    device = args.device
    if device == "cuda" and not torch.cuda.is_available():
        print("[WARNING] CUDA not available, using CPU")
        device = "cpu"
    
    print(f"[INFO] Using device: {device}")
    print(f"[INFO] Metrics: {args.metrics}")
    
            
    result_directories = find_result_directories(args.results_dir, args.recursive)
    if not result_directories:
        print(f"[ERROR] No valid result directories found in {args.results_dir}")
        print("[INFO] A valid directory must contain 'generated', 'source', 'target'")
        return
    
    print(f"[INFO] Found {len(result_directories)} result directories")
    
            
    print("[INFO] Initializing evaluator...")
    evaluator = ImageQualityEvaluator(device=device)
    
            
    all_results = []
    start_time = time.time()
    
    for results_dir in result_directories:
        results = evaluate_directory(results_dir, evaluator, args.metrics)
        if results:
            all_results.append(results)
            print_results(results)
    
    total_time = time.time() - start_time
    
          
    print(f"\n{'='*60}")
    print("Evaluation complete!")
    print(f"{'='*60}")
    print(f"Total result directories: {len(all_results)}")
    print(f"Elapsed time: {total_time:.2f}s")
    
    if len(all_results) > 1:
                     
        print(f"\n{'='*60}")
        print("Average results across directories")
        print(f"{'='*60}")
        
        avg_results = {}
        for metric in args.metrics:
            means = [r[metric]['mean'] for r in all_results if metric in r and r[metric]['count'] > 0]
            if means:
                avg_results[metric] = {
                    'mean': float(np.mean(means)),
                    'std': float(np.std(means)),
                    'min': float(np.min(means)),
                    'max': float(np.max(means)),
                    'count': len(means)
                }
        
        for metric, data in avg_results.items():
            metric_names = {
                'ssim': 'SSIM (structural similarity)',
                'psnr': 'PSNR (peak signal-to-noise)',
                'mse': 'MSE (mean squared error)',
                'mae': 'MAE (mean absolute error)',
                'lpips': 'LPIPS (perceptual distance)',
                'niqe': 'NIQE (no-reference quality)',
                'fid': 'FID (Fréchet distance)'
            }
            name = metric_names.get(metric, metric.upper())
            print(f"{name:20s}: {data['mean']:.4f} ± {data['std']:.4f}")
    
          
    if args.output:
        save_results(all_results, args.output)
    else:
                   
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        default_output = os.path.join(args.results_dir, f"evaluation_report_{timestamp}.json")
        save_results(all_results, default_output)


if __name__ == "__main__":
    main()
