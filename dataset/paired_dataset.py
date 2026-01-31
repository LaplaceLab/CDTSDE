import os
import torch
import cv2
import numpy as np
from torch.utils.data import Dataset


class PairedDataset(Dataset):


    def __init__(self, pair_list_path, out_size=128, crop_type='center'):





        self.pair_list_path = pair_list_path
        self.out_size = out_size
        self.crop_type = crop_type
        
                        
        self.pairs = []
        with open(pair_list_path, 'r') as f:
            for line in f:
                line = line.strip()
                if line and ',' in line:
                    lq_path, hq_path = line.split(',')
                                                             
                    hq_path = hq_path.replace('./', '')
                    self.pairs.append((lq_path.strip(), hq_path.strip()))
        
        print(f"Loaded {len(self.pairs)} image pairs from {pair_list_path}")
    
    def __len__(self):
        return len(self.pairs)
    
    def __getitem__(self, idx):
        lq_path, hq_path = self.pairs[idx]
        
                     
        lq_img = cv2.imread(lq_path, cv2.IMREAD_COLOR)
        hq_img = cv2.imread(hq_path, cv2.IMREAD_COLOR)
        
        if lq_img is None:
            raise ValueError(f"Cannot load LQ image: {lq_path}")
        if hq_img is None:
            raise ValueError(f"Cannot load HQ image: {hq_path}")
        
                            
        lq_img = cv2.cvtColor(lq_img, cv2.COLOR_BGR2RGB)
        hq_img = cv2.cvtColor(hq_img, cv2.COLOR_BGR2RGB)
        
                                    
        lq_img = self._process_image(lq_img, self.out_size)
        hq_img = self._process_image(hq_img, self.out_size)
        
                                         
                             
                                                                               
        lq_tensor = torch.from_numpy(lq_img.astype(np.float32) / 255.0)
        hq_tensor = torch.from_numpy(hq_img.astype(np.float32) / 255.0)
        
                                                                  
        lq_tensor = lq_tensor.permute(2, 0, 1)                          
        hq_tensor = hq_tensor.permute(2, 0, 1)                          
        
        return {
            'hq': hq_tensor,
            'lq': lq_tensor,
            'txt': ''                             
        }
    
    def _process_image(self, img, target_size):


        h, w = img.shape[:2]
        
        if self.crop_type == 'center':
                         
            min_dim = min(h, w)
            top = (h - min_dim) // 2
            left = (w - min_dim) // 2
            img = img[top:top+min_dim, left:left+min_dim]
        
                               
        img = cv2.resize(img, (target_size, target_size), interpolation=cv2.INTER_LINEAR)
        
        return img
