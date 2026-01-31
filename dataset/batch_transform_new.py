


from typing import Any, Dict, List
import torch
import numpy as np
from torch.nn import functional as F


class PairedBatchTransform:

    def __init__(self, **kwargs):

        pass
    
    def __call__(self, batch: Dict[str, Any]) -> Dict[str, Any]:


        hq_batch = batch['hq']                
        lq_batch = batch['lq']                
        txt_batch = batch['txt']             
        
                                   
        hq_batch = hq_batch.permute(0, 2, 3, 1)                                
        lq_batch = lq_batch.permute(0, 2, 3, 1)                                
        
                          
                                     
        jpg = hq_batch * 2.0 - 1.0
        
                                   
        hint = lq_batch
        
                                   
        jpg = jpg.float()
        hint = hint.float()
        
        return {
            'jpg': jpg,                                     
            'hint': hint,                                  
            'txt': txt_batch                            
        }


class IdentityBatchTransform:


    def __call__(self, batch: Any) -> Any:
        return batch
