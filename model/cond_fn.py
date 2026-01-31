from typing import overload, Optional
import torch
from torch.nn import functional as F


class Guidance:

    def __init__(
        self,
        scale: float,
        t_start: int,
        t_stop: int,
        space: str,
        repeat: int
    ) -> "Guidance":

        self.t_start = t_start
        self.t_stop = t_stop
        self.target = None
        self.space = space
        self.repeat = repeat
    
    def load_target(self, target: torch.Tensor) -> torch.Tensor:
        self.target = target

    def __call__(self, target_x0: torch.Tensor, pred_x0: torch.Tensor, t: int) -> Optional[torch.Tensor]:
        if self.t_stop < t and t < self.t_start:
                                                        
                                                          
            pred_x0 = pred_x0.detach().clone()
            target_x0 = target_x0.detach().clone()
            return self.scale * self._forward(target_x0, pred_x0)
        else:
            return None
    
    @overload
    def _forward(self, target_x0: torch.Tensor, pred_x0: torch.Tensor) -> torch.Tensor:
        ...


class MSEGuidance(Guidance):
    
    def __init__(
        self,
        scale: float,
        t_start: int,
        t_stop: int,
        space: str,
        repeat: int
    ) -> "MSEGuidance":
        super().__init__(
            scale, t_start, t_stop, space, repeat
        )
    
    @torch.enable_grad()
    def _forward(self, target_x0: torch.Tensor, pred_x0: torch.Tensor) -> torch.Tensor:
                                    
        pred_x0.requires_grad_(True)
        
                                       
        loss = (pred_x0 - target_x0).pow(2).mean((1, 2, 3)).sum()
        
        print(f"loss = {loss.item()}")
        return -torch.autograd.grad(loss, pred_x0)[0]
