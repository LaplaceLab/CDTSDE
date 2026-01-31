import torch
import torch.nn as nn
import torch.nn.functional as F
from fastkan import FastKAN as KAN

class LearnableNonlinearLambdaEnhanced(nn.Module):
    def __init__(self, degree=5, channels=4, enable_spatial_variation=True):
        super().__init__()
        self.channels = channels
        self.enable_spatial_variation = enable_spatial_variation
        
                                
        self.coeffs = nn.Parameter(torch.randn(channels, degree + 1) * 0.2)           
        
                               
        for c in range(channels):
            self.coeffs.data[c, 1] = 1.0 + 0.2 * (c - channels//2)             
            self.coeffs.data[c, 0] = 0.0       
                            
            for i in range(2, degree + 1):
                self.coeffs.data[c, i] = 0.15 * torch.randn(1) * (0.7 ** i)           
        
                            
        if enable_spatial_variation:
            self.spatial_net = nn.Sequential(
                nn.Conv2d(1, 8, kernel_size=5, padding=2),         
                nn.GroupNorm(2, 8),
                nn.GELU(), 
                nn.Conv2d(8, 16, kernel_size=3, padding=1),
                nn.GroupNorm(4, 16),
                nn.GELU(),
                nn.Conv2d(16, channels, kernel_size=3, padding=1),
                nn.Tanh()                      
            )
            
                                 
            self.register_buffer('pos_encoding', self._create_position_encoding(64, 64))
        
    def _create_position_encoding(self, H, W):
                              
        y_pos = torch.linspace(-1, 1, H).view(-1, 1).expand(-1, W)
        x_pos = torch.linspace(-1, 1, W).view(1, -1).expand(H, -1)
        pos_enc = torch.stack([
            torch.sin(3.14159 * y_pos), torch.cos(3.14159 * y_pos),
            torch.sin(3.14159 * x_pos), torch.cos(3.14159 * x_pos)
        ], dim=0)             
        return pos_enc.mean(dim=0, keepdim=True)             
        
    def forward(self, lam, target_shape):


        B, C, H, W = target_shape
        
                    
        if lam.dim() == 1:
            lam = lam.view(B, 1, 1, 1)
        elif lam.dim() == 4:
            pass           
        else:
            lam = lam.view(B, 1, 1, 1)
        
        lam_expanded = lam.expand(B, 1, H, W)
        
                              
        if self.enable_spatial_variation and hasattr(self, 'spatial_net'):
                                 
            pos_enc = self.pos_encoding[:, :H, :W].unsqueeze(0).expand(B, -1, -1, -1)
            spatial_input = lam_expanded + 0.2 * pos_enc          
            
                                           
            raw_output = self.spatial_net(spatial_input)              
            
                                        
            normalized_output = torch.sigmoid(raw_output)           
            
            result = lam_expanded * normalized_output + (1 - lam_expanded) * (lam_expanded * normalized_output)

            boundary_term = lam_expanded                       
            modulation_term = lam_expanded * (1 - lam_expanded) * (2 * normalized_output - 1)         
            
            result = boundary_term + modulation_term
            
        else:
                                
            result = torch.zeros(B, C, H, W, device=lam.device, dtype=lam.dtype)
            
            for c in range(C):
                channel_result = torch.zeros_like(lam_expanded)
                for i, coeff in enumerate(self.coeffs[c]):
                    channel_result += coeff * (lam_expanded ** i)
                
                        
                f_0 = self.coeffs[c, 0]
                f_1 = torch.sum(self.coeffs[c])
                
                if abs(f_1 - f_0) > 1e-6:
                    channel_result = (channel_result - f_0) / (f_1 - f_0)
                
                result[:, c:c+1] = channel_result
        
                                             
        result = torch.sigmoid(result * 6 - 3)                                      
        
        return result

