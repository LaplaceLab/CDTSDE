import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models


class AdvancedPerceptualLoss(nn.Module):
                                    
    
    def __init__(self, layers=None, weights=None):
        super().__init__()
        if layers is None:
                                          
            layers = ['relu1_2', 'relu2_2', 'relu3_3', 'relu4_3']
        if weights is None:
                                  
            weights = [1.0, 0.8, 0.6, 0.4]
            
        self.layers = layers
        self.weights = weights
        
                    
        vgg = models.vgg16(pretrained=True).features
        self.feature_extractors = nn.ModuleDict()
        
                  
        layer_map = {
            'relu1_2': 4,            
            'relu2_2': 9,             
            'relu3_3': 16,             
            'relu4_3': 23              
        }
        
        for layer_name in self.layers:
            layer_idx = layer_map[layer_name]
            self.feature_extractors[layer_name] = nn.Sequential(*list(vgg.children())[:layer_idx+1])
            
                 
        for extractor in self.feature_extractors.values():
            for param in extractor.parameters():
                param.requires_grad = False
            extractor.eval()
    
    def forward(self, pred_latent, target_latent, decode_fn):
        try:
                      
            pred_rgb = decode_fn(pred_latent)                    
            target_rgb = decode_fn(target_latent)                
            
                                   
            pred_rgb = (pred_rgb + 1.0) / 2.0
            target_rgb = (target_rgb + 1.0) / 2.0
            
                      
            total_loss = 0.0
            for layer_name, weight in zip(self.layers, self.weights):
                extractor = self.feature_extractors[layer_name]
                
                pred_feat = extractor(pred_rgb)
                target_feat = extractor(target_rgb)
                
                        
                feat_loss = F.mse_loss(pred_feat, target_feat)
                total_loss += weight * feat_loss
                
            return total_loss
            
        except Exception as e:
            print(f"Perceptual loss fell back to latent-space MSE: {e}")
            return F.mse_loss(pred_latent, target_latent)


class SimplePerceptualLoss(nn.Module):
                                
    
    def __init__(self, loss_type='mse'):
        super().__init__()
        self.loss_type = loss_type
        
    def forward(self, pred_latent, target_latent):
        if self.loss_type == 'mse':
            return F.mse_loss(pred_latent, target_latent)
        elif self.loss_type == 'l1':
            return F.l1_loss(pred_latent, target_latent)
        else:
                       
            return F.mse_loss(pred_latent, target_latent)


class EdgeAwareLoss(nn.Module):
                           
    
    def __init__(self):
        super().__init__()
                       
        self.register_buffer('sobel_x', torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=torch.float32).view(1, 1, 3, 3))
        self.register_buffer('sobel_y', torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=torch.float32).view(1, 1, 3, 3))
        
    def forward(self, pred, target):
        edge_loss = 0.0
        for i in range(pred.size(1)):
            pred_ch = pred[:, i:i+1, :, :]
            target_ch = target[:, i:i+1, :, :]
            
                  
            pred_edge_x = F.conv2d(pred_ch, self.sobel_x, padding=1)
            pred_edge_y = F.conv2d(pred_ch, self.sobel_y, padding=1)
            pred_edge = torch.sqrt(pred_edge_x**2 + pred_edge_y**2 + 1e-6)
            
            target_edge_x = F.conv2d(target_ch, self.sobel_x, padding=1)
            target_edge_y = F.conv2d(target_ch, self.sobel_y, padding=1)
            target_edge = torch.sqrt(target_edge_x**2 + target_edge_y**2 + 1e-6)
            
                  
            edge_loss += F.l1_loss(pred_edge, target_edge)
            
        return edge_loss / pred.size(1)
