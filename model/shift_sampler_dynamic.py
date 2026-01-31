from typing import Optional, Tuple, Dict, List, Callable
import torch
import numpy as np
from tqdm import tqdm

from ldm.modules.diffusionmodules.util import make_beta_schedule, make_eta_schedule
from utils.image import (
    wavelet_reconstruction, adaptive_instance_normalization
)
from ldm.modules.distributions.distributions import DiagonalGaussianDistribution

                                                                                  
def space_timesteps(num_timesteps, section_counts):
       
    if isinstance(section_counts, str):
        if section_counts.startswith("ddim"):
            desired_count = int(section_counts[len("ddim") :])
            for i in range(1, num_timesteps):
                if len(range(0, num_timesteps, i)) == desired_count:
                    return set(range(0, num_timesteps, i))
            raise ValueError(
                f"cannot create exactly {num_timesteps} steps with an integer stride"
            )
        section_counts = [int(x) for x in section_counts.split(",")]
    size_per = num_timesteps // len(section_counts)
    extra = num_timesteps % len(section_counts)
    start_idx = 0
    all_steps = []
    for i, section_count in enumerate(section_counts):
        size = size_per + (1 if i < extra else 0)
        if size < section_count:
            raise ValueError(
                f"cannot divide section of {size} steps into {section_count}"
            )
        if section_count <= 1:
            frac_stride = 1
        else:
            frac_stride = (size - 1) / (section_count - 1)
        cur_idx = 0.0
        taken_steps = []
        for _ in range(section_count):
            taken_steps.append(start_idx + round(cur_idx))
            cur_idx += frac_stride
        all_steps += taken_steps
        start_idx += size
    return set(all_steps)

                                                                                             
def _extract_into_tensor(arr, timesteps, broadcast_shape):
    try:
                                                                     
        res = torch.from_numpy(arr).to(device=timesteps.device)[timesteps].float()
    except:
                                   
        res = torch.from_numpy(arr.astype(np.float32)).to(device=timesteps.device)[timesteps].float()
        
    while len(res.shape) < len(broadcast_shape):
        res = res[..., None]
    return res.expand(broadcast_shape)



class SpacedSampler:

    def __init__(
        self,
        model: "ControlLDM",
        schedule: str = "linear",
        var_type: str = "fixed_small"
    ) -> None:
        self.model = model
        self.original_num_steps = model.num_timesteps             
        self.schedule = schedule
        self.var_type = var_type

                                                                        
    def make_schedule(self, num_steps: int) -> None:

                                   
        original_betas = make_beta_schedule(
            self.schedule, self.original_num_steps,
            linear_start=self.model.linear_start,
            linear_end=self.model.linear_end
        )
        original_alphas = 1.0 - original_betas
        original_alphas_cumprod = np.cumprod(original_alphas, axis=0)

                                          
        original_etas, t1 = make_eta_schedule(
            "trunc_12", self.original_num_steps, return_t1=True
        )
        used_timesteps = space_timesteps(t1, str(num_steps + 1))

        betas, etas = [], []
        last_alpha_cumprod = 1.0
        for i, alpha_cumprod in enumerate(original_alphas_cumprod):
            if i in used_timesteps:
                betas.append(1 - alpha_cumprod / last_alpha_cumprod)
                last_alpha_cumprod = alpha_cumprod
                etas.append(original_etas[i])

                     
        betas = np.array(betas, dtype=np.float64)
        self.betas = betas

                          
        etas = np.array(etas, dtype=np.float64)                
        device = next(self.model.parameters()).device
        with torch.no_grad():
                                        
                                            
            etas_tensor = torch.from_numpy(etas).to(device).float().view(-1, 1, 1, 1)
            
                                                  
                                    
            dummy_shape = (len(etas), 4, 32, 32)                   
            etas_hat_tensor = self.model.nonlinear_lambda(etas_tensor, dummy_shape)
            
                                       
            etas_hat_t = etas_hat_tensor.mean(dim=(1, 2, 3)).clamp_(0.0, 1.0)
        etas_hat = etas_hat_t.cpu().numpy().astype(np.float64)

                             
        self.etas = etas_hat                                       
        self.one_minus_etas = 1.0 - etas_hat                

                   
        self.timesteps = np.array(sorted(list(used_timesteps)), dtype=np.int32)

        alphas = 1.0 - betas                           
        self.alphas = alphas
        self.alphas_cumprod = np.cumprod(alphas, axis=0)         

        self.sqrt_alphas_cumprod = np.sqrt(self.alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = np.sqrt(1.0 - self.alphas_cumprod)
        self.sqrt_recip_alphas_cumprod = np.sqrt(1.0 / self.alphas_cumprod)
        self.sqrt_recipm1_alphas_cumprod = np.sqrt(1.0 / self.alphas_cumprod - 1)

                            
        self.alphas_multi_one_minus_etas = (
            self.sqrt_alphas_cumprod * self.one_minus_etas
        )

                     
        self.sigmas = np.sqrt(1.0 - self.alphas_cumprod)
        self.lambdas = self.sigmas / self.alphas_multi_one_minus_etas        

                                                                        
    def _predict_xstart_from_eps(
        self, x_t: torch.Tensor, t: torch.Tensor, eps: torch.Tensor
    ) -> torch.Tensor:
        return (
            _extract_into_tensor(self.sqrt_recip_alphas_cumprod, t, x_t.shape) * x_t
            - _extract_into_tensor(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape) * eps
        )

    def predict_noise(
        self,
        x: torch.Tensor,
        t: torch.Tensor,
        cond: Dict[str, torch.Tensor],
        cfg_scale: float,
        uncond: Optional[Dict[str, torch.Tensor]],
    ) -> torch.Tensor:
        if uncond is None or cfg_scale == 1.0:
            model_output = self.model.apply_model(x, t, cond)
        else:
                                      
            model_cond = self.model.apply_model(x, t, cond)
            model_uncond = self.model.apply_model(x, t, uncond)
            model_output = model_uncond + cfg_scale * (model_cond - model_uncond)

        if self.model.parameterization == "v":
            e_t = self.model.predict_eps_from_z_and_v(x, t, model_output)
        else:
            e_t = model_output
        return e_t

                                                                        
    @torch.no_grad()
    def shift_sample_1_order(
        self,
        steps: int,
        shape: Tuple[int],
        cond_img: torch.Tensor,
        positive_prompt: str = "clean, high-resolution, 8k",
        negative_prompt: str = "",
        x_T: Optional[torch.Tensor] = None,
        cfg_scale: float = 1.0,
        color_fix_type: str = "none",
    ) -> torch.Tensor:

                       
        self.make_schedule(num_steps=steps)

        device = next(self.model.parameters()).device
        b = shape[0]

        time_range = np.flip(self.timesteps)                         
        total_steps = len(self.timesteps)

        iterator = tqdm(time_range[:-1],
                        desc="Spaced Sampler", total=total_steps - 1)

                  
        cond = {
            "c_latent":   [self.model.apply_condition_encoder(cond_img)],
            "c_crossattn": [self.model.get_learned_conditioning([positive_prompt] * b)],
        }
        uncond = {
            "c_latent":   [self.model.apply_condition_encoder(cond_img)],
            "c_crossattn": [self.model.get_learned_conditioning([negative_prompt] * b)],
        }

                
        encoder_posterior = self.model.encode_first_stage(x_T * 2 - 1)
        x0_hat = self.model.get_first_stage_encoding(encoder_posterior).detach()

        with torch.no_grad():
                                       
                                                                          
            pass               

        img = (
            self.sqrt_alphas_cumprod[-1] * x0_hat
            + self.sigmas[-1] * torch.randn_like(x0_hat)
        )

                                      
        for i, _ in enumerate(iterator):
            ts = torch.full((b,), time_range[i],
                            device=device, dtype=torch.long)
            index = torch.full_like(ts, fill_value=total_steps - i - 1)
            j = total_steps - i - 1

                            
            e_t = self.predict_noise(img, ts, cond, cfg_scale, uncond)
            pred_x0 = self._predict_xstart_from_eps(img, index, e_t)

                      
            r_lambdas = self.lambdas[j - 1] / self.lambdas[j]
            r1 = (
                self.alphas_multi_one_minus_etas[j - 1]
                / self.alphas_multi_one_minus_etas[j]
            )

                       
            rec_x0_hat = (
                self.alphas_multi_one_minus_etas[j - 1]
                * (
                    self.etas[j - 1] / (1 - self.etas[j - 1])
                    - self.etas[j] / (1 - self.etas[j]) * r_lambdas ** 2
                )
                * x0_hat
            )
            rec_x0 = (
                self.alphas_multi_one_minus_etas[j - 1]
                * (1 - r_lambdas ** 2)
                * pred_x0
            )

                       
            sigma_hat = np.sqrt(
                self.lambdas[j - 1] ** 2
                - self.lambdas[j - 1] ** 2 * r_lambdas ** 2
            )
            noise = torch.randn_like(img) * sigma_hat * self.alphas_multi_one_minus_etas[j - 1]

                              
            img = r1 * r_lambdas ** 2 * img + rec_x0_hat + rec_x0 + noise

                   
        img_pixel = (self.model.decode_first_stage(img) + 1) / 2

                    
                                       
                                                                              
                                           
                                                                     
               
                                                                                             

        return img_pixel