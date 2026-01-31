from typing import Mapping, Any
import copy
from collections import OrderedDict
import torch.nn.functional as F

import einops
import torch
import torch as th
import torch.nn as nn

from ldm.modules.diffusionmodules.util import (
    conv_nd,
    linear,
    zero_module,
    timestep_embedding,
    extract_into_tensor,
)
from ldm.modules.attention import SpatialTransformer
from ldm.modules.diffusionmodules.openaimodel import TimestepEmbedSequential, ResBlock, Downsample, AttentionBlock, UNetModel
from ldm.models.diffusion.ddpm import LatentDiffusion
from ldm.util import log_txt_as_img, exists, instantiate_from_config
from ldm.modules.distributions.distributions import DiagonalGaussianDistribution
from utils.common import frozen_module

import matplotlib.pyplot as plt
import os
from utils.perceptual_loss import SimplePerceptualLoss, EdgeAwareLoss

def TV_loss(x):
    loss = torch.mean(torch.abs(x[:, :, :-1, :] - x[:, :, 1:, :])) +\
           torch.mean(torch.abs(x[:, :, :, :-1] - x[:, :, :, 1:]))
    return loss

class ControlledUnetModel(UNetModel):
    def forward(self, x, timesteps=None, context=None, control=None, only_mid_control=False, **kwargs):
        hs = []
        with torch.no_grad():
            t_emb = timestep_embedding(timesteps, self.model_channels, repeat_only=False)
            emb = self.time_embed(t_emb)
            h = x.type(self.dtype)
            for module in self.input_blocks:
                h = module(h, emb, context)
                hs.append(h)
            h = self.middle_block(h, emb, context)

        if control is not None:
            h += control.pop()

        for i, module in enumerate(self.output_blocks):
            if only_mid_control or control is None:
                h = torch.cat([h, hs.pop()], dim=1)
            else:
                h = torch.cat([h, hs.pop() + control.pop()], dim=1)
            h = module(h, emb, context)

        h = h.type(x.dtype)
        return self.out(h)


class ControlNet(nn.Module):
    def __init__(
        self,
        image_size,
        in_channels,
        model_channels,
        hint_channels,
        num_res_blocks,
        attention_resolutions,
        dropout=0,
        channel_mult=(1, 2, 4, 8),
        conv_resample=True,
        dims=2,
        use_checkpoint=False,
        use_fp16=False,
        num_heads=-1,
        num_head_channels=-1,
        num_heads_upsample=-1,
        use_scale_shift_norm=False,
        resblock_updown=False,
        use_new_attention_order=False,
        use_spatial_transformer=False,                              
        transformer_depth=1,                              
        context_dim=None,                              
        n_embed=None,                                                                                       
        legacy=True,
        disable_self_attentions=None,
        num_attention_blocks=None,
        disable_middle_self_attn=False,
        use_linear_in_transformer=False,
    ):
        super().__init__()
        if use_spatial_transformer:
            assert context_dim is not None, 'Fool!! You forgot to include the dimension of your cross-attention conditioning...'

        if context_dim is not None:
            assert use_spatial_transformer, 'Fool!! You forgot to use the spatial transformer for your cross-attention conditioning...'
            from omegaconf.listconfig import ListConfig
            if type(context_dim) == ListConfig:
                context_dim = list(context_dim)

        if num_heads_upsample == -1:
            num_heads_upsample = num_heads

        if num_heads == -1:
            assert num_head_channels != -1, 'Either num_heads or num_head_channels has to be set'

        if num_head_channels == -1:
            assert num_heads != -1, 'Either num_heads or num_head_channels has to be set'

        self.dims = dims
        self.image_size = image_size
        self.in_channels = in_channels
        self.model_channels = model_channels
        if isinstance(num_res_blocks, int):
            self.num_res_blocks = len(channel_mult) * [num_res_blocks]
        else:
            if len(num_res_blocks) != len(channel_mult):
                raise ValueError("provide num_res_blocks either as an int (globally constant) or "
                                 "as a list/tuple (per-level) with the same length as channel_mult")
            self.num_res_blocks = num_res_blocks
        if disable_self_attentions is not None:
                                                                                                                    
            assert len(disable_self_attentions) == len(channel_mult)
        if num_attention_blocks is not None:
            assert len(num_attention_blocks) == len(self.num_res_blocks)
            assert all(map(lambda i: self.num_res_blocks[i] >= num_attention_blocks[i], range(len(num_attention_blocks))))
            print(f"Constructor of UNetModel received num_attention_blocks={num_attention_blocks}. "
                  f"This option has LESS priority than attention_resolutions {attention_resolutions}, "
                  f"i.e., in cases where num_attention_blocks[i] > 0 but 2**i not in attention_resolutions, "
                  f"attention will still not be set.")

        self.attention_resolutions = attention_resolutions
        self.dropout = dropout
        self.channel_mult = channel_mult
        self.conv_resample = conv_resample
        self.use_checkpoint = use_checkpoint
        self.dtype = th.float16 if use_fp16 else th.float32
        self.num_heads = num_heads
        self.num_head_channels = num_head_channels
        self.num_heads_upsample = num_heads_upsample
        self.predict_codebook_ids = n_embed is not None

        time_embed_dim = model_channels * 4
        self.time_embed = nn.Sequential(
            linear(model_channels, time_embed_dim),
            nn.SiLU(),
            linear(time_embed_dim, time_embed_dim),
        )

        self.input_blocks = nn.ModuleList(
            [
                TimestepEmbedSequential(
                    conv_nd(dims, in_channels + hint_channels, model_channels, 3, padding=1)
                )
            ]
        )
        self.zero_convs = nn.ModuleList([self.make_zero_conv(model_channels)])

        self._feature_size = model_channels
        input_block_chans = [model_channels]
        ch = model_channels
        ds = 1
        for level, mult in enumerate(channel_mult):
            for nr in range(self.num_res_blocks[level]):
                layers = [
                    ResBlock(
                        ch,
                        time_embed_dim,
                        dropout,
                        out_channels=mult * model_channels,
                        dims=dims,
                        use_checkpoint=use_checkpoint,
                        use_scale_shift_norm=use_scale_shift_norm,
                    )
                ]
                ch = mult * model_channels
                if ds in attention_resolutions:
                    if num_head_channels == -1:
                        dim_head = ch // num_heads
                    else:
                        num_heads = ch // num_head_channels
                        dim_head = num_head_channels
                    if legacy:
                                       
                        dim_head = ch // num_heads if use_spatial_transformer else num_head_channels
                    if exists(disable_self_attentions):
                        disabled_sa = disable_self_attentions[level]
                    else:
                        disabled_sa = False

                    if not exists(num_attention_blocks) or nr < num_attention_blocks[level]:
                        layers.append(
                            AttentionBlock(
                                ch,
                                use_checkpoint=use_checkpoint,
                                num_heads=num_heads,
                                num_head_channels=dim_head,
                                use_new_attention_order=use_new_attention_order,
                            ) if not use_spatial_transformer else SpatialTransformer(
                                ch, num_heads, dim_head, depth=transformer_depth, context_dim=context_dim,
                                disable_self_attn=disabled_sa, use_linear=use_linear_in_transformer,
                                use_checkpoint=use_checkpoint
                            )
                        )
                self.input_blocks.append(TimestepEmbedSequential(*layers))
                self.zero_convs.append(self.make_zero_conv(ch))
                self._feature_size += ch
                input_block_chans.append(ch)
            if level != len(channel_mult) - 1:
                out_ch = ch
                self.input_blocks.append(
                    TimestepEmbedSequential(
                        ResBlock(
                            ch,
                            time_embed_dim,
                            dropout,
                            out_channels=out_ch,
                            dims=dims,
                            use_checkpoint=use_checkpoint,
                            use_scale_shift_norm=use_scale_shift_norm,
                            down=True,
                        )
                        if resblock_updown
                        else Downsample(
                            ch, conv_resample, dims=dims, out_channels=out_ch
                        )
                    )
                )
                ch = out_ch
                input_block_chans.append(ch)
                self.zero_convs.append(self.make_zero_conv(ch))
                ds *= 2
                self._feature_size += ch

        if num_head_channels == -1:
            dim_head = ch // num_heads
        else:
            num_heads = ch // num_head_channels
            dim_head = num_head_channels
        if legacy:
                           
            dim_head = ch // num_heads if use_spatial_transformer else num_head_channels
        self.middle_block = TimestepEmbedSequential(
            ResBlock(
                ch,
                time_embed_dim,
                dropout,
                dims=dims,
                use_checkpoint=use_checkpoint,
                use_scale_shift_norm=use_scale_shift_norm,
            ),
            AttentionBlock(
                ch,
                use_checkpoint=use_checkpoint,
                num_heads=num_heads,
                num_head_channels=dim_head,
                use_new_attention_order=use_new_attention_order,
            ) if not use_spatial_transformer else SpatialTransformer(                           
                ch, num_heads, dim_head, depth=transformer_depth, context_dim=context_dim,
                disable_self_attn=disable_middle_self_attn, use_linear=use_linear_in_transformer,
                use_checkpoint=use_checkpoint
            ),
            ResBlock(
                ch,
                time_embed_dim,
                dropout,
                dims=dims,
                use_checkpoint=use_checkpoint,
                use_scale_shift_norm=use_scale_shift_norm,
            ),
        )
        self.middle_block_out = self.make_zero_conv(ch)
        self._feature_size += ch

    def make_zero_conv(self, channels):
        return TimestepEmbedSequential(zero_module(conv_nd(self.dims, channels, channels, 1, padding=0)))

    def forward(self, x, hint, timesteps, context, **kwargs):
        t_emb = timestep_embedding(timesteps, self.model_channels, repeat_only=False)
        emb = self.time_embed(t_emb)
        x = torch.cat((x, hint), dim=1)
        outs = []

        h = x.type(self.dtype)
        for module, zero_conv in zip(self.input_blocks, self.zero_convs):
            h = module(h, emb, context)
            outs.append(zero_conv(h, emb, context))

        h = self.middle_block(h, emb, context)
        outs.append(self.middle_block_out(h, emb, context))

        return outs


class ControlLDM(LatentDiffusion):

    def __init__(
        self,
        control_stage_config: Mapping[str, Any],
        control_key: str,
        sd_locked: bool,
        only_mid_control: bool,
        learning_rate: float,
        shifting_sequence="none",
        cross_modal_mode=False,                    
        unet_encoder_only=False,                  
        *args,
        **kwargs
    ) -> "ControlLDM":
        super().__init__(*args, **kwargs)
                                    
        self.control_model: ControlNet = instantiate_from_config(control_stage_config)
        self.control_key = control_key
        self.sd_locked = sd_locked
        self.only_mid_control = only_mid_control
        self.learning_rate = learning_rate
        self.control_scales = [1.0] * 13
        self.shifting_sequence = shifting_sequence
        self.cross_modal_mode = cross_modal_mode               
        self.unet_encoder_only = unet_encoder_only                     

                                   
        print(f"shifting_sequence: {self.shifting_sequence}")
        print(f"cross_modal_mode: {self.cross_modal_mode}")                 
        print(f"unet_encoder_only: {self.unet_encoder_only}")                     

                                                
        self.perceptual_loss = SimplePerceptualLoss(loss_type='mse')
        self.edge_loss = EdgeAwareLoss()
        
                                                                                  
                                                                              
                                                                          
        self.cond_encoder = nn.Sequential(OrderedDict([
            ("encoder", copy.deepcopy(self.first_stage_model.encoder)),                            
            ("quant_conv", copy.deepcopy(self.first_stage_model.quant_conv))                          
        ]))
        frozen_module(self.cond_encoder)

    def apply_condition_encoder(self, control):
        c_latent_meanvar = self.cond_encoder(control * 2 - 1)
        c_latent = DiagonalGaussianDistribution(c_latent_meanvar).mode()                
        c_latent = c_latent * self.scale_factor
        return c_latent
    
    @torch.no_grad()
    def get_input(self, batch, k, bs=None, *args, **kwargs):
        x, c = super().get_input(batch, self.first_stage_key, *args, **kwargs)
        control = batch[self.control_key]
        if bs is not None:
            control = control[:bs]
        control = control.to(self.device)
        control = einops.rearrange(control, 'b h w c -> b c h w')
        control = control.to(memory_format=torch.contiguous_format).float()
        lq = control
                                
        H, W = lq.shape[-2], lq.shape[-1]
                                                                                                             
                                                          
                                 
        preprocess_output = lq
        preprocess_output_latent = self.apply_condition_encoder(preprocess_output)
                                 
        c2 = dict(c_crossattn=[c], c_latent=[preprocess_output_latent], lq=[lq], c_concat=[control], c_preprocess=[preprocess_output_latent])
        return x, c2

    def normalize_and_to_numpy(self, tensor):
        tensor = tensor.detach().cpu()
        tensor = (tensor - tensor.min()) / (tensor.max() - tensor.min() + 1e-5)
        return tensor.numpy()

    def shared_step(self, batch, **kwargs):
        

        x, c = self.get_input(batch, self.first_stage_key)

        preprocess_x = c['c_preprocess'][0]
        t = torch.randint(0, self.num_timesteps, (x.shape[0],), device=self.device).long()


        if self.shifting_sequence == "linear":
            pred_x0 = self.predict_x0_out_linear_mix(x, preprocess_x, c, t, **kwargs) 
            loss_cross = 0
        elif self.shifting_sequence == "nonlinear":
            pred_x0, cross_mod = self.predict_x0_out_non_linear_mix(x, preprocess_x, c, t, **kwargs) 


        elif self.shifting_sequence == "dynamic":
            pred_x0, _ = self.predict_x0_out_mix_dynamic_conv(x, preprocess_x, c, t, **kwargs) 
                                                        
            loss_cross = 0                    

                                     
        loss_reconstruction = F.l1_loss(pred_x0, x)                        
        
                                     
        loss_perceptual = self.perceptual_loss(pred_x0, x)                
        
                           
        loss_edge = self.edge_loss(pred_x0, x)          
        
                  
        loss_mse = F.mse_loss(pred_x0, x)
        total_loss = loss_mse + loss_cross
        self.log("loss_mse", loss_mse, prog_bar=True, logger=True, on_step=True, on_epoch=False)
        self.log("loss_cross_mod", loss_cross, prog_bar=True, logger=True, on_step=True, on_epoch=False)
        self.log("total_loss", total_loss, prog_bar=True, logger=True, on_step=True, on_epoch=False)

        return total_loss

    def training_step(self, batch, batch_idx):
                                          
                                                                    
                                                                   
                                                                     
                                                               
        for k in self.ucg_training:
            p = self.ucg_training[k]["p"]
            val = self.ucg_training[k]["val"]
            if val is None:
                val = ""
            for i in range(len(batch[k])):
                if self.ucg_prng.choice(2, p=[1 - p, p]):
                    batch[k][i] = val
        loss = self.shared_step(batch)
        self.log("global_step", self.global_step,
                 prog_bar=True, logger=True, on_step=True, on_epoch=False)
        
        if self.use_scheduler:
            lr = self.optimizers().param_groups[0]['lr']
            self.log('lr_abs', lr, prog_bar=True, logger=True, on_step=True, on_epoch=False)
        return loss

    def apply_model(self, x_noisy, t, cond, *args, **kwargs):
        assert isinstance(cond, dict)
        diffusion_model = self.model.diffusion_model

        cond_txt = torch.cat(cond['c_crossattn'], 1)

        if cond['c_latent'] is None:
            eps = diffusion_model(x=x_noisy, timesteps=t, context=cond_txt, control=None, only_mid_control=self.only_mid_control)
        else:
            control = self.control_model(
                x=x_noisy, hint=torch.cat(cond['c_latent'], 1),
                timesteps=t, context=cond_txt
            )
            control = [c * scale for c, scale in zip(control, self.control_scales)]
            eps = diffusion_model(x=x_noisy, timesteps=t, context=cond_txt, control=control, only_mid_control=self.only_mid_control)

        return eps

    @torch.no_grad()
    def get_unconditional_conditioning(self, N):
        return self.get_learned_conditioning([""] * N)

    @torch.no_grad()
    def log_images(self, batch, sample_steps=50):
        log = dict()
        z, c = self.get_input(batch, self.first_stage_key)
        c_lq = c["lq"][0]
        c_latent = c["c_latent"][0]
        c_cat, c_crossattn = c["c_concat"][0], c["c_crossattn"][0]

        log["hq"] = (self.decode_first_stage(z) + 1) / 2
        log["control"] = c_cat
        log["decoded_control"] = (self.decode_first_stage(c_latent) + 1) / 2
        log["lq"] = c_lq
        
                        
        log.update(self._generate_lambda_mixed_images(z, c_latent, c))
        
                                                                                                
        
        samples = self.sample_log(
            cond={"c_concat": [c_cat], "c_crossattn": [c_crossattn], "c_latent": [c_latent]},
            steps=sample_steps
        )
        
        print(f"[Log Images] Samples shape: {samples.shape}")
        print(f"[Log Images] Samples range: [{samples.min():.3f}, {samples.max():.3f}]")
        
                                      
        log["samples"] = samples
        
        return log
    
    @torch.no_grad()
    def _generate_lambda_mixed_images(self, x_true, x_condition, c):
        lambda_mixed_images = {}
        
                
        device = x_true.device
        batch_size = x_true.shape[0]
        
                                               
        if self.shifting_sequence in ["dynamic", "nonlinear"] and hasattr(self, 'nonlinear_lambda'):
                            
            lambda_input = torch.full((batch_size, 1, 1, 1), 0.5, device=device)
            lambda_matrix = self.nonlinear_lambda(lambda_input, x_true.shape)
            one_minus_lambda = 1.0 - lambda_matrix
            
                                          
            x_mixed = lambda_matrix * x_condition + one_minus_lambda * x_true
            
            print(f"[Lambda Mixed] lambda_matrix range: [{lambda_matrix.min():.6f}, {lambda_matrix.max():.6f}], mean={lambda_matrix.mean():.6f}")
            
        elif self.shifting_sequence == "linear" and hasattr(self, 'lambdas'):
                                      
            t_mid = torch.full((batch_size,), self.num_timesteps // 2, device=device).long()
            lambda_val = extract_into_tensor(self.lambdas, t_mid, x_true.shape)
            x_mixed = lambda_val * x_condition + (1.0 - lambda_val) * x_true
            
            print(f"[Lambda Mixed] linear lambda range: [{lambda_val.min():.6f}, {lambda_val.max():.6f}]")
            
        else:
                         
            x_mixed = 0.5 * x_condition + 0.5 * x_true
            print("[Lambda Mixed] using default 50% mix")
        
                                    
        x_mixed_decoded = (self.decode_first_stage(x_mixed) + 1) / 2             
        
        print(f"[Lambda Mixed] x_mixed range: [{x_mixed.min():.6f}, {x_mixed.max():.6f}]")
        print(f"[Lambda Mixed] decoded range: [{x_mixed_decoded.min():.6f}, {x_mixed_decoded.max():.6f}]")
        
                                
        lambda_mixed_images["lambda_mixed"] = x_mixed_decoded
        
        return lambda_mixed_images

    @torch.no_grad()
    def sample_log(self, cond, steps):

        if self.shifting_sequence == "linear":
            from .linear_sampler import SpacedSampler
        elif self.shifting_sequence == "nonlinear":
            from .shift_sampler import SpacedSampler
        elif self.shifting_sequence == "dynamic":
            from .shift_sampler_dynamic import SpacedSampler
        else:
            raise ValueError(f"Unsupported shifting_sequence: {self.shifting_sequence}")

        sampler = SpacedSampler(self)
        b, c, h, w = cond["c_concat"][0].shape           
        
                         
        shape = (b, self.channels, h // 8, w // 8)                        
        
                
        cond_img = cond["c_concat"][0]                           
        
                       
        x_T = cond_img               
        
        print(f"[Sample Log] Condition shape: {cond_img.shape}")
        print(f"[Sample Log] x_T shape: {x_T.shape}")
        print(f"[Sample Log] Target latent shape: {shape}")
        
               
        samples = sampler.shift_sample_1_order(
            steps=steps,
            shape=shape,
            cond_img=cond_img,
            positive_prompt="clean, high-resolution, 8k",
            negative_prompt="",
            x_T=x_T,
            cfg_scale=1.0,
            color_fix_type="none"
        )
        
        print(f"[Sample Log] Generated samples shape: {samples.shape}")
        return samples

    def configure_optimizers(self):
        lr = self.learning_rate
                                                       
                         
                                                                                  
        params = list(self.control_model.parameters())
        if not self.sd_locked:
            if self.unet_encoder_only:
                                                                         
                params += list(self.model.diffusion_model.input_blocks.parameters())
                params += list(self.model.diffusion_model.middle_block.parameters())
                print("Training mode: UNet Encoder + Middle Block only")
            else:
                                                                      
                params += list(self.model.diffusion_model.output_blocks.parameters())
                params += list(self.model.diffusion_model.out.parameters())
                print("Training mode: UNet Decoder (output_blocks + out)")
        else:
            print("Training mode: SD locked - ControlNet only")
                               
                                                       
                            
                                   
                                                                       
                                             
            
                                                                                     
                                                       
        opt = torch.optim.AdamW([
            {'params': self.nonlinear_lambda.parameters(), 'lr': 10 * lr},
                                                                                            
            {'params': params, 'lr': lr}
        ])
        return opt
        return opt

    def validation_step(self, batch, batch_idx):
                                               
        jpg, hint, txt = batch['jpg'], batch['hint'], batch['txt']
                                                                 
                                                                   
                                                             
                          
        loss = self.shared_step(batch)
        self.log('val_loss', loss, prog_bar=True, logger=True)
        return loss