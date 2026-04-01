#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2024 Apple Inc. All Rights Reserved.
#

import torch
from lpips import LPIPS
import torch.nn as nn
import torch.nn.functional as F

from hugs.utils.sampler import PatchSampler

from .utils import l1_loss, ssim, l2_loss
import random

class HumanSceneLoss(nn.Module):
    def __init__(
        self,
        l_ssim_w=0.2,
        l_l1_w=0.8,
        l_lpips_w=0.0,
        l_lbs_w=0.0,
        l_humansep_w=0.0,
        num_patches=4,
        patch_size=32,
        use_patches=True,
        bg_color='white',
        add_depth= False,
        normalize_depth = False,
        human_depth_w_l1 = 1.0,
        human_depth_w_ssim = 1.0,
        human_depth_w_lpips = 1.0,
        depth_scale_invariant_loss = False,
        depth_scale_invariant_loss_w = 1.0,
        add_mask = False,
        mask_loss_w = 0.0,
        add_cano_mask = False,
        cano_mask_prob = 0.0,
        cano_mask_w = 0.0,
    ):
        super(HumanSceneLoss, self).__init__()
        
        self.l_ssim_w = l_ssim_w
        self.l_l1_w = l_l1_w
        self.l_lpips_w = l_lpips_w
        self.l_lbs_w = l_lbs_w
        self.l_humansep_w = l_humansep_w
        self.use_patches = use_patches
        
        self.bg_color = bg_color
        self.lpips = LPIPS(net="vgg", pretrained=True).to('cuda')
    
        for param in self.lpips.parameters(): param.requires_grad=False
        
        if self.use_patches:
            self.patch_sampler = PatchSampler(num_patch=num_patches, patch_size=patch_size, ratio_mask=0.9, dilate=0)
        
        self.add_depth = add_depth
        self.normalize_depth = normalize_depth
        self.human_depth_w_l1 = human_depth_w_l1
        self.human_depth_w_ssim = human_depth_w_ssim
        self.human_depth_w_lpips = human_depth_w_lpips

        self.depth_scale_invariant_loss = depth_scale_invariant_loss
        self.depth_scale_invariant_loss_w = depth_scale_invariant_loss_w

        self.add_mask = add_mask
        self.mask_loss_w = mask_loss_w

        self.add_cano_mask = add_cano_mask
        self.cano_mask_prob = cano_mask_prob
        self.cano_mask_w = cano_mask_w

        
    def forward(
        self, 
        data, 
        render_pkg,
        human_gs_out,
        render_mode, 
        human_gs_init_values=None,
        bg_color=None,
        human_bg_color=None,
        gt_depth = None,

        filter_depth1 = False,
        filtering_depth_list = None,
        dataset_id = None,
        filter_depth2 = False,
        filter_depth2_prob = 0.0,

        render_mask_cano = None,
        cano_mask = None,

        # cls_mask_gt = None,
        # cls_mask_rd = None,
        # N_cls = 23,
    ):
        loss_dict = {}
        extras_dict = {}
        
        if bg_color is not None:
            self.bg_color = bg_color
            
        if human_bg_color is None:
            human_bg_color = self.bg_color
            
        gt_image = data['rgb']
        mask = data['mask'].unsqueeze(0)
        
        pred_img = render_pkg['render']
        
        if self.add_depth:
            pred_depth = render_pkg['human_depth']


        if render_mode == "human":
            gt_image = gt_image * mask + human_bg_color[:, None, None] * (1. - mask)
            extras_dict['gt_img'] = gt_image
            extras_dict['pred_img'] = pred_img
        elif render_mode == "scene":
            # invert the mask
            extras_dict['pred_img'] = pred_img
            
            mask = (1. - data['mask'].unsqueeze(0))
            gt_image = gt_image * mask
            pred_img = pred_img * mask
            
            extras_dict['gt_img'] = gt_image
        else:
            extras_dict['gt_img'] = gt_image
            extras_dict['pred_img'] = pred_img
            if self.add_depth:
                extras_dict['gt_depth'] = gt_depth
                extras_dict['pred_depth'] = pred_depth
            if self.normalize_depth:
                pred_depth_normalized = pred_depth
                mask1 = (pred_depth > 0.0)
                pred_depth_normalized[mask1] = (pred_depth[mask1]-pred_depth[mask1].min())/(pred_depth[mask1].max()-pred_depth[mask1].min())
                                
                gt_depth_normalized = gt_depth
                mask2 = (gt_depth > 0.0)
                gt_depth_normalized[mask2] = (gt_depth[mask2]-gt_depth[mask2].min())/(gt_depth[mask2].max()-gt_depth[mask2].min())
                
                gt_depth = gt_depth_normalized
                pred_depth = pred_depth_normalized

                extras_dict['gt_depth_normalized'] = gt_depth
                extras_dict['pred_depth_normalized'] = pred_depth


        if self.l_l1_w > 0.0:
            if render_mode == "human":
                Ll1 = l1_loss(pred_img, gt_image, mask)
            elif render_mode == "scene":
                Ll1 = l1_loss(pred_img, gt_image, 1 - mask)
            elif render_mode == "human_scene":
                Ll1 = l1_loss(pred_img, gt_image)
                # if self.add_depth:
                #     Ll1 = Ll1 + self.human_depth_w * l1_loss(pred_depth, gt_depth, mask)

            else:
                raise NotImplementedError
            loss_dict['l1'] = self.l_l1_w * Ll1

        if self.l_ssim_w > 0.0:
            loss_ssim = 1.0 - ssim(pred_img, gt_image)
            if render_mode == "human":
                loss_ssim = loss_ssim * (mask.sum() / (pred_img.shape[-1] * pred_img.shape[-2]))
            elif render_mode == "scene":
                loss_ssim = loss_ssim * ((1 - mask).sum() / (pred_img.shape[-1] * pred_img.shape[-2]))
            elif render_mode == "human_scene":
                loss_ssim = loss_ssim
                # if self.add_depth:
                #     loss_ssim = loss_ssim + self.human_depth_w * (1.0 - ssim(pred_depth, gt_depth))
                
            loss_dict['ssim'] = self.l_ssim_w * loss_ssim
        
        if self.l_lpips_w > 0.0 and not render_mode == "scene":
            if self.use_patches:
                if render_mode == "human":
                    bg_color_lpips = torch.rand_like(pred_img)
                    image_bg = pred_img * mask + bg_color_lpips * (1. - mask)
                    gt_image_bg = gt_image * mask + bg_color_lpips * (1. - mask)
                    _, pred_patches, gt_patches = self.patch_sampler.sample(mask, image_bg, gt_image_bg)
                else:
                    _, pred_patches, gt_patches = self.patch_sampler.sample(mask, pred_img, gt_image)
                    
                loss_lpips = self.lpips(pred_patches.clip(max=1), gt_patches).mean()
                loss_dict['lpips_patch'] = self.l_lpips_w * loss_lpips
            else:
                bbox = data['bbox'].to(int)
                cropped_gt_image = gt_image[:, bbox[0]:bbox[2], bbox[1]:bbox[3]]
                cropped_pred_img = pred_img[:, bbox[0]:bbox[2], bbox[1]:bbox[3]]
                loss_lpips = self.lpips(cropped_pred_img.clip(max=1), cropped_gt_image).mean()
                loss_dict['lpips'] = self.l_lpips_w * loss_lpips
                
        if self.l_humansep_w > 0.0 and render_mode == "human_scene":
            pred_human_img = render_pkg['human_img']
            gt_human_image = gt_image * mask + human_bg_color[:, None, None] * (1. - mask)
            
            Ll1_human = l1_loss(pred_human_img, gt_human_image, mask)
            if self.add_depth and not filter_depth1:
                Ll1_human_depth = l1_loss(pred_depth, gt_depth, mask)
                loss_dict['l1_human_depth'] = self.l_l1_w * self.human_depth_w_l1 * Ll1_human_depth * self.l_humansep_w
            if self.add_depth and filter_depth1:
                if dataset_id in filtering_depth_list:
                    loss_dict['l1_human_depth'] = torch.tensor(0.0).to('cuda')
                else:
                    Ll1_human_depth = l1_loss(pred_depth, gt_depth, mask)
                    loss_dict['l1_human_depth'] = self.l_l1_w * self.human_depth_w_l1 * Ll1_human_depth * self.l_humansep_w
        
            loss_dict['l1_human'] = self.l_l1_w * Ll1_human * self.l_humansep_w
            
            loss_ssim_human = 1.0 - ssim(pred_human_img, gt_human_image)
            loss_ssim_human = loss_ssim_human * (mask.sum() / (pred_human_img.shape[-1] * pred_human_img.shape[-2]))
            if self.add_depth and not filter_depth1:
                loss_ssim_human_depth = 1.0 - ssim(pred_depth, gt_depth)
                loss_ssim_human_depth = loss_ssim_human_depth * (mask.sum() / (pred_depth.shape[-1] * pred_depth.shape[-2]))
                loss_dict['ssim_human_depth'] = self.l_ssim_w * self.human_depth_w_ssim * loss_ssim_human_depth * self.l_humansep_w
            if self.add_depth and filter_depth1:
                if dataset_id in filtering_depth_list:
                    loss_dict['ssim_human_depth'] = torch.tensor(0.0).to('cuda')
                else:
                    loss_ssim_human_depth = 1.0 - ssim(pred_depth, gt_depth)
                    loss_ssim_human_depth = loss_ssim_human_depth * (mask.sum() / (pred_depth.shape[-1] * pred_depth.shape[-2]))
                    loss_dict['ssim_human_depth'] = self.l_ssim_w * self.human_depth_w_ssim * loss_ssim_human_depth * self.l_humansep_w

            loss_dict['ssim_human'] = self.l_ssim_w * loss_ssim_human * self.l_humansep_w
            
            bg_color_lpips = torch.rand_like(pred_human_img)
            image_bg = pred_human_img * mask + bg_color_lpips * (1. - mask)
            gt_image_bg = gt_human_image * mask + bg_color_lpips * (1. - mask)
            _, pred_patches, gt_patches = self.patch_sampler.sample(mask, image_bg, gt_image_bg)
            loss_lpips_human = self.lpips(pred_patches.clip(max=1), gt_patches).mean()
            if self.add_depth and not filter_depth1:
                loss_lpips_human_depth = self.lpips(pred_depth, gt_depth).squeeze(0).squeeze(0).squeeze(0).squeeze(0)
                loss_dict['lpips_patch_human_depth'] = self.l_lpips_w * self.human_depth_w_lpips * loss_lpips_human_depth * self.l_humansep_w
            if self.add_depth and filter_depth1:
                if dataset_id in filtering_depth_list:
                    loss_dict['lpips_patch_human_depth'] = torch.tensor(0.0).to('cuda')
                else:
                    loss_lpips_human_depth = self.lpips(pred_depth, gt_depth).squeeze(0).squeeze(0).squeeze(0).squeeze(0)
                    loss_dict['lpips_patch_human_depth'] = self.l_lpips_w * self.human_depth_w_lpips * loss_lpips_human_depth * self.l_humansep_w

            loss_dict['lpips_patch_human'] = self.l_lpips_w * loss_lpips_human * self.l_humansep_w


            if self.add_depth and filter_depth2:
                if random.random() <= filter_depth2_prob:
                    loss_dict['l1_human_depth'] = torch.tensor(0.0).to('cuda')
                    loss_dict['ssim_human_depth'] = torch.tensor(0.0).to('cuda')
                    loss_dict['lpips_patch_human_depth'] = torch.tensor(0.0).to('cuda')


            if self.add_depth and self.depth_scale_invariant_loss:
                render_human_depth = pred_depth.squeeze(0).clone()
                sapiens_depth_human = gt_depth.squeeze(0).clone()

                render_human_depth_normalized = render_human_depth
                mask = (render_human_depth > 0.0)
                render_human_depth_normalized[mask] = (render_human_depth[mask]-render_human_depth[mask].min())/(render_human_depth[mask].max()-render_human_depth[mask].min())
                
                sapiens_depth_human_normalized = sapiens_depth_human
                mask = (sapiens_depth_human > 0.0)
                sapiens_depth_human_normalized[mask] = (sapiens_depth_human[mask]-sapiens_depth_human[mask].min())/(sapiens_depth_human[mask].max()-sapiens_depth_human[mask].min())

                mask_sapiens = (sapiens_depth_human_normalized > 0.0)
                mask_render = (render_human_depth_normalized > 0.0)
                sapiens_depth_human_normalized_log = sapiens_depth_human_normalized
                sapiens_depth_human_normalized_log[mask_sapiens] = torch.log(sapiens_depth_human_normalized[mask_sapiens])
                render_human_depth_normalized_log = render_human_depth_normalized
                render_human_depth_normalized_log[mask_render] = torch.log(render_human_depth_normalized[mask_render])

                delta_d = sapiens_depth_human_normalized_log - render_human_depth_normalized_log

                delta_d_mean = delta_d.mean()
                delta_d_squaremean = (delta_d ** 2).mean()
                loss_depth = torch.sqrt(delta_d_squaremean - 0.5*((delta_d_mean)**2))
                loss_dict['depth_scale_invariant_loss'] = loss_depth * self.depth_scale_invariant_loss_w
            
            if self.add_mask:
                gt_mask = data['mask']
                render_mask = render_pkg['human_alpha'].squeeze(0)
                # mask_loss = ((gt_mask - render_mask) ** 2).sum()
                mask_loss = l2_loss(gt_mask, render_mask)
                loss_dict['mask'] = self.mask_loss_w * mask_loss

            if self.add_cano_mask:
                if random.random() <= self.cano_mask_prob:
                    loss_dict['mask_cano'] = self.cano_mask_w * l2_loss(cano_mask, render_mask_cano)
                else: loss_dict['mask_cano'] = torch.tensor(0.0).to('cuda')
            else: loss_dict['mask_cano'] = torch.tensor(0.0).to('cuda')


        if self.l_lbs_w > 0.0 and human_gs_out['lbs_weights'] is not None and not render_mode == "scene":
            if 'gt_lbs_weights' in human_gs_out.keys():
                loss_lbs = F.mse_loss(
                    human_gs_out['lbs_weights'], 
                    human_gs_out['gt_lbs_weights'].detach()).mean()
            else:
                loss_lbs = F.mse_loss(
                    human_gs_out['lbs_weights'], 
                    human_gs_init_values['lbs_weights']).mean()
            loss_dict['lbs'] = self.l_lbs_w * loss_lbs
        
        loss = 0.0
        for k, v in loss_dict.items():
            loss += v
        
        return loss, loss_dict, extras_dict
    