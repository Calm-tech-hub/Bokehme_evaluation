#!/usr/bin/env python
# encoding: utf-8

import os

os.environ['CUDA_VISIBLE_DEVICES'] = '2'

import matplotlib.pyplot as plt
import numpy as np
import cv2
import argparse

import torch
import torch.nn.functional as F

from neural_renderer import ARNet, IUNet

from classical_renderer.scatter import ModuleRenderScatter  # circular aperture


def gaussian_blur(x, r, sigma=None):
    r = int(round(r))
    if sigma is None:
        sigma = 0.3 * (r - 1) + 0.8
    x_grid, y_grid = torch.meshgrid(torch.arange(-int(r), int(r) + 1), torch.arange(-int(r), int(r) + 1))
    kernel = torch.exp(-(x_grid ** 2 + y_grid ** 2) / 2 / sigma ** 2)
    kernel = kernel.float() / kernel.sum()
    kernel = kernel.expand(1, 1, 2*r+1, 2*r+1).to(x.device)
    x = F.pad(x, pad=(r, r, r, r), mode='replicate')
    x = F.conv2d(x, weight=kernel, padding=0)
    return x



device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

parser = argparse.ArgumentParser(description='Bokeh Rendering', fromfile_prefix_chars='@')


parser.add_argument('--defocus_scale',             type=float, default=10.)
parser.add_argument('--gamma_min',                 type=float, default=1.)
parser.add_argument('--gamma_max',                 type=float, default=5.)

# Model 1
parser.add_argument('--arnet_shuffle_rate',          type=int,   default=2)
parser.add_argument('--arnet_in_channels',           type=int,   default=5)
parser.add_argument('--arnet_out_channels',          type=int,   default=4)
parser.add_argument('--arnet_middle_channels',       type=int,   default=128)
parser.add_argument('--arnet_num_block',             type=int,   default=3)
parser.add_argument('--arnet_share_weight',                      action='store_true')
parser.add_argument('--arnet_connect_mode',          type=str,   default='distinct_source')
parser.add_argument('--arnet_use_bn',                            action='store_true')
parser.add_argument('--arnet_activation',            type=str,   default='elu')

# Model 2
parser.add_argument('--iunet_shuffle_rate',          type=int,   default=2)
parser.add_argument('--iunet_in_channels',           type=int,   default=8)
parser.add_argument('--iunet_out_channels',          type=int,   default=3)
parser.add_argument('--iunet_middle_channels',       type=int,   default=64)
parser.add_argument('--iunet_num_block',             type=int,   default=3)
parser.add_argument('--iunet_share_weight',                      action='store_true')
parser.add_argument('--iunet_connect_mode',          type=str,   default='distinct_source')
parser.add_argument('--iunet_use_bn',                            action='store_true')
parser.add_argument('--iunet_activation',            type=str,   default='elu')


# Log and save
parser.add_argument('--arnet_checkpoint_dir_path',   type=str,   help='path to a checkpoint to load', default='./checkpoints/arnet')
parser.add_argument('--arnet_checkpoint_name',       type=str,   help='path to a checkpoint to load', default='model.pth')

parser.add_argument('--iunet_checkpoint_dir_path',   type=str,   help='directory path to save checkpoints and summaries', default='./checkpoints/iunet')
parser.add_argument('--iunet_checkpoint_name',       type=str,   help='path to a checkpoint to load', default='model.pth')

# K & disp_focus
# 新增 K 和 disp_focus 参数
parser.add_argument('--K',                           type=int, default=125, help='Blur parameter')
parser.add_argument('--disp_focus',                  type=float, default=90/255, help='Focus disparity value (0~1)') 

#新增save_root参数
parser.add_argument('--save_root',                   type= str, default='./outputs_plus', help='output image path')

args = parser.parse_args()


arnet_checkpoint_path = os.path.join(args.arnet_checkpoint_dir_path, args.arnet_checkpoint_name)
iunet_checkpoint_path = os.path.join(args.iunet_checkpoint_dir_path, args.iunet_checkpoint_name)


classical_renderer = ModuleRenderScatter().to(device)


arnet = ARNet(args.arnet_shuffle_rate, args.arnet_in_channels, args.arnet_out_channels, args.arnet_middle_channels,
              args.arnet_num_block, args.arnet_share_weight, args.arnet_connect_mode, args.arnet_use_bn, args.arnet_activation)
iunet = IUNet(args.iunet_shuffle_rate, args.iunet_in_channels, args.iunet_out_channels, args.iunet_middle_channels,
              args.iunet_num_block, args.iunet_share_weight, args.iunet_connect_mode, args.iunet_use_bn, args.iunet_activation)

arnet.cuda()
iunet.cuda()

checkpoint = torch.load(arnet_checkpoint_path)
arnet.load_state_dict(checkpoint['model'])
checkpoint = torch.load(iunet_checkpoint_path)
iunet.load_state_dict(checkpoint['model'])

arnet.eval()
iunet.eval()



root = './inputs'
image_name = '278'
save_root = args.save_root
os.makedirs(save_root, exist_ok=True)

K = args.K    # blur parameter
gamma = 2.2  # 1~5
disp_focus = args.disp_focus    # 0~1


image = cv2.imread(os.path.join(root, image_name + '.jpg')).astype(np.float32) / 255.0
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

disp = np.float32(cv2.imread(os.path.join(root, image_name + '.png'), cv2.IMREAD_GRAYSCALE))
disp = (disp - disp.min()) / (disp.max() - disp.min())


##### Highlight Processing #####
highlight_threshold = 255/255
increase_ratio = 0.2
mask = np.clip(np.tanh(200 * (np.abs(disp - disp_focus)**2 - 0.01)), 0, 1)[..., np.newaxis]  # out-of-focus areas
mask = mask * (np.max(image, axis=2, keepdims=True) > highlight_threshold).astype(np.float32) ** 10  # highlight areas
image = image * (1 + mask * increase_ratio)
################################


signed_disp = disp - disp_focus

defocus = K * signed_disp / args.defocus_scale

with torch.no_grad():
    image = torch.from_numpy(image).permute(2, 0, 1).unsqueeze(0).contiguous()
    defocus = torch.from_numpy(defocus).unsqueeze(0).unsqueeze(0).contiguous()

    image = image.cuda()
    defocus = defocus.cuda()

    bokeh_classical, defocus_dilate = classical_renderer(image**gamma, defocus*args.defocus_scale)

    bokeh_classical = bokeh_classical ** (1/gamma)
    defocus_dilate = defocus_dilate / args.defocus_scale
    gamma = (gamma - args.gamma_min) / (args.gamma_max - args.gamma_min)
    adapt_scale = max(defocus.abs().max().item(), 1)

    image_re = F.interpolate(image, scale_factor=1/adapt_scale, mode='bilinear', align_corners=True)
    defocus_re = 1 / adapt_scale * F.interpolate(defocus, scale_factor=1/adapt_scale, mode='bilinear', align_corners=True)
    bokeh_neural, error_map = arnet(image_re, defocus_re, gamma)
    error_map = F.interpolate(error_map, size=(image.shape[2], image.shape[3]), mode='bilinear', align_corners=True)
    bokeh_neural.clamp_(0, 1e5)


    scale = -1
    for scale in range(int(np.log2(adapt_scale))):
        ratio = 2**(scale+1) / adapt_scale
        h_re, w_re = int(ratio * image.shape[2]), int(ratio * image.shape[3])
        
        image_re = F.interpolate(image, size=(h_re, w_re), mode='bilinear', align_corners=True)
        defocus_re = ratio * F.interpolate(defocus, size=(h_re, w_re), mode='bilinear', align_corners=True)
        defocus_dilate_re = ratio * F.interpolate(defocus_dilate, size=(h_re, w_re), mode='bilinear', align_corners=True)
        bokeh_neural_refine = iunet(image_re, defocus_re.clamp(-1, 1), bokeh_neural, gamma).clamp(0, 1e5)
        mask = gaussian_blur(((defocus_dilate_re < 1) * (defocus_dilate_re > -1)).float(), 0.005 * (defocus_dilate_re.shape[2] + defocus_dilate_re.shape[3]))
        bokeh_neural = mask * bokeh_neural_refine + (1 - mask) * F.interpolate(bokeh_neural, size=(h_re, w_re), mode='bilinear', align_corners=True)

    bokeh_neural_refine = iunet(image, defocus.clamp(-1, 1), bokeh_neural, gamma).clamp(0, 1e5)
    mask = gaussian_blur(((defocus_dilate < 1) * (defocus_dilate > -1)).float(), 0.005 * (defocus_dilate.shape[2] + defocus_dilate.shape[3]))
    bokeh_neural = mask * bokeh_neural_refine + (1 - mask) * F.interpolate(bokeh_neural, size=(image.shape[2], image.shape[3]), mode='bilinear', align_corners=True)
    bokeh_pred = bokeh_classical * (1 - error_map) + bokeh_neural * error_map


image = image[0].cpu().clone().permute(1, 2, 0).numpy()
defocus = defocus[0][0].cpu().clone().numpy()
error_map = error_map[0][0].cpu().clone().numpy()
bokeh_classical = bokeh_classical[0].cpu().clone().permute(1, 2, 0).numpy()
bokeh_neural = bokeh_neural[0].cpu().clone().permute(1, 2, 0).detach().numpy()
bokeh_pred = bokeh_pred[0].cpu().clone().permute(1, 2, 0).detach().numpy()



#cv2.imwrite(os.path.join(save_root, f'K{K}_disp{disp_focus}_bokeh_classical.jpg'), bokeh_classical[..., ::-1] * 255)
#cv2.imwrite(os.path.join(save_root, f'K{K}_disp{disp_focus}_bokeh_neural.jpg'), bokeh_neural[..., ::-1] * 255)
cv2.imwrite(os.path.join(save_root, f"K{K}_disp{disp_focus}_bokeh_pred.jpg"), bokeh_pred[..., ::-1] * 255)
