import torch 
import torch.nn as nn
import os
import yaml 
import h5py
import argparse
from data.Custom_image_dataset import dataset, test_dataset
from basicsr.metrics.psnr_ssim import calculate_psnr_pt, calculate_ssim_pt
from torch.utils.data import DataLoader
from arch.model2 import *
from  basicsr.utils.img_util import tensor2img_fast
from timm import utils
import cv2

def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    elif v.lower() in ('None', "none"):
        return None
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')
def get_norm_layer(layer_name):
    """Maps string names to PyTorch normalization classes."""
    if layer_name == 'LayerNorm' or layer_name == 'nn.LayerNorm':
        return nn.LayerNorm
    elif layer_name == 'BatchNorm':
        return nn.BatchNorm2d
    elif layer_name == 'Identity':
        return nn.Identity
    else:
        raise NotImplementedError(f"Normalization layer {layer_name} is not found")
def get_activation(act):
    if act=='GELU':
        return nn.GELU
    else:
        raise NotImplementedError(f"Activation function {act} is not found")
def load_config_and_parse_args():
    # --- 1. Initial Parser to get the config file path ---
    config_parser = argparse.ArgumentParser(add_help=False)
    config_parser.add_argument('-c', '--config', default='/home/cjrathod/projects/def-mhassanz/cjrathod/HAMBA/options/journal_model_tiny.yaml', type=str, metavar='FILE',
                                help='YAML config file specifying default arguments')
    
    # Parse just the config file path, leaving other arguments for the main parser
    config_args, remaining_argv = config_parser.parse_known_args()

    defaults = {}
    if config_args.config and os.path.exists(config_args.config):
        print(f"Configuration loaded from YAML: {config_args.config}")
        with open(config_args.config, 'r') as f:
            # Load all YAML content into a single flat dictionary
            defaults = yaml.safe_load(f) or {}
    else:
        # If no config file is provided or found, this message is useful for debugging
        print("No valid YAML configuration file loaded. Using hardcoded defaults.")
    
    if isinstance(defaults.get('norm_layer'), str):
        defaults['norm_layer']= get_norm_layer(defaults['norm_layer'])
    if isinstance(defaults.get('qk_scale'), str):
        defaults['qk_scale']= str2bool(defaults['qk_scale'])
    main_parser= argparse.ArgumentParser()
    main_parser.add_argument('--train_hr_pth', type = str)
    main_parser.add_argument('--train_lr_pth', type = str)
    main_parser.add_argument('--val_hr_pth', type = str)
    main_parser.add_argument('--val_lr_pth', type = str)
    main_parser.add_argument('--checkpoint_folder', type = str)
    main_parser.set_defaults(**defaults)
    args = main_parser.parse_args(remaining_argv)
    return args
        
def inference(model, lr_img, window_size=8, scale=2):
    """
    lr_img: Tensor (1, 3, H, W)
    window_size: The 'window_size' your model uses (usually 8 or 16 for Swin-based models)
    """
    _, _, h_old, w_old = lr_img.shape
    
    # 1. Calculate how much padding is needed
    #    We want (h_old + pad) to be divisible by window_size
    h_pad = (window_size - h_old % window_size) % window_size
    w_pad = (window_size - w_old % window_size) % window_size
    
    # 2. Pad the image (Reflect padding avoids border artifacts)
    #    F.pad order is (Left, Right, Top, Bottom)
    lr_img_padded = F.pad(lr_img, (0, w_pad, 0, h_pad), mode='reflect')

    # 3. Run Inference
    with torch.no_grad():
        sr_img_padded = model(lr_img_padded)

    # 4. Crop back to original scale
    #    If we padded 2 pixels in LR, we must crop 2*scale pixels in HR
    h_target = h_old * scale
    w_target = w_old * scale
    
    # Slice: [All Batches, All Channels, 0:Target_H, 0:Target_W]
    sr_img = sr_img_padded[:, :, :h_target, :w_target]

    return sr_img

def load_checkpoint(chkpt_pth, model, device):
    print(f"Loading checkpoint from {chkpt_pth}...")
    
    # Load to CPU first to avoid OOM, then move to device
    checkpoint = torch.load(chkpt_pth, map_location=device)
    state_dict = checkpoint['Model State']

    # FIX: Remove 'module.' prefix if it exists (from DDP training)
    new_state_dict = {}
    for k, v in state_dict.items():
        if k.startswith('module.'):
            name = k[7:] # remove 'module.'
        else:
            name = k
        new_state_dict[name] = v
        
    model.load_state_dict(new_state_dict)
    print("Checkpoint loaded successfully.")
    return model
def align_and_crop(sr_img, hr_img):
    """
    Crops both images to the minimum common height and width.
    Assumes [B, C, H, W] layout.
    """
    # Get dimensions
    _, _, h_sr, w_sr = sr_img.shape
    _, _, h_hr, w_hr = hr_img.shape
    
    # Calculate min height and width
    h_min = min(h_sr, h_hr)
    w_min = min(w_sr, w_hr)
    
    # Crop both to the minimum size
    # We crop from the top-left (0,0) because padding usually happens 
    # on the bottom/right in standard inference loops.
    sr_cropped = sr_img[:, :, :h_min, :w_min]
    hr_cropped = hr_img[:, :, :h_min, :w_min]
    return sr_cropped, hr_cropped
def main():
    utils.setup_default_logging()
    args = load_config_and_parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Running Inference on: {device}") 
    
    model = Main_Model(
            img_size=args.img_size,
            in_chans=args.in_chans,
            patch_size= args.patch_size,
            dims= args.dims,
            input_resolution=args.input_resolution, 
            depth= args.depth,
            attention_depth= args.attention_depth,
            num_heads= args.num_heads,
            kernel_size= args.kernel_size,  
            stride= args.stride, 
            dilation= args.dilation,
            window_size= args.window_size,
            overlap_ratio= args.overlap_ratio,
            qkv_bias= args.qkv_bias,
            qk_scale= args.qk_scale,
            patch_norm= args.patch_norm,
            ape= args.ape,
            downsample= args.downsample,
            d_state= args.d_state, 
            d_conv= args.d_conv, 
            expand= args.expand, 
            num_tokens= args.num_tokens,
            inner_rank= args.inner_rank,
            mlp_ratio= args.mlp_ratio,
            norm_layer= args.norm_layer,
            upsampler= args.upsampler,
            upscale= args.upscale,
            resi_connection= args.resi_connection,
            img_range= args.img_range,
            drop_rate= args.drop_rate,
            attn_drop_rate= args.attn_drop_rate,
            drop_path_rate= args.drop_path_rate).to(device)

    model = model.to(memory_format=torch.channels_last)
    model.eval()  
    latest_ckpt_path = os.path.join(args.checkpoint_folder, 'latest_checkpoint_journal_New_model_tiny.pt')
    loss_fn = nn.L1Loss()
    model = load_checkpoint(latest_ckpt_path, model, device)
   
    test_data= test_dataset(args.test_file_pth_HR,args.test_file_pth_LR)
    test_loader= DataLoader(test_data, batch_size= 1, shuffle = False)
    total_psnr = 0.0
    total_ssim = 0.0
    total_loss = 0.0
    with torch.no_grad():
        for i, img_dict in enumerate(test_loader): 
            gt_img = img_dict['gt'].to(device)
            lr_img = img_dict['lq'].to(device)
            file_name = f"Set5_{i}"#img_dict['filename'][0]
            sr_img = inference(model, lr_img, args.window_size, args.upscale)
            sr_img, gt_img= align_and_crop(sr_img, gt_img)
            file_pth= os.path.join(args.results, f"{file_name}_SR_X2_.png")
            img_sr= tensor2img_fast(sr_img, rgb2bgr= False)
            cv2.imwrite(file_pth, img_sr)
            psnr = calculate_psnr_pt(sr_img, gt_img, crop_border= args.upscale, test_y_channel = True)
            ssim = calculate_ssim_pt(sr_img, gt_img, crop_border= args.upscale, test_y_channel = True)
            loss= loss_fn(sr_img, gt_img)
            total_psnr+= psnr.item()
            total_ssim+= ssim.item()
            total_loss+= loss.item()
            print(f"Image:{file_name}|Loss:{loss.item():.4f}|PSNR:{psnr.item():.4f}|SSIM:{ssim.item():.4f}")

        avg_loss = total_loss/len(test_loader)
        avg_psnr = total_psnr/len(test_loader)
        avg_ssim = total_ssim/len(test_loader)

        print(f"Metrics for the {args.name} | Average Loss:{avg_loss:.4f} | Average PSNR: {avg_psnr:.4f} | Average SSIM: {avg_ssim:.4f}")
    
if __name__ == '__main__':
    main()