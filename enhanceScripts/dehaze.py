import os
import argparse
import cv2
import glob
import torch
import torch_npu
from torch_npu.contrib import transfer_to_npu
import numpy as np
from tqdm import tqdm
from basicsr.utils import imwrite, tensor2img, img2tensor
from basicsr.utils import parallel_decode
from basicsr.archs.dehazeToken_arch import DehazeTokenNet, Critic

class DehazePredictor:
    def __init__(self, predictor_path, critic_path, device=None, max_size=1024, num_iterations=8):
        self.device = device if device else torch.device('npu')
        self.max_size = max_size
        self.num_iterations = num_iterations

        print(f"Initializing DehazePredictor on {self.device}...")
        
        self.net_g = DehazeTokenNet(codebook_params=[64, 1024, 256], blk_depth=16, LQ_stage=True, predictor_name='swinLayer').to(self.device)
        self.net_g.load_state_dict(torch.load(predictor_path, map_location=self.device)['params'], strict=True)
        self.net_g.eval()
        
        self.net_critic = Critic().to(self.device)
        self.net_critic.load_state_dict(torch.load(critic_path, map_location=self.device)['params'], strict=True)
        self.net_critic.eval()
        print("Models loaded successfully.")

    def tokens_to_logits(self, seq: torch.Tensor, h=0, w=0, critic=False) -> torch.Tensor:
        if critic:
            logits = self.net_critic(seq, h, w)
        else:
            logits = self.net_g.transformer(seq, critic)
        return logits 

    def tokens_to_feats(self, seq: torch.Tensor) -> torch.Tensor:
        '''
        seq :b, 1, h, w
        '''
        feats = self.net_g.vqgan.quantize.get_codebook_entry(seq.long())
        return feats

    def process(self, image: np.ndarray) -> np.ndarray:
        """
        Process a single image.
        :param image: Input image as numpy array (H, W, 3). Assumed RGB (0-255).
        :return: Output image as numpy array (H, W, 3). RGB (0-255).
        """
        # Pre-process: Normalize and Convert to Tensor
        # basicsr img2tensor expects float32 [0,1] or uint8 [0,255].
        # If float32=True (default), it converts to float32 [0,1].
        # bgr2rgb=False because input is already RGB.
        
        img_tensor = img2tensor(image.astype(np.float32) / 255.0, bgr2rgb=False, float32=True)
        img_tensor = img_tensor.unsqueeze(0).to(self.device)

        with torch.no_grad():
            _, _, origin_h, origin_w = img_tensor.shape

            # Resize if too large
            if origin_h * origin_w >= self.max_size**2:
                scale = self.max_size / max(origin_h, origin_w)
                img_tensor = torch.nn.UpsamplingBilinear2d(scale_factor=scale)(img_tensor)

            # Padding
            wsz = 32
            _, _, h_old, w_old = img_tensor.shape
            h_pad = (h_old // wsz + 1) * wsz - h_old
            w_pad = (w_old // wsz + 1) * wsz - w_old
            img_tensor = torch.cat([img_tensor, torch.flip(img_tensor, [2])], 2)[:, :, :h_old + h_pad, :]
            img_tensor = torch.cat([img_tensor, torch.flip(img_tensor, [3])], 3)[:, :, :, :w_old + w_pad]

            # Encoder
            enc_feats = self.net_g.vqgan.multiscale_encoder(img_tensor.detach())
            enc_feats = enc_feats[::-1]

            x = enc_feats[0]
            b, c, h, w = x.shape

            feat_to_quant = self.net_g.vqgan.before_quant(x)
            
            mask_tokens = -1 * torch.ones(b, h*w).to(feat_to_quant.device).long()

            # Parallel Decode
            output_tokens, mask_tokens = parallel_decode.decode_critic(
                mask_tokens,
                feat_to_quant,
                self.tokens_to_logits,
                self.tokens_to_feats,
                num_iter=self.num_iterations,
            )
            
            quant_feats = self.tokens_to_feats(output_tokens[:, -1, :].reshape(b, 1, h, w))

            # Generator
            after_quant_feat = self.net_g.vqgan.after_quant(quant_feats)

            x = after_quant_feat
            for i in range(self.net_g.max_depth):
                cur_res = self.net_g.gt_res // 2**self.net_g.max_depth * 2**i
                x = self.net_g.fuse_convs_dict[str(cur_res)](enc_feats[i].detach(), x, 1)
                x = self.net_g.vqgan.decoder_group[i](x)
                
            output_img = self.net_g.vqgan.out_conv(x)
            output_img = output_img[..., :h_old , :w_old ]
            
            if origin_h * origin_w >= self.max_size**2:
                output_img = torch.nn.UpsamplingBilinear2d((origin_h, origin_w))(output_img)
                
            # Post-process: Tensor to Numpy
            # rgb2bgr=False because we want RGB output.
            output_np = tensor2img(output_img, rgb2bgr=False)
            
            return output_np

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--predictor_path', type=str, default='/home/gynas3/wyworkspace/IPC-Dehaze/ckpt/predictor.pth')
    parser.add_argument('--critic_path', type=str, default='/home/gynas3/wyworkspace/IPC-Dehaze/ckpt/critic.pth')
    parser.add_argument('-i', '--input', type=str, default='/home/gynas3/wyworkspace/IPC-Dehaze/a', help='input test image folder')
    parser.add_argument('-o', '--output', type=str, default='/home/gynas3/wyworkspace/IPC-Dehaze/a1', help='output test image folder')
    parser.add_argument('-n', type=int, default=8, help='num_iterations')
    parser.add_argument('--max_size', type=int, default=1024, help='max_size')
    
    args = parser.parse_args()
    
    os.makedirs(args.output, exist_ok=True)
    
    predictor = DehazePredictor(
        predictor_path=args.predictor_path,
        critic_path=args.critic_path,
        max_size=args.max_size,
        num_iterations=args.n
    )

    if os.path.isfile(args.input):
        paths = [args.input]
    else:
        paths = sorted(glob.glob(os.path.join(args.input, '*')))
   
    pbar = tqdm(total=len(paths), unit='image')
    for idx, path in enumerate(paths):
        imgname = os.path.splitext(os.path.basename(path))[0]
        pbar.set_description(f'Test {idx} {imgname}')
        
        # Read BGR
        img = cv2.imread(path, cv2.IMREAD_COLOR)
        # Convert to RGB for predictor
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        # Process
        result_rgb = predictor.process(img_rgb)
        
        # Convert back to BGR for saving
        result_bgr = cv2.cvtColor(result_rgb, cv2.COLOR_RGB2BGR)
        
        cv2.imwrite(os.path.join(args.output, f'{imgname}.png'), result_bgr)

        pbar.update(1)
           
    pbar.close()

if __name__ == '__main__':
    main()
