## Restormer: Efficient Transformer for High-Resolution Image Restoration
## Syed Waqas Zamir, Aditya Arora, Salman Khan, Munawar Hayat, Fahad Shahbaz Khan, and Ming-Hsuan Yang
## https://arxiv.org/abs/2111.09881

##--------------------------------------------------------------
##------- Demo file to test Restormer on your own images---------
## Example usage on directory containing several images:   python demo.py --task Single_Image_Defocus_Deblurring --input_dir './demo/degraded/' --result_dir './demo/restored/'
## Example usage on a image directly: python demo.py --task Single_Image_Defocus_Deblurring --input_dir './demo/degraded/portrait.jpg' --result_dir './demo/restored/'
## Example usage with tile option on a large image: python demo.py --task Single_Image_Defocus_Deblurring --input_dir './demo/degraded/portrait.jpg' --result_dir './demo/restored/' --tile 720 --tile_overlap 32
##--------------------------------------------------------------

import torch
import torch.nn.functional as F
import torch_npu
from torch_npu.contrib import transfer_to_npu
import os
from runpy import run_path
from glob import glob
import cv2
from tqdm import tqdm
import argparse
import numpy as np

class RestormerPredictor:
    def __init__(self, model_path: str, task: str = 'Motion_Deblurring', device: str = None, tile: int = 720, tile_overlap: int = 32):
        """
        初始化 Restormer 模型
        :param model_path: 权重文件路径 (.pth)
        :param task: 任务类型
        :param device: 'cuda' 或 'cpu'
        :param tile: 瓦片大小
        :param tile_overlap: 瓦片重叠大小
        """
        self.device = device if device else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.task = task
        self.tile = tile
        self.tile_overlap = tile_overlap

        # Define parameters based on task
        parameters = {'inp_channels':3, 'out_channels':3, 'dim':48, 'num_blocks':[4,6,6,8], 'num_refinement_blocks':4, 'heads':[1,2,4,8], 'ffn_expansion_factor':2.66, 'bias':False, 'LayerNorm_type':'WithBias', 'dual_pixel_task':False}

        if task == 'Real_Denoising':
            parameters['LayerNorm_type'] =  'BiasFree'
        elif task == 'Gaussian_Color_Denoising':
            parameters['LayerNorm_type'] =  'BiasFree'
        elif task == 'Gaussian_Gray_Denoising':
            parameters['inp_channels'] =  1
            parameters['out_channels'] =  1
            parameters['LayerNorm_type'] =  'BiasFree'

        # Load model architecture
        current_dir = '/root/autodl-tmp/backend'
        arch_path = os.path.join(current_dir, 'basicsr', 'models', 'archs', 'restormer_arch.py')
        load_arch = run_path(arch_path)
        self.model = load_arch['Restormer'](**parameters)
        self.model.to(self.device)

        # Load weights
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model weights not found at: {model_path}")
            
        checkpoint = torch.load(model_path, map_location=self.device)
        self.model.load_state_dict(checkpoint['params'])
        self.model.eval()
        print(f"Restormer initialized for task: {task} on {self.device}")

    def process(self, image: np.ndarray) -> np.ndarray:
        """
        处理图像
        :param image: 输入图像 (H, W, 3) RGB, uint8
        :return: 处理后的图像 (H, W, 3) RGB, uint8
        """
        img_multiple_of = 8
        
        # Pre-processing
        img = image.astype(np.float32) / 255.0
        if len(img.shape) == 2:
             img = np.expand_dims(img, axis=2)
             
        input_ = torch.from_numpy(img).permute(2,0,1).unsqueeze(0).to(self.device)

        # Pad the input if not_multiple_of 8
        height, width = input_.shape[2], input_.shape[3]
        H, W = ((height+img_multiple_of)//img_multiple_of)*img_multiple_of, ((width+img_multiple_of)//img_multiple_of)*img_multiple_of
        padh = H - height if height % img_multiple_of != 0 else 0
        padw = W - width if width % img_multiple_of != 0 else 0
        input_ = F.pad(input_, (0, padw, 0, padh), 'reflect')

        with torch.no_grad():
            if self.tile is None:
                ## Testing on the original resolution image
                restored = self.model(input_)
            else:
                # test the image tile by tile
                b, c, h, w = input_.shape
                tile = min(self.tile, h, w)
                assert tile % 8 == 0, "tile size should be multiple of 8"
                tile_overlap = self.tile_overlap

                stride = tile - tile_overlap
                h_idx_list = list(range(0, h-tile, stride)) + [h-tile]
                w_idx_list = list(range(0, w-tile, stride)) + [w-tile]
                E = torch.zeros(b, c, h, w).type_as(input_)
                W_mask = torch.zeros_like(E)

                for h_idx in h_idx_list:
                    for w_idx in w_idx_list:
                        in_patch = input_[..., h_idx:h_idx+tile, w_idx:w_idx+tile]
                        out_patch = self.model(in_patch)
                        out_patch_mask = torch.ones_like(out_patch)

                        E[..., h_idx:(h_idx+tile), w_idx:(w_idx+tile)].add_(out_patch)
                        W_mask[..., h_idx:(h_idx+tile), w_idx:(w_idx+tile)].add_(out_patch_mask)
                restored = E.div_(W_mask)

            restored = torch.clamp(restored, 0, 1)

            # Unpad the output
            restored = restored[:, :, :height, :width]

            restored = restored.permute(0, 2, 3, 1).cpu().detach().numpy()
            restored = restored[0] * 255.0
            restored = restored.astype(np.uint8)
            
            return restored

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Test Restormer on your own images')
    parser.add_argument('--input_dir', default='/home/gynas3/wyworkspace/Restormer/a', type=str, help='Directory of input images or path of single image')
    parser.add_argument('--result_dir', default='/home/gynas3/wyworkspace/Restormer/a1', type=str, help='Directory for restored results')
    parser.add_argument('--task', default='Motion_Deblurring', type=str, help='Task to run', choices=['Motion_Deblurring',
                                                                                        'Single_Image_Defocus_Deblurring',
                                                                                        'Deraining',
                                                                                        'Real_Denoising',
                                                                                        'Gaussian_Gray_Denoising',
                                                                                        'Gaussian_Color_Denoising'])
    parser.add_argument('--tile', type=int, default=None, help='Tile size (e.g 720). None means testing on the original resolution image')
    parser.add_argument('--tile_overlap', type=int, default=32, help='Overlapping of different tiles')

    args = parser.parse_args()

    def load_img(filepath):
        return cv2.cvtColor(cv2.imread(filepath), cv2.COLOR_BGR2RGB)

    def save_img(filepath, img):
        cv2.imwrite(filepath, cv2.cvtColor(img, cv2.COLOR_RGB2BGR))

    def load_gray_img(filepath):
        return np.expand_dims(cv2.imread(filepath, cv2.IMREAD_GRAYSCALE), axis=2)

    def save_gray_img(filepath, img):
        cv2.imwrite(filepath, img)

    def get_weights_and_parameters(task):
        if task == 'Motion_Deblurring':
            weights = os.path.join('Motion_Deblurring', 'pretrained_models', 'motion_deblurring.pth')
        elif task == 'Single_Image_Defocus_Deblurring':
            weights = os.path.join('Defocus_Deblurring', 'pretrained_models', 'single_image_defocus_deblurring.pth')
        elif task == 'Deraining':
            weights = os.path.join('Deraining', 'pretrained_models', 'deraining.pth')
        elif task == 'Real_Denoising':
            weights = os.path.join('Denoising', 'pretrained_models', 'real_denoising.pth')
        elif task == 'Gaussian_Color_Denoising':
            weights = os.path.join('Denoising', 'pretrained_models', 'gaussian_color_denoising_blind.pth')
        elif task == 'Gaussian_Gray_Denoising':
            weights = os.path.join('Denoising', 'pretrained_models', 'gaussian_gray_denoising_blind.pth')
        return weights

    task = args.task
    inp_dir = args.input_dir
    out_dir = os.path.join(args.result_dir, task)

    os.makedirs(out_dir, exist_ok=True)

    extensions = ['jpg', 'JPG', 'png', 'PNG', 'jpeg', 'JPEG', 'bmp', 'BMP']

    if any([inp_dir.endswith(ext) for ext in extensions]):
        files = [inp_dir]
    else:
        files = []
        for ext in extensions:
            files.extend(glob(os.path.join(inp_dir, '*.'+ext)))

    if len(files) == 0:
        raise Exception(f'No files found at {inp_dir}')

    # Get model weights
    weights = get_weights_and_parameters(task)
    
    # Initialize Predictor
    predictor = RestormerPredictor(
        model_path=weights, 
        task=task, 
        tile=args.tile, 
        tile_overlap=args.tile_overlap
    )

    print(f"\n ==> Running {task} with weights {weights}\n ")

    for file_ in tqdm(files):
        if torch.cuda.is_available():
            torch.cuda.ipc_collect()
            torch.cuda.empty_cache()

        if task == 'Gaussian_Gray_Denoising':
            img = load_gray_img(file_)
        else:
            img = load_img(file_)

        restored = predictor.process(img)

        f = os.path.splitext(os.path.split(file_)[-1])[0]
        if task == 'Gaussian_Gray_Denoising':
            save_gray_img((os.path.join(out_dir, f+'.png')), restored)
        else:
            save_img((os.path.join(out_dir, f+'.png')), restored)

    print(f"\nRestored images are saved at {out_dir}")
