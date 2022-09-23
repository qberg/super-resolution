import argparse

import os
import torch
import numpy as np
import pandas as pd
from time import time
import PIL.Image as Image
import matplotlib.pyplot as plt
import torchvision.transforms as transforms

from model import SRCNN
from utils import calc_psnr, plt_images

parser = argparse.ArgumentParser(
    description='Parameters for deploying SRCNN on a low resoltion image')

parser.add_argument('--model',
                    type=str,
                    required=True,
                    help='Path to the trained model pth file')
parser.add_argument('--input_images_dir',
                    type=str,
                    required=True,
                    help='Path to the directory of images for conversion')
parser.add_argument('--output_dir', type=str, required=True, help='Path to the directory in which the converted images will be stored')
parser.add_argument('--scale', type=int, default=3)
parser.add_argument('--zoom', type=bool, default=False, help='Inidcates whether the output should be in higher resolution')

args = vars(parser.parse_args()) 

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

if device == 'cuda':
    print(f'\n[INFO] DEVICE: {torch.cuda.get_device_name(0)}')
    map_location=lambda storage, loc: storage.cuda()
else:
    print(f'\n[INFO] DEVICE: {device}')
    map_location='cpu'

scale = args['scale']
test_dir = args['input_images_dir']
results_dir = os.path.join(args['output_dir'],f'x{scale}/srcnn_reconstructed_images')

if (not os.path.isdir(test_dir)) or (not os.listdir(test_dir)):
    raise Exception('The given test images directory does not exist or it does not contain any images')

if not os.path.isdir(results_dir):
    os.mkdir(results_dir)


image_files = os.listdir(test_dir)
image_names = [string.split('.')[0] for string in image_files]
psnr_dict = {}

for test_image,image_file in zip(image_names,image_files):
    images = {}
    image = Image.open(os.path.join(test_dir,image_file)).convert('RGB')
    images['GROUND TRUTH'] = image

    if args['zoom']:
        w,h = int(image.width*scale), int(image.height*scale)
        image = image.resize((w,h), resample = Image.BICUBIC)

    else:
        w,h = int(image.width/scale)*scale, int(image.height/scale)*scale
        image = image.resize((w,h), resample=Image.BICUBIC)

        w,h = int(w/scale), int(h/scale)
        image = image.resize((w,h), resample=Image.BICUBIC)
        images['Low Resolution'] = image

        w,h = int(w*scale), int(h*scale)
        image = image.resize((w,h), resample=Image.BICUBIC)

    images['BICUBIC'] = image
    image.save(results_dir+ f'/{test_image}_bicubic.png')

    image = image.convert('YCbCr')
    y, Cb , Cr = image.split()

    toTensor = transforms.ToTensor()
    input_image = toTensor(y).view(1, -1,y.height, y.width).to(device)

    model = torch.load(args['model'], map_location=map_location)
    
    model.eval()

    with torch.no_grad():
        print(f'[INFO] the network is being set in inference mode...Done')

        startTime = time()
        print(f'[INFO] {test_image} image passed onto the network...')

        output_image = model(input_image)

    psnr = calc_psnr(input_image,output_image)
    print(f'[INFO] PSNR value of {test_image} image: {psnr:.4f} dB')
    psnr_dict[test_image] = psnr

    out_img_y = output_image[0].cpu().detach().numpy()
    out_img_y *= 255.0
    out_img_y = out_img_y.clip(0, 255)
    out_img_y = Image.fromarray(np.uint8(out_img_y[0]), mode='L')

    srcnn_image = Image.merge('YCbCr', [out_img_y, Cb, Cr]).convert('RGB')

    images[f'SRCNN (PSNR: {psnr:.4f} dB)'] = srcnn_image
    srcnn_image.save(results_dir + f'/{test_image}_srcnn.png')

    endTime = time()
    print(f'[INFO] Done reconstructing the {test_image} image in: {(endTime - startTime):.3f}s') 
    print(f'[INFO] reconstructed image stored in {results_dir}')

    fig = plt_images(test_image,images)
    fig.savefig(results_dir + f'/{test_image}_comparison.png' )
    print('---'*25)

psnr_df = pd.DataFrame([psnr_dict], index=['PSNR'])
psnr_df.to_csv(results_dir + f'/psnr_values.csv')
