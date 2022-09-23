import argparse

import os
import torch
import torch.nn as nn
import torch.optim as optim

import shutil
import numpy as np
import pandas as pd
from tqdm import tqdm
from time import time
from math import log10
import matplotlib.pyplot as plt
plt.style.use('seaborn-whitegrid')

from model import SRCNN
from customDatasets import HDF5Dataset
from torch.utils.data import DataLoader

parser = argparse.ArgumentParser()
parser.add_argument('--root_dir', type=str, required=True, help='The root directory of the project')
parser.add_argument('--num_epochs', type=int, required=True, help='The number of epochs for training')
parser.add_argument('--output_dir', type=str, required=True, help='The directory into which the results will be stored')
parser.add_argument('--store_logs', type=bool, default=True, help='Indicates whether to store the training and validation logs')
parser.add_argument('--num_workers', type=int, default=2, help='Parmater for the dataloader')
parser.add_argument('--batch_size', type=int, default=64, help='Parameter for the dataloader')
parser.add_argument('--scale', type=int, default=3, help='The factor by which the images are scaled')
parser.add_argument('--lr', type=float, default=0.001, help='Learning rate for the parameters')
args = vars(parser.parse_args())

modes = ['TRAIN','EVAL']

if not set(modes).issubset(os.listdir(args['root_dir'])):
    raise Exception('The train and validation images folders are not present in the root directory.')

scale = 'x' + str(args['scale'])

args['output_dir'] = os.path.join(args['output_dir'], scale)

if not os.path.isdir(args['output_dir']):
    os.mkdir(args['output_dir'])

ckp_dir = args['root_dir'] + f'model_ckps/{scale}/'
plots_dir  = os.path.join(args['output_dir'], 'plots')
tr_models_dir = os.path.join(args['output_dir'], 'trained_models')

for dir in [ckp_dir,plots_dir, tr_models_dir]:
    if not os.path.isdir(dir):
        os.mkdir(dir)


data_paths = {}
train_data = {}
eval_data =  {}

for mode in modes:
    hdf_files = os.listdir(args['root_dir']+mode)
    hdf_files = [hdf_file for hdf_file in hdf_files if scale in hdf_file ]
    data_paths[mode] = [os.path.join(args['root_dir'],mode,hdf_file) for hdf_file in hdf_files]

for dname,path in zip([scale],data_paths['TRAIN']):
    train_data[dname] = HDF5Dataset(path,mode = 'TRAIN') 
    train_data[dname+'_loader'] = DataLoader(train_data[dname], batch_size=args['batch_size'], shuffle=True, num_workers=args['num_workers'])

for dname,path in zip([scale],data_paths['EVAL']):
    eval_data[dname] = HDF5Dataset(path, mode = 'EVAL')  
    eval_data[dname+'_loader'] = DataLoader(eval_data[dname], batch_size=1, shuffle=False)

train_loader = train_data[scale + '_loader']
eval_loader  = eval_data[scale + '_loader']

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

if DEVICE == 'cuda':
    print(f'[INFO] DEVICE for training: {torch.cuda.get_device_name(0)}')
else:
    print(f'[INFO] DEVICE for training: {DEVICE}')

torch.manual_seed(42)
torch.cuda.manual_seed(42)

num_epochs = args['num_epochs']

model = SRCNN().to(DEVICE)
criterion = nn.MSELoss().to(DEVICE)
optimizer = optim.Adam(model.parameters(), lr=args['lr'])

logs = {key: [] for key in ['train_loss','train_psnr','val_loss','val_psnr']}

print('[INFO] Training the network...')
startTime = time()

ckp_psnr = 0.0

for epoch in tqdm(range(num_epochs)):

    #Training.
    train_loss = 0
    train_psnr = 0

    model.train()
    
    print(f'\n[INFO] EPOCH {epoch+1}/{num_epochs}')


    for batch in tqdm(train_loader,disable=True):
        input_image = batch[0].to(DEVICE)
        label = batch[1].to(DEVICE)
        
        #Forward pass.
        output_image = model(input_image)
        loss = criterion(output_image,label)

        #Zeroing out the gradients.
        optimizer.zero_grad()
        #Backpropagation.
        loss.backward()
        #Update the parameters.
        optimizer.step()

        train_loss += loss.item()
        psnr = 10 * log10(1 / loss.item())
        train_psnr += psnr

    train_loss = train_loss / len(train_loader)
    logs['train_loss'].append(train_loss)
    train_psnr = train_psnr/len(train_loader)
    logs['train_psnr'].append(train_psnr)
    

    print(f'\n[INFO] Evaluating the model trained for {epoch+1} epochs...')
    #Validation.
    val_loss = 0
    val_psnr = 0

    with torch.no_grad():
        model.eval()

        for batch in eval_loader:
            input_image = batch[0].to(DEVICE)
            label = batch[1].to(DEVICE)

            output_image = model(input_image)
            loss = criterion(output_image, label)

            val_loss += loss.item()
            psnr = 10 * log10(1 / loss.item())
            val_psnr += psnr
    
    val_loss = val_loss/len(eval_loader)
    logs['val_loss'].append(val_loss)
    val_psnr = val_psnr/len(eval_loader)
    logs['val_psnr'].append(val_psnr)


    if val_psnr > ckp_psnr:
        ckp_epoch = epoch + 1
        torch.save(model,os.path.join(ckp_dir+f'{scale}_ep{ckp_epoch}.pth'))
    
    
    print(f'[INFO] Printing the results...')
    print('---'*25)
    print('---'*25)
    print(f'Train Loss: {train_loss}')
    print(f'Average Training PSNR: {train_psnr:.4f} dB.')
    print(f'Validation Loss: {val_loss}')
    print(f"Average Validation PSNR: {val_psnr:.4f} dB.")
    print('---'*25)
    print('---'*25)

endTime = time()
print(f'Total time taken for training and validation: {(endTime - startTime):.3f}s')  

if args['store_logs']:
    logs_df = pd.DataFrame(logs)
    logs_dir = ckp_dir+'logs/'

    if not os.path.isdir(logs_dir):
        os.mkdir(logs_dir)

    logs_df.to_csv(logs_dir + scale + f'_ep{num_epochs}_logs.csv')

min_val_loss = min(logs['val_loss'])
best_ep = logs['val_loss'].index(min_val_loss) 
tr_loss = logs['train_loss'][best_ep]
print(f'[INFO]The best model is the one trained for {best_ep+1} epochs and has...')
print(f'\t\t Validation loss: {min_val_loss}')
print(f'\t\t Training loss: {tr_loss}')

best_model =  f'{scale}_ep{best_ep+1}.pth'
shutil.copy(ckp_dir+best_model, tr_models_dir)


plt.figure(figsize=(11.7, 8.7))
plt.plot(logs['train_loss'], color='blue', label='train loss')
plt.plot(logs['val_loss'], color='red', label='validataion loss')
plt.xlabel('Number of Epochs')
plt.ylabel('Loss')
plt.legend()
plt.savefig(plots_dir+ f'/{scale}_loss_{num_epochs}.png')

plt.figure(figsize=(11.7, 8.7))
plt.plot(logs['train_psnr'], color='blue', label='train PSNR dB')
plt.plot(logs['val_psnr'], color='red', label='validataion PSNR dB')
plt.xlabel('Epochs')
plt.ylabel('PSNR (dB)')
plt.legend()
plt.savefig(plots_dir+ f'/{scale}_psnr_{num_epochs}.png')
