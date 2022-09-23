
#Loading the required libraries.
import os
import h5py
import torch
import random
import numpy as np
from torchvision import utils
import matplotlib.pyplot as plt
from torchvision import transforms as T
from torch.utils.data import Dataset, DataLoader, random_split


class InvalidNetworkInputException(Exception):
    def __init__(
        self,
        message='The number of the training images given as input does not match with the number of labels...'
    ):
        self.message = message
        super().__init__(self.message)


class HDF5Dataset(Dataset):
    '''
    A custom made python class for the source
    SRCNN dataset. 
    '''
    def __init__(self,images_hd5_path, mode='TRAIN', load_to_mem=False, transform_ops=T.ToTensor()):
        '''
        images_hd5_path -> (str) path to the hd5 file 
                                 of the images.
        load_to_mem     ->  boolean to indicate whether to load the 
                            entire dataset to memory.
        transform_ops   ->  (torchvisions.tranforms())
        '''
        super(HDF5Dataset,self).__init__()
        self.load_to_mem = load_to_mem
        self.mode = mode
        self.path = images_hd5_path
        self.lookAt_hd5()
        self.transform = transform_ops

    def lookAt_hd5(self):
        '''
        A method that read and gets information about the 
        hd5 file and loads it to memory if neccesary.
        '''
        with h5py.File(self.path) as f:
            self.image_types = list(f.keys())
            num_types = [len(f[key]) for key in self.image_types]

            if num_types[0] != num_types[1]:
                raise InvalidNetworkInputException()
            self.len = num_types[0]

            if self.load_to_mem and mode =='TRAIN':
                self.images_dict = {}
                print('Loading the entire the dataset into memeory...')
                self.images_dict = self.read_hd5(images_hd5_path)
                for key in keys:
                    self.images_dict[key] = list(f[key])

    def __len__(self):
        '''
        A method that returns the number of samples 
        in the dataset.
        '''
        return self.len

    def __getitem__(self,idx):
        '''
        A method loads and returns a sample (lr,hr) 
        from the dataset at the given index. 
        '''
        if self.mode == 'EVAL':
            idx = str(idx)
            
        if self.load_to_mem:
            lr_image = self.images_dict['lr'][idx][:]
            hr_image = self.images_dict['hr'][idx][:]

        else:
            with h5py.File(self.path) as f:
                lr_image = f['lr'][idx][:]
                hr_image = f['hr'][idx][:]

        images = [lr_image,hr_image]

        for i,image in enumerate(images):

            if not np.logical_and(image>=0, image<=1).all():
                image = image/255.0

            if self.transform:
                image = self.transform(image)
            images[i] = image

        return images[0],images[1]

    def show_dataset(self,num_images=5,figsize=(20,40)):
        '''
        A method that displays the num_images 
        from the dataset.
        '''
        ncols = 2
        nrows = int(num_images)
        idx_list = [random.randint(0,self.len-1) for _ in range(num_images)]

        fig,axes = plt.subplots(nrows=nrows,ncols=ncols,figsize=figsize)

        for i,idx in enumerate(idx_list):
            lr,hr = self.__getitem__(idx)

            for img,ax in zip([lr,hr],axes[i,:].flat):
                img = img.squeeze()
                ax.imshow(img)
                plt.setp(ax.get_xticklabels(), visible=False)
                plt.setp(ax.get_yticklabels(), visible=False)
                ax.tick_params(axis='both', which='both', length=0)

        plt.suptitle('Low Resolution--High Resolution pairs',fontsize=20)
        fig.tight_layout()
        fig.subplots_adjust(top=0.90)

        return fig


