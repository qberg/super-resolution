import torch
import numpy as np
import matplotlib.pyplot as plt

def plt_images(image_name,image_dict):
    image_types = list(image_dict.keys())
    ncols = len(image_types)
    nrows = 1

    fig,axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(20,5))

    for image_type,ax in zip(image_types,axes.flat):
        image = image_dict[image_type]
        ax.imshow(image)

        plt.setp(ax.get_xticklabels(), visible=False)
        plt.setp(ax.get_yticklabels(), visible=False)
        ax.tick_params(axis='both', which='both', length=0)
        
        ax.set_title(f'{image_type}')

    #fig.set_size_inches(np.array(fig.get_size_inches())*(ncols))
    fig.tight_layout()
    #fig.suptitle(f'Comparison of different SR-methods for {image_name} image', fontsize=20)

    return fig

def calc_psnr(img1, img2):
    return 10. * torch.log10(1. / torch.mean((img1 - img2) ** 2))