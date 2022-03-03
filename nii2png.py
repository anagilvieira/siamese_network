import os 
import numpy as np
import nibabel as nib
import imageio
import torch
import scipy.ndimage as ndi
from data_transformations import Window, Normalize
from torch.utils import data


niifilepath = 'C:/Users/Ana Beatriz Vieira/Desktop/TAC_111443/'
savepath = 'C:/Users/Ana Beatriz Vieira/Desktop/Preprocess/'


class Nii2Png(data.Dataset):
    def __init__(self, directory):
        self.directory = directory
        self.window = Window()
        self.normalize = Normalize()
        self.ids = []

    def load_nii(self):
        folder = self.directory
        self.ids = [os.path.splitext(file)[0] for file in os.listdir(folder)]
        self.ids = np.array(self.ids)
        for i in range(len(self.ids)):
            for filename in os.listdir(f'{folder}{self.ids[i]}'):  # percorre as pastas na diretoria
                print(f'{folder}{self.ids[i]}')
                if filename.endswith(".nii"):  # garante que só lê os ficheiros .nii
                    n = nib.load(f'{folder}{self.ids[i]}/{filename}')  # carrega o ficheiro
                    y = n.get_fdata()  # transforma em array
                    y = torch.from_numpy(y).float()  # cria um tensor
                    y = self.window(y, 40, 80)
                    # y = self.normalize(y)
                    y = ndi.rotate(y, 180, reshape=True)
                    y = y.T  # depth, height(y), width(x)
                    # y = y.unsqueeze(0)
                    pngpath = os.path.join(savepath, self.ids[i])
                    if not os.path.exists(pngpath):
                        os.makedirs(pngpath)
                    for k in range(y.shape[0]):
                        silce = y[k, :, :]
                        imageio.imwrite(os.path.join(pngpath, '{}.png'.format(k)), silce)
        return y


volume = Nii2Png(niifilepath)
volume.load_nii()

