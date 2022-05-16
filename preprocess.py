from PIL import Image
import os 
import numpy as np
from data_transformations import Remove_Noise, Tilt_Correction, Crop_Image, Add_Pad
from torch.utils import data
from dataset import load_data_set
import torch

filepath = '/media/avcstorage/DadosReais_Skull/'
savepath = '/media/avcstorage/Preprocess/Sym/'


class Dataset(data.Dataset):
    def __init__(self, ids, labels, directory):
        self.directory = directory
        self.images = []
        self.rot_images = []
        self.ids = ids
        self.labels = labels
        #self.remove_noise = Remove_Noise()
        #self.rotate_image = Tilt_Correction()
        #self.crop = Crop_Image()
        #self.padding = Add_Pad()
        self.final_image = []

    def load_images_from_folder(self, id, idx):
        folder = self.directory
        for filename in sorted(os.listdir(f'{folder}{id}'), key=lambda x: int(os.path.splitext(x)[0])):
            if filename.endswith(".png"):
                im = Image.open(f'{folder}{id}/{filename}')
                img = np.array(im, dtype=np.uint8)
                # print(img.shape)  # (512, 512)
                self.images.append(img)
        #img_data = np.array(self.images, dtype=np.uint8)  # list of images (n_images, 512, 512)
        #masked_image = self.remove_noise(img_data, idx)
        #masked_image = np.array(masked_image, dtype=np.uint8)
        #rotated_image = self.rotate_image(masked_image, idx)
        #rotated_image = np.array(rotated_image, dtype=np.uint8)
        #croped_image = self.crop(img_data, idx)
        #self.final_image = self.padding(croped_image, idx)
        #self.final_image = np.array(self.final_image, dtype=np.uint8)
        self.final_image = np.array(self.images, dtype=np.uint8)
        return self.final_image

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, index):
        # Select sample
        id = self.ids[index]
        x = np.array(self.load_images_from_folder(id, index), dtype=np.uint8)  # list of images
        y = self.labels[id]
        return x, y

# Caso a diretoria em savepath não existe, cria-a
if not os.path.exists(savepath):
    os.makedirs(savepath)

# Chama a classe Dataset e tem como argumentos a saída da função load_data_set e a diretoria
# load_data_set retorna: ids_with_label, labels
dataset = Dataset(*load_data_set(filepath), filepath)  # returns x and y

for i, (x, y) in enumerate(dataset):
    print(f"Preprocessing: {i}/{len(dataset)}")
    torch.save(x[i], f'{savepath}/{dataset.ids[i]}')  # x[i]: array de cada uma das imagens
