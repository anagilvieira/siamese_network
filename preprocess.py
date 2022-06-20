from PIL import Image
import os 
import numpy as np
from torch.utils import data
from dataset import load_data_set
import torch

filepath = '/media/avcstorage/DadosReais_Skull2/'
savepath = '/media/avcstorage/Preprocess/Sym/'


class Dataset(data.Dataset):
    def __init__(self, ids, labels, directory):
        self.directory = directory
        self.images = []
        self.rot_images = []
        self.ids = ids
        self.labels = labels
        self.final_image = []

    def load_images_from_folder(self, id, idx):
        folder = self.directory
        for filename in sorted(os.listdir(f'{folder}{id}'), key=lambda x: int(os.path.splitext(x)[0])):
            if filename.endswith(".png"):
                im = Image.open(f'{folder}{id}/{filename}')
                img = np.array(im, dtype=np.uint8)  # (512, 512)
                self.images.append(img)
        self.final_image = np.array(self.images, dtype=np.uint8)
        return self.final_image

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, index):
        # Select sample
        id = self.ids[index]
        x = np.array(self.load_images_from_folder(id, index), dtype=np.uint8)
        y = self.labels[id]
        return x, y

if not os.path.exists(savepath):
    os.makedirs(savepath)

dataset = Dataset(*load_data_set(filepath), filepath)  # returns x and y

for i, (x, y) in enumerate(dataset):
    print(f"Preprocessing: {i}/{len(dataset)}")
    torch.save(x[i], f'{savepath}/{dataset.ids[i]}')
