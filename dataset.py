import os
import torch
from torch.utils import data
import pandas as pd
import torchvision.transforms as trans  # ToTensor(): Adds one dimensition --> (batch_size, 1, 512, 512)
from sklearn.model_selection import train_test_split

savepath = '/media/avcstorage/Preprocess/Sym/'


class Dataset(data.Dataset):
    def __init__(self, ids, labels, directory, transform=None):
        self.ids = ids
        self.labels = labels
        self.directory = directory
        self.transform = transform

    def load_data(self, id):
        image = torch.load(f'{savepath}/{id}')  # pasta onde estão guardados os dados processados
        if self.transform:
            image = (image - image.min()) / (image.max() - image.min())  #MinMaxScaler
            image = self.transform(image)
        return image

    def __len__(self):
        # retorna o tamanho do dataset
        return len(self.ids)

    def __getitem__(self, index):
        # Seleciona uma amostra
        id = self.ids[index]  # Patient index
        x = self.load_data(id)  # List
        # print(x.shape)  # (1, 512, 512)
        y = self.labels[id]  # Label 0 or 1
        return x, y

def load_data_set(directory):
    label_xl = pd.read_excel(io="Assimetria2.xlsx", engine='openpyxl', sheet_name=0, header=[0])
    #ids = [int(file_id.split(".")[0]) for file_id in sorted(os.listdir(directory))]
    ids = [int(file_id) for file_id in sorted(os.listdir(directory))]

    # print("Tamanho do dataset:", len(ids), "pacientes")

    labels = {}
    ids_with_label = []
    count = 0
    count_class_0 = 0
    count_class_1 = 0

    for idx, row in label_xl.iterrows():
        idP = int(row['Slices'])
        count += 1
        if idP in ids:
            ids_with_label.append(idP)  # adiciona o idP à lista ids_with_label
            if row['Simetria'] == 1:
                labels[idP] = 1
                #print("Paciente classe 1:", idP)
                count_class_1 += 1
            else:
                labels[idP] = 0
                #print("Paciente classe 0", idP)
                count_class_0 += 1

    # print("Informação clínica:", count, "pacientes")
    # print(ids_with_label)
    # print("Dados da classe 1:", count_class_1, "\nDados da classe 0:", count_class_0)

    return ids_with_label, labels


train_augment = trans.Compose([trans.ToTensor(), trans.RandomVerticalFlip(p=1), trans.RandomRotation(degrees=(0, 180))])
transformation = trans.Compose([trans.ToTensor()])

def get_data_sets(directory, train_size=0.8, val_size=0.2, test_size=0.2, seed=37):
    ids, labels = load_data_set(directory)
    train, test = train_test_split(ids, test_size=test_size, random_state=seed)  # conjunto teste é 20% do dataset (ids)
    train, val = train_test_split(train, test_size=val_size, random_state=seed)  # conjunto validação é 20% do conjunto de treino
    
    return (
        Dataset(train, labels, directory, transform=transformation),
        Dataset(val, labels, directory, transform=transformation),
        Dataset(test, labels, directory, transform=transformation))
