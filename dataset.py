import os
import torch
from torch.utils import data
import pandas as pd
import torchvision.transforms as trans
from sklearn.model_selection import train_test_split

savepath = '/media/avcstorage/Preprocess/Sym/'


class Dataset(data.Dataset):
    def __init__(self, ids, labels, directory, transform=None):
        self.ids = ids
        self.labels = labels
        self.directory = directory
        self.transform = transform

    def load_data(self, id):
        image = torch.load(f'{savepath}/{id}')
        if self.transform:
            image = self.transform(image)
        return image

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, index):
        id = self.ids[index]  # Patient index
        x = self.load_data(id)  # List
        # print(x.shape)  # (1, 512, 512)
        y = self.labels[id]  # Label 0 or 1
        return x, y

def load_data_set(directory):
    label_xl = pd.read_excel(io="Assimetria2_v2.xlsx", engine='openpyxl', sheet_name=0, header=[0])
    ids = [int(file_id) for file_id in sorted(os.listdir(directory))]

    # print("Dataset lenght:", len(ids), "patients")

    labels = {}
    ids_with_label = []
    count = 0
    count_class_0 = 0
    count_class_1 = 0

    for idx, row in label_xl.iterrows():
        idP = int(row['Slices'])
        count += 1
        if idP in ids:
            ids_with_label.append(idP)
            if row['Simetria'] == 1:
                labels[idP] = 1
                #print("Patients from class 1:", idP)
                count_class_1 += 1
            else:
                labels[idP] = 0
                #print("Patients from class 0", idP)
                count_class_0 += 1

    # print("Clinical information:", count, "patients")
    # print(ids_with_label)
    print("Class 1:", count_class_1, "\nClass 0:", count_class_0)

    return ids_with_label, labels


# Data Agumentation 1:
#train_augment = trans.Compose([trans.ToTensor(), trans.RandomVerticalFlip(p=1), trans.RandomRotation(degrees=(0, 180))])

# Data Augmentation 2:
#train_augment = trans.Compose([trans.ToTensor(), trans.GaussianBlur(kernel_size=(5, 9), sigma=(0.1, 5))])

# Data Augmentation 3:
#train_augment = trans.Compose([trans.ToTensor(), trans.RandomVerticalFlip(p=1)])

# Data Augmentation 4:
#train_augment = trans.Compose([trans.ToTensor(), trans.RandomRotation(degrees=(0, 180))])

transformation = trans.Compose([trans.ToTensor()])

def get_data_sets(directory, train_size=0.8, val_size=0.2, test_size=0.2, seed=None):
    ids, labels = load_data_set(directory)
    
    # ---------- TRAIN-VALIDATION-TEST SPLIT ----------
    #train, test = train_test_split(ids, test_size=test_size, random_state=seed)  
    #train, val = train_test_split(train, test_size=val_size, random_state=seed)
    
    #return (
    #    Dataset(train, labels, directory, transform=train_augment),
    #    Dataset(val, labels, directory, transform=transformation),
    #    Dataset(test, labels, directory, transform=transformation))
    
    # ---------- K-FOLD CROSS VALIDATION ----------
    return(Dataset(ids, labels, directory, transform=transformation))
