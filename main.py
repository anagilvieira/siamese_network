import torch
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import EarlyStopping
from sklearn import datasets
from model_lightning import ModelLightning
from dataset import get_data_sets
from torch.utils import data
from pytorch_lightning.loggers import WandbLogger
from argparse import ArgumentParser
import torchvision.models as models
from torchsummary import summary
from sklearn.model_selection import StratifiedKFold
import torchvision.transforms as trans



directory = '/media/avcstorage/DadosReais_Skull2/'
savepath = '/media/avcstorage/Preprocess/Sym/'


class WrapperDataset:
    def __init__(self, dataset, transform=None):
        self.dataset = dataset
        self.transform = transform

    def __getitem__(self, index):
        image, label = self.dataset[index]
        if self.transform is not None:
            image = trans.ToPILImage()(image)
            image = self.transform(image)
            image = (image - image.min()) / (image.max() - image.min())  #MinMaxScaler
        else:
            image = (image - image.min()) / (image.max() - image.min())  #MinMaxScaler
        return image, label

    def __len__(self):
        return len(self.dataset)


def create_model(hparams):
    if hparams.model in models.MODELS:  # MODELS contida no ficheiro utils.py
        model = models.MODELS[hparams.model]()
        #print(model)
        #print("Batch size:", hparams.batch_size)
        #summary(model, [(1, 512, 512), (1, 512, 512)], dtypes=[torch.float, torch.float], device='cuda') 
    return model


def main(hparams):
    dataset = get_data_sets(directory)
    labels = dataset.labels.values()
    labels = list(labels)

    # ---------- TRAIN-VALIDATION-TEST SPLIT ----------
    #train, val, test = get_data_sets(directory)
    
    #print("\n\nTrain length:", len(train), "\nValidation length:", len(val), "\nTest length:", len(test))
    #print("\nTest dataset:", test.ids)
    
    #train_dataloader, val_dataloader, test_dataloader = (
    #    data.DataLoader(train, batch_size=hparams.batch_size, num_workers=12, drop_last=False, pin_memory=True),
    #    data.DataLoader(val, batch_size=hparams.batch_size, num_workers=12),
    #    data.DataLoader(test, batch_size=1, num_workers=12))

    
    # ---------- K-FOLD CROSS VALIDATION ----------
    train_augment = trans.Compose([trans.GaussianBlur(kernel_size=(5, 9), sigma=(0.1, 5)), trans.ToTensor()])
    kfold = StratifiedKFold(n_splits=5, shuffle=False, random_state=None)
    for fold, (train_ids, test_ids) in enumerate(kfold.split(dataset, labels)):
        train_subsampler = torch.utils.data.SubsetRandomSampler(train_ids)
        print("\n Train dataset:", train_subsampler.indices)
        test_subsampler = torch.utils.data.SubsetRandomSampler(test_ids)
        print("\nTest dataset:", test_subsampler.indices)
        
        # Define data loaders for training and testing data in this fold
        train_dataloader = torch.utils.data.DataLoader(
                        WrapperDataset(dataset, transform=train_augment),
                        batch_size=hparams.batch_size,
                        num_workers=12, 
                        sampler=train_subsampler
                        )
        
        test_dataloader = torch.utils.data.DataLoader(
                        WrapperDataset(dataset, transform=None),
                        batch_size=1,
                        num_workers=12, 
                        sampler=test_subsampler)
        
        #early_stop_callback = EarlyStopping(monitor='val_loss', min_delta=0.00, patience=10, verbose=False, mode='min')
    
        wandb_logger = WandbLogger(project=hparams.wb_name, log_model=True, save_dir='/media/avcstorage/Preprocess/')
        trainer = Trainer.from_argparse_args(args, logger=wandb_logger, max_epochs=200, gpus='1')
        model = ModelLightning(model=create_model(hparams), hparams=hparams)
        trainer.fit(model, train_dataloader=train_dataloader)
        trainer.test(model, test_dataloaders=test_dataloader)
        #trainer.fit(model, train_dataloader, val_dataloader)
        #trainer.test(model, test_dataloaders=test_dataloader)


if __name__ == '__main__':

    parser = ArgumentParser()
    parser = Trainer.add_argparse_args(parser)
    parser.add_argument('--lr', type=float, default=0.00001)  # learning rate
    parser.add_argument('--batch_size', type=int, default=32)
    # weight decay adds the L2 norm to the loss function (regularization)
    parser.add_argument('--wd', type=float, default=0.0005)  # weight decay --> l2 regularizer
    parser.add_argument('--momentum', type=float, default=0.9)  # SGD optimizer
    parser.add_argument('--optimizer', type=str, default="ADAM")
    parser.add_argument('--wb_name', type=str, default="Runs_17_05", help='Project name for WandB')
    parser.add_argument('--model', type=str, default="SiameseNetwork", help='Model to use')

    args = parser.parse_args()
    main(args)