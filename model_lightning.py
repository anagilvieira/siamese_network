import pytorch_lightning as pl
import torchmetrics.functional as plm
import torch
import wandb
import torch.nn.functional as F
import numpy as np
from data_transformations import Half_Brain, Padding_Half
from torch.utils.tensorboard import SummaryWriter
import torchvision
from openpyxl import Workbook
from grad_functions import visualize_cam
from gradcam import GradCAM, GradCAMpp
from torchvision.utils import make_grid, save_image
import os


def Hemispheres(data):
    croped_left, croped_right = Half_Brain(data)
    image_left, image_right = Padding_Half(croped_left, croped_right)
    image_left = np.array(image_left)
    image_left = torch.tensor(image_left, device='cuda', requires_grad=True).unsqueeze(1)
    image_right = np.array(image_right)
    image_right = torch.tensor(image_right, device='cuda', requires_grad=True).unsqueeze(1)
    image_right = torch.fliplr(image_right)
    return (image_left, image_right)


class ModelLightning(pl.LightningModule):

    def __init__(self, model, hparams):
        super(ModelLightning, self).__init__()
        self.model = model
        self.loss = torch.nn.CrossEntropyLoss()  # contains a softmax function inside of the loss function  
        # self.loss = torch.nn.BCEWithLogitsLoss()  # contains a sigmoid function inside of the loss function
        self.hparams = hparams

    def forward(self, data):
        # image_merged = torch.cat((left, right), 1)  # Used for ResNet - each channel has one hemisphere
        # print(data.shape)
        return self.model(data)

    def configure_optimizers(self):
        if self.hparams.optimizer == "SGD":
            optimizer = torch.optim.SGD(self.model.parameters(), lr=self.hparams.lr, momentum=self.hparams.momentum,
                                        weight_decay=self.hparams.wd)
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min')
            return {'optimizer': optimizer, 'scheduler': scheduler, 'metric': 'val_acc'}
        elif self.hparams.optimizer == "ADAM":
            return torch.optim.Adam(self.model.parameters(), lr=self.hparams.lr, weight_decay=self.hparams.wd)

    def training_step(self, batch, batch_nb):
        data, target = batch
        #image_left, image_right = Hemispheres(data)
        #output = self.forward(image_left, image_right)  # Predictions from model (probabilities)
        output = self.forward(data)  # Used for ResNet
        pred = output.argmax(dim=1, keepdim=True).view(-1)  # label predicted
        return {"loss": self.loss(output, target), "train_acc": plm.accuracy(pred, target, num_classes=2)}

    def training_epoch_end(self, outputs):
        accuracy = sum(x["train_acc"] for x in outputs) / len(outputs)
        loss = sum(x["loss"] for x in outputs) / len(outputs)
        return {"log": {"loss": loss, "train_acc": accuracy}}

    def validation_step(self, batch, batch_nb):
        data, target = batch
        #image_left, image_right = Hemispheres(data)
        #output = self.forward(image_left, image_right)  # Used for Siamese Net
        output = self.forward(data)  # Used for ResNet
        pred = output.argmax(dim=1, keepdim=True).view(-1)
        return {"val_acc": plm.accuracy(pred, target, num_classes=2),
                "val_loss": self.loss(output, target),
                "val_f1": plm.f1(pred, target, num_classes=2), "target": target, "output": output}

    def validation_epoch_end(self, outputs):
        accuracy = sum(x["val_acc"] for x in outputs) / len(outputs)
        loss = sum(x["val_loss"] for x in outputs) / len(outputs)
        f1 = sum(x["val_f1"] for x in outputs) / len(outputs)
        return {"log": {"val_acc": accuracy, "val_loss": loss, "val_f1": f1}, "val_loss": loss}

    def test_step(self, batch, batch_nb):
        data, target = batch
        #data = data.clone().detach().requires_grad_(True)
        #image_left, image_right = Hemispheres(data)
        #output = self.forward(image_left, image_right)  # Used for Siamese Net
        output = self.forward(data)  # Used for ResNet
        pred = output.argmax(dim=1, keepdim=True).view(-1)

        """
        cam_dict = dict()
        resnet_model_dict = dict(type='resnet', arch=self.model, layer_name='layer4', input_size=(512, 512))
        resnet_gradcam = GradCAM(resnet_model_dict, True)
        resnet_gradcampp = GradCAMpp(resnet_model_dict, True)
        cam_dict['resnet'] = [resnet_gradcam, resnet_gradcampp]

        #siamese_model_dict = dict(type='siamese', arch=self.model, layer_name='features_11', input_size=(512, 512))
        #siamese_gradcam = GradCAM(siamese_model_dict, True)
        #siamese_gradcampp = GradCAMpp(siamese_model_dict, True)
        #cam_dict['siamese'] = [siamese_gradcam, siamese_gradcampp]
        
        output_dir = '/media/avcstorage/Preprocess/GradCam/Results_1/'
        os.makedirs(output_dir, exist_ok=True)
        images = []
        for gradcam, gradcam_pp in cam_dict.values():
            mask, _ = gradcam(data, class_idx=0)  # 1x1x512x512
            heatmap, result = visualize_cam(mask, data)  # 3x512x512
            mask_pp, _ = gradcam_pp(data, class_idx=0)
            heatmap_pp, result_pp = visualize_cam(mask_pp, data)
            images.append(torch.stack([batch.repeat(1,3,1,1).squeeze(), result, result_pp], 0))
        images = make_grid(torch.cat(images, 0), nrow=3)
        output_path = os.path.join(output_dir, f'output{batch_nb}.png')
        save_image(images, output_path)
        """
        
        #writer = SummaryWriter()
        #for i in range(len(data)):
        #    image = (data[i].detach().unsqueeze(1))
        #    writer.add_images('Image ' + str(i), image)
        #writer.close()
        
        #metric_test = Similairty(data)

        f = open("info.txt", "a")
        f.write("\nTrue label: ")
        f.write(str(target))
        f.write("\nPredicted label: ")
        f.write(str(pred))
        f.close
        
        #workbook = Workbook()
        #sheet = workbook.active
        #rows = ((target, pred))
        #for row in rows:
        #    sheet.append(row)
        
        #workbook.save("Runs.xlsx")

        #print("\nTrue label:", target, "\nPredicted label:", pred)
        return {"target": target, "output": output}

    def test_epoch_end(self, outputs):
        target = torch.cat([x["target"] for x in outputs], dim=0)
        output = torch.cat([x["output"] for x in outputs], dim=0)
        pred = output.argmax(dim=1, keepdim=True).view(-1)
        self.logger.experiment.log({
            "Test Confusion Matrix": wandb.sklearn.plot_confusion_matrix(target.cpu(), pred.cpu(), [0, 1]),
            "Test ROC Curve": wandb.plots.ROC(target.cpu(), output.cpu(), [0, 1]), })
        return {"log": {
            "test_acc": plm.accuracy(pred, target, num_classes=2),
            "f1_score": plm.f1(pred, target, num_classes=2),
            "precision": plm.precision(pred, target, num_classes=2),
            "recall": plm.recall(pred, target, num_classes=2)}}
