import pytorch_lightning as pl
import torch
import wandb
import numpy as np
from data_transformations import Flip_Brain
import sklearn


class ModelLightning(pl.LightningModule):

    def __init__(self, model, hparams):
        super(ModelLightning, self).__init__()
        self.model = model
        self.loss = torch.nn.CrossEntropyLoss()  # contains a softmax function inside of the loss function  
        #self.loss = torch.nn.BCEWithLogitsLoss()  # contains a sigmoid function inside of the loss function
        self.hparams = hparams


    def forward(self, data, data2):
        return self.model(data, data2)


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
        data = data.cuda()
        data.requires_grad_(True)
        data2 = Flip_Brain(data)
        data2 = np.array(data2)
        data2 = torch.tensor(data2, device='cuda', requires_grad=True).unsqueeze(1)
        output = self.forward(data, data2)
        pred = output.argmax(dim=1, keepdim=True).view(-1)  # label predicted
        return {"loss": self.loss(output, target), "train_acc": sklearn.metrics.accuracy_score(target.cpu(), pred.cpu())}

    def training_epoch_end(self, outputs):
        accuracy = sum(x["train_acc"] for x in outputs) / len(outputs)
        loss = sum(x["loss"] for x in outputs) / len(outputs)
        return {"log": {"loss": loss, "train_acc": accuracy}}
    
    """
    def validation_step(self, batch, batch_nb):
        data, target = batch
        data = data.cuda()
        data.requires_grad_(True)
        data2 = Flip_Brain(data)
        data2 = np.array(data2)
        data2 = torch.tensor(data2, device='cuda', requires_grad=True).unsqueeze(1)
        output = self.forward(data, data2)
        pred = output.argmax(dim=1, keepdim=True).view(-1)
        return {"val_acc": sklearn.metrics.accuracy_score(pred.cpu(), target.cpu()),
                "val_loss": self.loss(output, target),
                "val_f1": sklearn.metrics.f1_score(pred.cpu(), target.cpu()), "target": target, "output": output}

    def validation_epoch_end(self, outputs):
        accuracy = sum(x["val_acc"] for x in outputs) / len(outputs)
        loss = sum(x["val_loss"] for x in outputs) / len(outputs)
        f1 = sum(x["val_f1"] for x in outputs) / len(outputs)
        return {"log": {"val_acc": accuracy, "val_loss": loss, "val_f1": f1}, "val_loss": loss}
    """
    
    def test_step(self, batch, batch_nb):
        data, target = batch
        data = data.cuda()
        data.requires_grad_(True)
        data2 = Flip_Brain(data)
        data2 = np.array(data2)
        data2 = torch.tensor(data2, device='cuda', requires_grad=True).unsqueeze(1)
        output = self.forward(data, data2)
        pred = output.argmax(dim=1, keepdim=True).view(-1)
        #print("\nTrue label:", target, "\nPredicted label:", pred)
        return {"target": target, "output": output}
        
    def test_epoch_end(self, outputs):
        target = torch.cat([x["target"] for x in outputs], dim=0)
        output = torch.cat([x["output"] for x in outputs], dim=0)
        pred = output.argmax(dim=1, keepdim=True).view(-1)
        self.logger.experiment.log({
            #"Test Confusion Matrix": wandb.sklearn.plot_confusion_matrix(target.cpu(), pred.cpu(), [0, 1]),
            #"Test ROC Curve": wandb.plots.ROC(target.cpu(), output.cpu(), [0, 1]), })  # depends on the wandb version
            "Test Confusion Matrix": wandb.sklearn.plot_confusion_matrix(target.cpu(), pred.cpu(), [0, 1]),
            "Test ROC Curve": wandb.plot.roc_curve(target.cpu(), output.cpu(), [0, 1])})  # depends on the wandb version
        return {"log": {
            "test_acc": sklearn.metrics.accuracy_score(target.cpu(), pred.cpu()),
            "f1_score": sklearn.metrics.f1_score(target.cpu(), pred.cpu()),
            "precision": sklearn.metrics.precision_score(target.cpu(), pred.cpu()),
            "recall": sklearn.metrics.recall_score(target.cpu(), pred.cpu()),
            "ROC_AUC": sklearn.metrics.roc_auc_score(target.cpu(), pred.cpu())}}
