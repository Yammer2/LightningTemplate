from typing import Any
import lightning.pytorch as pl
from lightning.pytorch.utilities.types import STEP_OUTPUT
from torchmetrics import MetricCollection, Accuracy, Recall, Precision, F1Score

import torch
import torch.nn.functional as F
import pandas as pd
import wandb

from .model import Resblock, ResNet, get_ResNet50

class ResNet_pl(pl.LightningModule):

    def __init__(self,
                 num_class: int,
                 batch_size: int,
                 lr: float,
                 ) -> None:

        super().__init__()

        self.num_class = num_class
        self.batch_size = batch_size
        self.lr = lr

        # network
        self.resnet = get_ResNet50(Resblock, num_class)

        # metrics
        metrics = MetricCollection({
            'acc': Accuracy(task='multiclass', num_classes=self.num_class, average='macro'),
            'f1': F1Score(task='multiclass', num_classes=self.num_class, average='macro'),
            'recall': Recall(task='multiclass', num_classes=self.num_class, average='macro'),
            'precision': Precision(task='multiclass', num_classes=self.num_class, average='macro'),
        })
        self.train_metrics = metrics.clone(prefix='train_')
        self.val_metrics = metrics.clone(prefix='val_')
        self.test_metrics = metrics.clone(prefix='test_')


        # save hparams
        self.save_hyperparameters(ignore=['resnet'])


    ########################################
    # forward
    ########################################
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.resnet(x)


    ########################################
    # training step
    ########################################
    def training_step(self, batch: Any, batch_idx: int) -> Any:
        x, y = batch
        out = self(x)

        # loss
        loss = F.cross_entropy(out, y)
        self.log('train_loss', loss)

        # accuracy
        metrics = self.train_metrics(out, y)
        self.log_dict(metrics)

        return loss


    ########################################
    # validation step
    ########################################
    def validation_step(self, batch: Any, batch_idx: int) -> STEP_OUTPUT:
        x, y = batch
        out = self(x)

        # loss
        loss = F.cross_entropy(out, y)
        self.log('val_loss', loss, prog_bar=True)      # <--- ログに保存

        # metrics
        metrics = self.val_metrics(out, y)
        self.log_dict(metrics)     # <--- 辞書形式で保存もできる

        return loss


    ########################################
    # test step
    ########################################
    def on_test_start(self) -> None:
        self.labels = []
        self.preds = []
        self.corrects = []
        self.images = []
        self.table = wandb.Table(columns=['image', 'label', 'pred', 'correct'])


    def test_step(self, batch: Any, batch_idx: int) -> STEP_OUTPUT:
        x, y = batch
        out = self(x)

        # loss
        loss = F.cross_entropy(out, y)
        self.log('test_loss', loss, prog_bar=True)

        # metrics
        y_pred = torch.argmax(out, dim=1)
        metrics = self.test_metrics(out, y)
        self.log_dict(metrics)

        # wandb
        for i in range(10):
            img = x[i].cpu().numpy().transpose(1,2,0)
            label = int(y[i].cpu())
            pred = int(y_pred[i].cpu())
            self.table.add_data(wandb.Image(img), label, pred, label == pred)

        return loss


    def on_test_end(self) -> None:

        # wandb に保存
        wandb.log({'test_result': self.table})

        # confusion matrix
        class_name = ['airplane', 'automobile', 'bird', 'cat', 'deer','dog', 'frog', 'horse', 'ship', 'truck']
        conf_mat = wandb.plot.confusion_matrix(
            probs=None,
            y_true=self.table.get_column('label'),
            preds=self.table.get_column('pred'),
            class_names=class_name,
            title="Confusion Matrix",
            )
        conf_mat_sk = wandb.sklearn.plot_confusion_matrix(
            self.table.get_column('label'),
            self.table.get_column('pred'),
            class_name,
            )

        wandb.log({"conf_mat": conf_mat, "conf_mat_sk": conf_mat_sk})

    ########################################
    # configure optimizers
    ########################################
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        return optimizer