from typing import Any
import lightning.pytorch as pl
from lightning.pytorch.utilities.types import STEP_OUTPUT

import torch
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, f1_score
import pandas as pd

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
        y_pred = torch.argmax(out, dim=1)
        acc = accuracy_score(y.cpu().numpy(), y_pred.cpu().numpy())
        f1 = f1_score(y.cpu().numpy(), y_pred.cpu().numpy(), average='macro')
        self.log_dict({'train_acc': acc, 'train_f1': f1})

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

        # accuracy
        y_pred = torch.argmax(out, dim=1)
        acc = accuracy_score(y.cpu().numpy(), y_pred.cpu().numpy())
        f1 = f1_score(y.cpu().numpy(), y_pred.cpu().numpy(), average='macro')
        self.log_dict({'val_acc': acc, 'val_f1': f1})     # <--- 辞書形式で保存もできる

        return loss


    ########################################
    # test step
    ########################################
    def on_test_start(self) -> None:
        self.labels = []
        self.preds = []
        self.corrects = []


    def test_step(self, batch: Any, batch_idx: int) -> STEP_OUTPUT:
        x, y = batch
        out = self(x)

        # loss
        loss = F.cross_entropy(out, y)
        self.log('test_loss', loss, prog_bar=True)

        # accuracy
        y_pred = torch.argmax(out, dim=1)
        acc = accuracy_score(y.cpu().numpy(), y_pred.cpu().numpy())
        f1 = f1_score(y.cpu().numpy(), y_pred.cpu().numpy(), average='macro')
        self.log_dict({'test_acc': acc, 'test_f1': f1})

        # save labels and preds
        self.labels.append(y)
        self.preds.append(y_pred)
        self.corrects.append(y == y_pred)

        return loss


    def on_test_end(self) -> None:
        # save labels and preds
        self.labels = torch.cat(self.labels).tolist()
        self.preds = torch.cat(self.preds).tolist()
        self.corrects = torch.cat(self.corrects).tolist()

        df = pd.DataFrame({'label': self.labels, 'pred': self.preds, 'correct': self.corrects})
        df.to_csv('test_result.csv')


    ########################################
    # configure optimizers
    ########################################
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        return optimizer