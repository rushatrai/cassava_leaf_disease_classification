import torch
from torch import nn
from torch.nn import functional as F
import pytorch_lightning as pl


class VGG16(pl.LightningModule):
    def __init__(self, config):
        super().__init__()
        self.dims = config['img_dims']
        self.learning_rate = config['lr']

        self.save_hyperparameters()

        self.train_accuracy = pl.metrics.Accuracy()
        self.val_accuracy = pl.metrics.Accuracy()

        # conv layers
        self.conv_layers = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=64,
                      kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(64),
            nn.Conv2d(in_channels=64, out_channels=64,
                      kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(in_channels=64, out_channels=128,
                      kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(128),
            nn.Conv2d(in_channels=128, out_channels=128,
                      kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(128),
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(in_channels=128, out_channels=256,
                      kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(256),
            nn.Conv2d(in_channels=256, out_channels=256,
                      kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(256),
            nn.Conv2d(in_channels=256, out_channels=256,
                      kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(256),
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(in_channels=256, out_channels=512,
                      kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(512),
            nn.Conv2d(in_channels=512, out_channels=512,
                      kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(512),
            nn.Conv2d(in_channels=512, out_channels=512,
                      kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(512),
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(in_channels=512, out_channels=512,
                      kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(512),
            nn.Conv2d(in_channels=512, out_channels=512,
                      kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(512),
            nn.Conv2d(in_channels=512, out_channels=512,
                      kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(512),
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Flatten(),
        )

        # passes dummy x matrix to find the input size of the fc layer
        x = torch.randn(1, 3, self.dims[0], self.dims[1])
        self._to_linear = None
        self.forward(x)

        # fc layers
        self.fc_layers = nn.Sequential(
            nn.Linear(in_features=self._to_linear, out_features=4096),
            nn.ReLU(inplace=True),
            nn.Dropout(),

            nn.Linear(in_features=4096, out_features=4096),
            nn.ReLU(inplace=True),

            nn.Linear(4096, 5),
        )

    def forward(self, x):
        x = self.conv_layers(x)

        if self._to_linear is None:
            # does not run fc layer if input size is not determined yet
            self._to_linear = x.shape[1]
        else:
            x = self.fc_layers(x)
        return x

    def loss_function(self, logits, y):
        return F.cross_entropy(logits, y)

    def training_step(self, train_batch, batch_idx):
        x, y = train_batch
        logits = self.forward(x)
        preds = F.softmax(logits, dim=1)

        train_loss = self.loss_function(logits, y)
        train_acc = self.train_accuracy(preds, y)

        self.log('train_loss', train_loss, on_step=True, on_epoch=True)
        self.log('train_acc', train_acc, on_step=True, on_epoch=True)

        return train_loss

    def validation_step(self, val_batch, batch_idx):
        x, y = val_batch
        logits = self.forward(x)
        preds = F.softmax(logits, dim=1)

        val_loss = self.loss_function(logits, y)
        val_acc = self.val_accuracy(preds, y)

        self.log('val_loss', val_loss, on_step=True, on_epoch=True)
        self.log('val_acc', val_acc, on_step=True, on_epoch=True)
    
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=0.2, patience=5, eps=1e-6)
        return {'optimizer': optimizer, 'lr_scheduler': scheduler, 'monitor': 'val_loss_epoch'}
