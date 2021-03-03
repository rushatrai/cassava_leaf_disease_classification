import torch
from torch import nn
from torch.nn import functional as F
import pytorch_lightning as pl


# ResNet residual blocks
class Block(nn.Module):
    def __init__(self, in_channels, out_channels, identity_downsample=None, stride=1):
        super().__init__()
        self.expansion = 4  # factor by which layer sizes increase in ResNet

        self.conv1 = nn.Conv2d(in_channels, out_channels,
                               kernel_size=1, stride=1, padding=0)
        self.bn1 = nn.BatchNorm2d(out_channels)

        self.conv2 = nn.Conv2d(out_channels, out_channels,
                               kernel_size=3, stride=stride, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)

        self.conv3 = nn.Conv2d(
            out_channels, out_channels*self.expansion, kernel_size=1, stride=1, padding=0)
        self.bn3 = nn.BatchNorm2d(out_channels*self.expansion)

        self.relu = nn.ReLU()

        # conv layer that applies the identity mapping to change shape for next layers
        self.identity_downsample = identity_downsample

    def forward(self, x):
        identity = x

        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)
        x = self.conv3(x)
        x = self.bn3(x)

        # adds id downsample layer if the shape needs to be changed (only first block)
        if self.identity_downsample is not None:
            identity = self.identity_downsample(identity)

        x += identity  # skip connection
        x = self.relu(x)

        return x


# ResNet definition
class ResNet(pl.LightningModule):
    def __init__(self, config, block, layers, image_channels, num_classes):
        super().__init__()
        self.learning_rate = config['lr']
        
        self.save_hyperparameters()

        self.train_accuracy = pl.metrics.Accuracy()
        self.val_accuracy = pl.metrics.Accuracy()

        # initial layers
        self.in_channels = 64
        self.conv1 = nn.Conv2d(
            image_channels, 64, kernel_size=7, stride=2, padding=3)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU()
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        # ResNet layers
        self.layer1 = self._make_layer(
            block, layers[0], out_channels=64, stride=1)
        self.layer2 = self._make_layer(
            block, layers[1], out_channels=128, stride=2)
        self.layer3 = self._make_layer(
            block, layers[2], out_channels=256, stride=2)
        self.layer4 = self._make_layer(
            block, layers[3], out_channels=512, stride=2)

        # avg pools and adapts to desired output size (1,1)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512*4, num_classes)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = x.reshape(x.shape[0], -1)
        x = self.fc(x)

        return x

    def _make_layer(self, block, num_blocks, out_channels, stride):
        identity_downsample = None
        layers = []

        # conv block to downsample between different-sized blocks
        if stride != 1 or self.in_channels != out_channels*4:
            identity_downsample = nn.Sequential(
                nn.Conv2d(self.in_channels, out_channels *
                          4, kernel_size=1, stride=stride),
                nn.BatchNorm2d(out_channels*4),
            )

        # first block always uses the downsample
        layers.append(Block(self.in_channels, out_channels,
                            identity_downsample, stride))
        self.in_channels = out_channels*4

        for i in range(num_blocks - 1):  # other blocks do not downsample
            layers.append(Block(self.in_channels, out_channels))

        return nn.Sequential(*layers)  # unpacks list

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


def ResNet50(config, img_channels=3, num_classes=5):
    return ResNet(config, Block, [3, 4, 6, 3], img_channels, num_classes)
