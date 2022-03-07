import os
import numpy as np
from PIL import Image

import torch
from torch import nn

from model import Model

IMAGE_SIZE = 84
IMAGES_DIR = os.path.join('..', 'data', 'celeba', 'data', 'raw', 'img_align_celeba')

class ConvNet(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.conv = nn.Conv2d(in_channels=in_channels, out_channels=32, kernel_size=3, padding="same")
        self.bn = nn.BatchNorm2d(num_features=32)
        self.max_pooling = nn.MaxPool2d(kernel_size=(2,2), stride=2)
        self.relu = nn.ReLU()

    def forward(self, features):
        out = self.conv(features)
        out = self.bn(out)
        out = self.max_pooling(out)
        out = self.relu(out)
        return out

class ClientModel(Model):
    def __init__(self, seed, lr, num_classes):
        super().__init__(seed, lr)
        self.num_classes = num_classes
        self.conv_list = nn.ModuleList([ConvNet(3), ConvNet(32), ConvNet(32), ConvNet(32)])
        self.dense = nn.Linear(32 * 5 * 5, self.num_classes)

        self.loss_fn = nn.CrossEntropyLoss()
        super().__post_init__()

    def forward(self, features, labels):
        out = features.reshape(-1, IMAGE_SIZE, IMAGE_SIZE, 3).transpose(1, 3)
        for i in range(len(self.conv_list)):
            out = self.conv_list[i](out)
        logits = self.dense(out.reshape(-1, 32 * 5 * 5))
        loss = self.loss_fn(logits, labels)
        return logits, loss

    def process_x(self, raw_x_batch):
        x_batch = [self._load_image(i) for i in raw_x_batch]
        x_batch = torch.FloatTensor(x_batch).cuda()
        return x_batch

    def process_y(self, raw_y_batch):
        return torch.LongTensor(raw_y_batch).cuda()

    def _load_image(self, img_name):
        img = Image.open(os.path.join(IMAGES_DIR, img_name))
        img = img.resize((IMAGE_SIZE, IMAGE_SIZE)).convert('RGB')
        return np.array(img)