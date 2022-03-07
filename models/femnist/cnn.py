import torch
from torch import nn 

from model import Model
IMAGE_SIZE = 28

class ClientModel(Model):
    def __init__(self, seed, lr, num_classes):
        super().__init__(seed, lr)
        self.num_classes = num_classes
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=32, kernel_size=(5, 5), padding="same")
        self.relu = nn.ReLU()
        self.pool1 = nn.MaxPool2d(kernel_size=(2,2), stride=2)
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=(5, 5), padding="same")
        self.pool2 = nn.MaxPool2d(kernel_size=(2,2), stride=2)
        self.l1 = nn.Linear(7 * 7 * 64, 2048)
        self.l2 = nn.Linear(2048, self.num_classes)

        self.loss_fn = nn.CrossEntropyLoss()
        
        super().__post_init__()

    def forward(self, features, labels):
        input_layer = features.reshape(-1, 1, IMAGE_SIZE, IMAGE_SIZE)
        conv1 = self.relu(self.conv1(input_layer))
        pool1 = self.pool1(conv1)
        conv2 = self.relu(self.conv2(pool1))
        pool2 = self.pool2(conv2)
        pool2_flat = pool2.reshape(-1, 7 * 7 * 64)

        dense = self.relu(self.l1(pool2_flat))
        logits = self.l2(dense)

        loss = self.loss_fn(logits, labels)
        return logits, loss

    def process_x(self, raw_x_batch):
        return torch.FloatTensor(raw_x_batch).cuda()

    def process_y(self, raw_y_batch):
        return torch.LongTensor(raw_y_batch).cuda()


