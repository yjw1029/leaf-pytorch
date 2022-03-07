
import torch
from torch import nn

from model import Model

from utils.language_utils import letter_to_vec, word_to_indices

VOCAB_DIR = 'sent140/embs.json'

class ClientModel(Model):
    def __init__(self, seed, lr, seq_len, num_classes, n_hidden, emb_arr=None):
        super().__init__(seed, lr)
        self.seq_len = seq_len
        self.num_classes = num_classes
        self.n_hidden = n_hidden

        self.word_embedding = nn.Embedding(self.num_classes, 8)
        self.lstm = nn.LSTM(input_size=8, hidden_size=self.n_hidden, num_layers=2, batch_first=True)
        self.pred = nn.Linear(self.n_hidden * 2, self.num_classes)

        self.loss_fn = nn.CrossEntropyLoss()

        super().__post_init__()

    def forward(self, features, labels):
        emb = self.word_embedding(features)
        _, (h0, _) = self.lstm(emb)
        h0 = h0.transpose(0, 1).reshape(-1, 2 * self.n_hidden)
        logits = self.pred(h0)

        loss = self.loss_fn(logits, labels)
        return logits, loss

    def process_x(self, raw_x_batch, max_words=25):
        x_batch = [word_to_indices(word) for word in raw_x_batch]
        x_batch = torch.LongTensor(x_batch).cuda()
        return x_batch

    def process_y(self, raw_y_batch):
        y_batch = [letter_to_vec(c) for c in raw_y_batch]
        y_batch = torch.LongTensor(y_batch).cuda()
        return y_batch

