
import torch
from torch import nn

from model import Model

from utils.language_utils import get_word_emb_arr, line_to_indices

VOCAB_DIR = 'sent140/embs.json'

class ClientModel(Model):
    def __init__(self, seed, lr, seq_len, num_classes, n_hidden, emb_arr=None):
        super().__init__(seed, lr)
        self.seq_len = seq_len
        self.num_classes = num_classes
        self.n_hidden = n_hidden
        _, self.indd, vocab = get_word_emb_arr(VOCAB_DIR)
        self.vocab_size = len(vocab)

        self.word_embedding = nn.Embedding(self.vocab_size + 1, self.n_hidden)
        self.lstm = nn.LSTM(input_size=self.n_hidden, hidden_size=self.n_hidden, num_layers=2, batch_first=True)
        self.fc1 = nn.Linear(self.n_hidden * 2, 128)
        self.pred = nn.Linear(128, self.num_classes)

        self.loss_fn = nn.CrossEntropyLoss()

        super().__post_init__()

    def forward(self, features, labels):
        emb = self.word_embedding(features)
        _, (h0, _) = self.lstm(emb)
        h0 = h0.transpose(0, 1).reshape(-1, 2 * self.n_hidden)
        logits = self.pred(self.fc1(h0))

        loss = self.loss_fn(logits, labels)
        return logits, loss

    def process_x(self, raw_x_batch, max_words=25):
        x_batch = [e[4] for e in raw_x_batch]
        x_batch = [line_to_indices(e, self.indd, max_words) for e in x_batch]
        x_batch = torch.LongTensor(x_batch).cuda()
        return x_batch

    def process_y(self, raw_y_batch):
        y_batch = [int(e) for e in raw_y_batch]
        y_batch = torch.LongTensor(y_batch).cuda()
        return y_batch

