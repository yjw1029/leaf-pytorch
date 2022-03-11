import pickle
import numpy as np
import collections

import torch
from torch import nn

from model import Model


VOCABULARY_PATH = '../data/reddit/vocab/reddit_vocab.pck'

def acc_fn(y_true, y_hat, unk_symbol, pad_symbol):
    y_hat = torch.argmax(y_hat, dim=-1)
    unk_cnt = torch.sum((y_true == unk_symbol) & (y_true == y_hat))
    pad_cnt = torch.sum((y_true == pad_symbol) & (y_true == y_hat))
    hit = torch.sum(y_true == y_hat)

    return hit - unk_cnt - pad_cnt


class ClientModel(Model):
    def __init__(self, seed, lr, seq_len, n_hidden, num_layers, 
        keep_prob=1.0, max_grad_norm=5, init_scale=0.1):
        super().__init__(seed, lr)
        self.seq_len = seq_len
        self.n_hidden = n_hidden
        self.num_layers = num_layers
        self.keep_prob = keep_prob
        self.max_grad_norm = max_grad_norm

        # initialize vocabulary
        self.vocab, self.vocab_size, self.unk_symbol, self.pad_symbol = self.load_vocab()

        self.word_embedding = nn.Embedding(self.vocab_size, self.n_hidden)
        self.lstm = nn.LSTM(input_size=self.n_hidden, hidden_size=self.n_hidden, num_layers=self.num_layers,
            batch_first=True, dropout=1-self.keep_prob)
        self.pred = nn.Linear(self.n_hidden, self.vocab_size)

        self.loss_fn = nn.CrossEntropyLoss(ignore_index=self.pad_symbol)

        super().__post_init__()

    def forward(self, features, labels, sequence_length_ph, sequence_mask_ph):
        batch_size = features.size(0)

        inputs = self.word_embedding(features)
        packed_inputs = nn.utils.rnn.pack_padded_sequence(inputs, sequence_length_ph,
            batch_first=True, enforce_sorted=False)
        output, _ = self.lstm(packed_inputs)
        output, _ = nn.utils.rnn.pad_packed_sequence(output, batch_first=True, total_length=self.seq_len)
        logits = self.pred(output) 

        reshaped_logits = logits.reshape(batch_size * self.seq_len, self.vocab_size)
        reshaped_labels = labels.reshape(-1)

        loss = self.loss_fn(reshaped_logits, reshaped_labels)
        return logits, loss

    def process_x(self, raw_x_batch):
        tokens = self._tokens_to_ids([s for s in raw_x_batch])
        lengths = np.sum(tokens != self.pad_symbol, axis=1)
        return tokens, lengths

    def process_y(self, raw_y_batch):
        tokens = self._tokens_to_ids([s for s in raw_y_batch])
        return tokens

    def _tokens_to_ids(self, raw_batch):
        def tokens_to_word_ids(tokens, vocab):
            return [vocab[word] for word in tokens]

        to_ret = [tokens_to_word_ids(seq, self.vocab) for seq in raw_batch]
        return np.array(to_ret)

    def batch_data(self, data, batch_size):
        data_x = data['x']
        data_y = data['y']

        perm = np.random.permutation(len(data['x']))
        data_x = [data_x[i] for i in perm]
        data_y = [data_y[i] for i in perm]

        # flatten lists
        def flatten_lists(data_x_by_comment, data_y_by_comment):
            data_x_by_seq, data_y_by_seq, mask_by_seq = [], [], []
            for c, l in zip(data_x_by_comment, data_y_by_comment):
                data_x_by_seq.extend(c)
                data_y_by_seq.extend(l['target_tokens'])
                mask_by_seq.extend(l['count_tokens'])

            if len(data_x_by_seq) % batch_size != 0:
                dummy_tokens = [self.pad_symbol for _ in range(self.seq_len)]
                dummy_mask = [0 for _ in range(self.seq_len)]
                num_dummy = batch_size - len(data_x_by_seq) % batch_size

                data_x_by_seq.extend([dummy_tokens for _ in range(num_dummy)])
                data_y_by_seq.extend([dummy_tokens for _ in range(num_dummy)])
                mask_by_seq.extend([dummy_mask for _ in range(num_dummy)])

            return data_x_by_seq, data_y_by_seq, mask_by_seq
        
        data_x, data_y, data_mask = flatten_lists(data_x, data_y)

        for i in range(0, len(data_x), batch_size):
            batched_x = data_x[i:i+batch_size]
            batched_y = data_y[i:i+batch_size]
            batched_mask = data_mask[i:i+batch_size]

            input_data, input_lengths = self.process_x(batched_x)
            target_data = self.process_y(batched_y)

            input_data = torch.LongTensor(input_data).cuda()
            target_data = torch.LongTensor(target_data).cuda()
            input_lengths = torch.LongTensor(input_lengths)
            batched_mask = torch.FloatTensor(batched_mask).cuda()

            yield (input_data, target_data, input_lengths, batched_mask)

    def load_vocab(self):
        vocab_file = pickle.load(open(VOCABULARY_PATH, 'rb'))
        vocab = collections.defaultdict(lambda: vocab_file['unk_symbol'])
        vocab.update(vocab_file['vocab'])

        return vocab, vocab_file['size'], vocab_file['unk_symbol'], vocab_file['pad_symbol']

    def test(self, data, batch_size=5):
        tot_acc, tot_samples = 0, 0
        tot_loss, tot_batches = 0, 0

        self.eval()
        with torch.no_grad():
            for input_data, target_data, input_lengths, input_mask in self.batch_data(data, batch_size):
                logits, loss = self.forward(input_data, target_data, input_lengths, input_mask)
                
                tot_acc += acc_fn(target_data, logits, unk_symbol=self.unk_symbol, pad_symbol=self.pad_symbol).detach().cpu().numpy()
                tot_samples += np.sum(input_lengths.detach().cpu().numpy())

                tot_loss += loss.detach().cpu().numpy()
                tot_batches += 1

        acc = float(tot_acc) / tot_samples # this top 1 accuracy considers every pred. of unknown and padding as wrong
        loss = tot_loss / tot_batches # the loss is already averaged over samples
        return {'accuracy': acc, 'loss': loss}

    def run_epoch(self, data, batch_size=5):
        self.train()
        for input_data, target_data, input_lengths, input_mask in self.batch_data(data, batch_size):
            self.optimizer.zero_grad()

            logits, loss = self.forward(input_data, target_data, input_lengths, input_mask)
            loss.backward()
            self.optimizer.step()
