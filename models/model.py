import torch
from torch import nn
from torch import optim

from baseline_constants import ACCURACY_KEY

from utils.data import batch_data


def acc_fn(y_true, y_hat):
    y_hat = torch.argmax(y_hat, dim=-1)
    tot = y_true.shape[0]
    hit = torch.sum(y_true == y_hat)
    return hit.data.float() * 1.0 / tot

class Model(nn.Module):
    def __init__(self, seed, lr, optimizer=None):
        super().__init__()
        self.lr = lr
        self.seed = seed
        self.optimizer = optimizer

        # TODO: compute model flops
        self.flops = 0
        self.size = 0

    def get_params(self):
        return self.state_dict()
    
    def set_params(self, state_dict):
        self.load_state_dict(state_dict)

    def __post_init__(self):
        """Init optimizer after init parameters"""
        if self.optimizer is None:
            self.optimizer = optim.SGD(self.parameters(), lr=self.lr)


    def train_model(self, data, num_epochs=1, batch_size=10):
        """
        Trains the client model.

        Args:
            data: Dict of the form {'x': [list], 'y': [list]}.
            num_epochs: Number of epochs to train.
            batch_size: Size of training batches.
        Return:
            comp: Number of FLOPs computed while training given data
            update: List of np.ndarray weights, with each weight array
                corresponding to a variable in the resulting graph
        """
        for _ in range(num_epochs):
            self.run_epoch(data, batch_size)

        update = self.get_params()
        comp = num_epochs * (len(data['y'])//batch_size) * batch_size * self.flops
        return comp, update

    def test(self, data):
        """
        Tests the current model on the given data.

        Args:
            data: dict of the form {'x': [list], 'y': [list]}
        Return:
            dict of metrics that will be recorded by the simulation.
        """
        x_vecs = self.process_x(data['x'])
        labels = self.process_y(data['y'])
        
        self.eval()
        with torch.no_grad():
            logits, loss = self.__call__(x_vecs, labels)
            acc = acc_fn(labels, logits)
        return {ACCURACY_KEY: acc.detach().cpu().numpy(), 'loss': loss.detach().cpu().numpy()}

    def run_epoch(self, data, batch_size):
        self.train()
        for batched_x, batched_y in batch_data(data, batch_size, seed=self.seed):
            self.optimizer.zero_grad()
            input_data = self.process_x(batched_x)
            target_data = self.process_y(batched_y)
            
            logits, loss = self.forward(input_data, target_data)
            loss.backward()
            self.optimizer.step()
