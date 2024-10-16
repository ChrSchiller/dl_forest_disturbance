import torch
import torch.nn as nn
from .bert import SBERT


class SBERTClassification(nn.Module):
    """
    Downstream task: Satellite Time Series Classification
    """

    def __init__(self, sbert: SBERT, num_classes, seq_len):
        super().__init__()
        self.sbert = sbert
        self.hidden_clfr_head = self.sbert.hidden_clfr_head
        self.classification = MulticlassClassification(self.sbert.hidden, num_classes,
                                                       seq_len, hidden_clfr_head=self.hidden_clfr_head)

    def forward(self, x, doy, mask):
        x = self.sbert(x, doy, mask)
        return self.classification(x, mask)

class MulticlassClassification(nn.Module):

    def __init__(self, hidden, num_classes, seq_len, hidden_clfr_head=None):
        super().__init__()
        ### note that 64 as value for MaxPool1d only works if max_length == 64 (meaning that it is hard-coded),
        ### otherwise the code throws an error
        ### (also then the code does not meet the description in the paper)
        ### a better way to do it is like to use nn.MaxPool1d(max_length)
        ### also because then the 'squeeze' method makes more sense (the '1' dimension will be dropped)
        self.max_len = seq_len
        self.relu = nn.ReLU()
        # self.pooling = nn.MaxPool1d(64)
        self.pooling = nn.MaxPool1d(self.max_len)
        self.hidden_clfr_head = hidden_clfr_head
        if self.hidden_clfr_head:
            ### more complex classifier head:
            self.linear = nn.ModuleList()
            ### hidden_dim is hardcoded right now; it should be given as a parameter
            ### length of hidden_dim+1 will be the number of hidden layers in classifier (+1 for output dim)
            hidden_dim = [math.ceil(hidden / 2), math.ceil(hidden / 2), math.ceil(hidden / 4)]
            current_dim = hidden
            for hdim in hidden_dim:
                self.linear.append(nn.Linear(current_dim, hdim))
                current_dim = hdim
            self.linear.append(nn.Linear(current_dim, num_classes))
        else:
            ### the following is the original code:
            self.linear = nn.Linear(hidden, num_classes)


    def forward(self, x, mask):
        x = self.pooling(x.permute(0, 2, 1)).squeeze()

        if self.hidden_clfr_head:
            for layer in self.linear[:-1]:
                x = self.relu(layer(x))
            out = self.linear[-1](x) # with BCELoss: torch.sigmoid(self.linear[-1](x))
            # we can drop sigmoid function as last layer if we use BCEWithLogitsLoss
            return out
        else:
            x = self.linear(x)
            return x