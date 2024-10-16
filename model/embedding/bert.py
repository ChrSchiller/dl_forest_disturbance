import torch
import torch.nn as nn
from .position import PositionalEncoding


class BERTEmbedding(nn.Module):
    """
    BERT Embedding which is consisted with under features
        1. InputEmbedding : project the input to embedding size through a fully connected layer
        2. PositionalEncoding : adding positional information using sin, cos
        sum of both features are output of BERTEmbedding
    """

    def __init__(self, num_features, embedding_dim, cnn_embedding, dropout=0.2):
        """
        :param feature_num: number of input features
        :param embedding_dim: embedding size of token embedding
        :param dropout: dropout rate
        """
        super().__init__()

        self.cnn_embedding = cnn_embedding
        self.relu = nn.ReLU()

        if self.cnn_embedding:
            ### my code with 1D-CNN layers as embedding
            ### inspired by:
            # https://towardsdatascience.com/heart-disease-classification-using-transformers-in-pytorch-8dbd277e079
            # and (code for the latter)
            # https://github.com/bh1995/AF-classification/blob/master/src/models/TransformerModel.py
            ### note that the github code mentioned above suggests to
            # resize to --> [batch, input_channels, signal_length]
            # at the moment, it is [batch, signal_length, input_channels]
            self.input = nn.ModuleList()
            self.input.append(nn.Conv1d(in_channels=num_features, out_channels=128,
                                        kernel_size=1, stride=1, padding=0))
            self.input.append(self.relu)
            self.input.append(nn.Conv1d(in_channels=128, out_channels=embedding_dim,
                                        kernel_size=3, stride=1, padding=1))
            self.input.append(self.relu)
            self.input.append(nn.Conv1d(in_channels=embedding_dim, out_channels=embedding_dim * 2,
                                        kernel_size=3, stride=1, padding=1))
            self.input.append(self.relu)
            self.input.append(nn.MaxPool1d(kernel_size=2))
        else:
            ### original code with one fully connected embedding layer:
            self.input = nn.Linear(in_features=num_features, out_features=embedding_dim)
            ### my code with some more fc layers:
            # self.input = nn.ModuleList()
            # hidden_dim = [512, 512, 512]
            # current_dim = num_features
            # for hdim in hidden_dim:
            #     self.input.append(nn.Linear(current_dim, hdim))
            #     current_dim = hdim
            # self.input.append(nn.Linear(current_dim, embedding_dim))

        # max_len 1825 = 5 years, but smaller is enough as well
        # CUDA throws error if highest DOY value higher than this max_len value
        # (basically 'index out of bounds')
        # so we need to keep it high if the time series are long
        self.position = PositionalEncoding(d_model=embedding_dim, max_len=1825)
        self.dropout = nn.Dropout(p=dropout)
        self.embed_size = embedding_dim

    def forward(self, input_sequence, doy_sequence):
        batch_size = input_sequence.size(0)
        seq_length = input_sequence.size(1)

        # ### own code extending the embedding depth:
        # for layer in self.input[:-1]:
        #     input_sequence = F.relu(layer(input_sequence))
        # obs_embed = self.input[-1](input_sequence)

        if self.cnn_embedding:
            ### this code is needed in case of 1D-CNN embeddings
            obs_embed = input_sequence.permute((0, 2, 1))
            # obs_embed = self.input[0](obs_embed)
            for conv in self.input[0:]:
                obs_embed = conv(obs_embed)
            # obs_embed = self.input[-1](obs_embed)
            ### change code for 1D-CNN implementation
            ### the above permute command has to be reversed
            x = obs_embed.repeat(1, 1, 2).permute(0, 2, 1)
        else:
            ### this is the original code:
            obs_embed = self.input(input_sequence)  # [batch_size, seq_length, embedding_dim]
            ### the following line is the original code:
            x = obs_embed.repeat(1, 1, 2)           # [batch_size, seq_length, embedding_dim*2]

        for i in range(batch_size):
            x[i, :, self.embed_size:] = self.position(doy_sequence[i, :])     # [seq_length, embedding_dim]

        return self.dropout(x)