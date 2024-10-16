from torch.utils.data import Dataset
import torch
import numpy as np
import random
import os
import pandas as pd
from random import randrange
from utils import window_warp

class FinetuneDataset(Dataset):
    """Time Series Disturbance dataset."""

    def __init__(self, labels_ohe, root_dir, feature_num, seq_len, collist, classifier, data_aug, indices, only_indices):
        """
        Args:
            labels_ohe (string): dataframe with integer class labels
            root_dir (string): Directory with all the time series files.
        """
        self.labels = labels_ohe
        self.root_dir = root_dir
        self.collist = collist
        self.seq_len = seq_len
        self.dimension = feature_num
        self.classifier = classifier
        self.data_aug = data_aug
        self.indices = indices
        self.only_indices = only_indices

    def __len__(self):
        return self.labels.shape[0]  # number of samples in the dataset

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        X_name = os.path.join(self.root_dir,
                              self.labels.iloc[idx, 0] +
                              '.csv')
        
        # col_list specifies the dataframe columns to use as model input
        X = pd.read_csv(X_name, sep=',', usecols=self.collist)

        ### include indices?
        if (self.indices.lower() == 'true') or self.only_indices:
            # BLUE = B02, GREEN = B03, RED = B04, RE1 = B05, RE2 = B06, RE3 = B07,
            # NIR = B8A (= Narrow NIR) = 865nm, BNIR = B08 = 832nm, SWIR1 = B11, SWIR2 = B12
            X['CRSWIR'] = X['SW1_mean'] / (X['NIR_mean'] + ((X['SW2_mean'] - X['NIR_mean'] )/(2185.7 - 864)) * (1610.4 - 864))
            EVI = 2.5 * (X['BNR_mean'] - X['RED_mean']) / ((X['BNR_mean'] + 6 * X['RED_mean'] - 7.5 * X['BLU_mean']) + 1)
            X['NBR'] = (X['BNR_mean'] - X['SW2_mean']) / (X['BNR_mean'] + X['SW2_mean'])
            X['TCW'] = 0.1509 * X['BLU_mean'] + 0.1973 * X['GRN_mean'] + 0.3279 * X['RED_mean'] + 0.3406 * X['BNR_mean'] - 0.7112 * X['SW1_mean'] - 0.4572 * X['SW2_mean']
            TCG = -0.2848 * X['BLU_mean'] - 0.2435 * X['GRN_mean'] - 0.5436 * X['RED_mean'] + 0.7243 * X['BNR_mean'] + 0.084 * X['SW1_mean'] - 0.18 * X['SW2_mean']
            TCB = 0.3037 * X['BLU_mean'] + 0.2793 * X['GRN_mean'] + 0.4743 * X['RED_mean'] + 0.5585 * X['BNR_mean'] + 0.5082 * X['SW1_mean'] + 0.1863 * X['SW2_mean']
            X['TCD'] = TCB - (TCG + X['TCW'])
            X['NDVI'] = (X['BNR_mean'] - X['RED_mean']) / (X['BNR_mean'] + X['RED_mean'])
            X['NDWI'] = (X['NIR_mean'] - X['SW1_mean']) / (X['NIR_mean'] + X['SW1_mean'])
            X['NDMI'] = (X['BNR_mean'] - X['SW1_mean']) / (X['BNR_mean'] + X['SW1_mean'])
            # https://kaflekrishna.com.np/blog-detail/retrieving-leaf-area-index-lai-sentinel-2-image-google-earth-engine-gee/
            X['LAI'] = (3.618 * EVI) - .118
            X['MSI'] = X['SW1_mean'] / X['BNR_mean']
            X['NDRE'] = (X['BNR_mean'] - X['RE1_mean']) / (X['BNR_mean'] + X['RE1_mean'])
            # X['CRE'] = X['NIR_mean'] / X['RE1_mean'] - 1.0

            # put X['DOY'] at the end of the dataframe
            X = X.reindex(columns=[col for col in X.columns if col != 'DOY'] + ['DOY'])

            ### replace inf values by max of column
            for col in X.columns.tolist():
                max_value = np.nanmax(X[col][X[col] != np.inf])
                X[col].replace([np.inf, -np.inf], max_value, inplace=True)
            
            ### replace NA by 0 values
            X.fillna(0, inplace=True)

        ### convert to SITS-BERT input format
        if self.only_indices:
            X = X.drop(labels=self.collist[0:10], axis=1)

        X = X.unstack().to_frame().T
        X = X.reset_index(drop=True)
        line_data = X.values.flatten().tolist()
        line_data = np.array(line_data, dtype=float)

        ### extract and reshape time series
        ts = np.reshape(line_data, (self.dimension + 1, -1)).T

        ### data augmentation 1: randomly changing sequence length (= window slicing)
        if self.data_aug:
            # get max doy
            max_doy = ts[-1, -1]

            # randomly choose to cut at beginning or end of sequence
            if bool(random.getrandbits(1)):  # beginning

                # get value for 3/4*max_doy but min 1 year (=doy 365)
                if max_doy > 1280: # 1280 = 3.5 years
                    # keep at least one year in the time series
                    latest_doy = int(max_doy - 1280) # 1280 = 3.5 years

                    # get (= count) number of (non-zero!) observations until the latter value
                    obs = (ts[:, ts.shape[1]-1] < latest_doy).sum()

                    if (
                            # at least 10 observations before latest_doy
                            (obs > 1) &
                            # at least 10 non-zero observations before latest_doy
                            ((np.count_nonzero(ts[ts[:, ts.shape[1]-1] < latest_doy, :ts.shape[1]-1]))
                             / (ts.shape[1]-1) > 1)
                    ):
                        # pick random integer number between 0 and the latter value
                        rdm = randrange(0, obs)

                        # remove this number of observations starting at first observation
                        ts = ts[rdm:, :]

                        # subtract 365 from all doy values until first doy value is smaller than 365 (while-loop)
                        # this way, the time series always starts in 'first' year
                        while ts[0, ts.shape[1]-1] > 365:
                            ts[:, ts.shape[1]-1] = ts[:, ts.shape[1]-1] - 365

            ### data augmentation 2: window warping
            ts = window_warp(ts)

        # get number of observations for further processing
        ts_length = ts.shape[0]

        ### we always take the latest seq_len observations
        bert_mask = np.zeros((self.seq_len,), dtype=int)
        bert_mask[:ts_length] = 1

        ### day of year
        doy = np.zeros((self.seq_len,), dtype=int)

        # BOA reflectances
        ts_origin = np.zeros((self.seq_len, self.dimension))
        if self.seq_len > ts_length:
            ts_origin[:ts_length, :] = ts[:ts_length, :-1]
            doy[:ts_length] = np.squeeze(ts[:ts_length, -1])

            ### apply z-transformation on each band individually
            ### note that we can leave the DOY values as they are,
            ### since they are being transformed later in PositionalEncoding
            ### apply it only for non-masked part
            ts_origin[:ts_length, :] = (ts_origin[:ts_length, :] - ts_origin[:ts_length, :].mean(axis=0)) / (ts_origin[:ts_length, :].std(axis=0) + 1e-6)

            if self.data_aug:
                ### data augmentation 3
                ### 1. slightly change DOY, but keep the range
                ### range [-5, 5] will make difference of +/- one satellite revisit time
                doy_noise = np.random.randint(-5, 5, doy[:ts_length].shape[0])
                minimum = doy[:ts_length].min()
                maximum = doy[:ts_length].max()
                doy[:ts_length] = doy[:ts_length] + doy_noise
                doy[:ts_length] = np.clip(doy[:ts_length], minimum, maximum)

                ### data augmentation 4
                ### add a bit of noise to every value with respect to standard deviation
                noise = np.random.normal(0, .1, ts_origin[:ts_length, :].shape)
                ts_origin[:ts_length, :] = ts_origin[:ts_length, :] + noise

        else:
            ts_origin[:self.seq_len, :] = ts[:self.seq_len, :-1]
            doy[:self.seq_len] = np.squeeze(ts[:self.seq_len, -1])

            ### apply z-transformation on each band individually
            ### note that we can leave the DOY values as they are,
            ### since they are being transformed later in PositionalEncoding
            ### apply it only for non-masked part
            ts_origin[:self.seq_len, :] = (ts_origin[:self.seq_len, :] - ts_origin[:self.seq_len, :].mean(axis=0)) / (ts_origin[:self.seq_len, :].std(axis=0) + 1e-6)

            if self.data_aug:

                ### data augmentation 3
                ### slightly change DOY, but keep the range
                doy_noise = np.random.randint(-3, 3, doy[:self.seq_len].shape[0])
                minimum = doy[:self.seq_len].min()
                maximum = doy[:self.seq_len].max()
                doy[:self.seq_len] = doy[:self.seq_len] + doy_noise
                doy[:self.seq_len] = np.clip(doy[:self.seq_len], minimum, maximum)

                ### data augmentation 4
                ### add a bit of noise to every value with respect to standard deviation
                noise = np.random.normal(0, .1, ts_origin[:self.seq_len, :].shape)
                ts_origin[:self.seq_len, :] = ts_origin[:self.seq_len, :] + noise

        ### get class label
        class_label = np.array(self.labels.iloc[idx, 1:], dtype=int)

        output = {"bert_input": ts_origin,
                  "bert_mask": bert_mask,
                  "class_label": class_label,
                  "time": doy
                  }

        return {key: torch.from_numpy(value) for key, value in output.items()}
