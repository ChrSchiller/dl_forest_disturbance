import torch
from torch.utils.data import DataLoader
from torch.utils.data import Dataset, ConcatDataset
import torch.nn.functional as F
import torch.nn as nn
from torch.optim import Adam
import time
from sklearn.utils import resample
# from torch.optim.lr_scheduler import OneCycleLR
from captum.attr import (
    GradientShap,
    DeepLift,
    DeepLiftShap,
    IntegratedGradients,
    FeatureAblation,
    LayerConductance,
    NeuronConductance,
    NoiseTunnel,
    Saliency,
    visualization as viz,
    configure_interpretable_embedding_layer,
    remove_interpretable_embedding_layer
)
import math
# from model import SBERT
# from trainer import SBERTFineTuner
# from dataset import FinetuneDataset
from multiprocessing import Pool
import numpy as np
from numpy import ndarray
import random
from random import randrange
import math
import glob
import argparse
import os
import matplotlib.pyplot as plt
from matplotlib.pyplot import axis, figure
from matplotlib import cm, colors, pyplot as plt
from matplotlib.collections import LineCollection
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.figure import Figure
from mpl_toolkits.axes_grid1 import make_axes_locatable
import pandas as pd
from enum import Enum
import warnings
from typing import Any, Iterable, List, Optional, Tuple, Union
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, classification_report
# Importing LabelEncoder from Sklearn
# library from preprocessing Module.
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, precision_score, recall_score, ConfusionMatrixDisplay
from sklearn.model_selection import RandomizedSearchCV
import pickle
from sktime.classification.interval_based import TimeSeriesForestClassifier
from sktime.datasets import load_arrow_head, load_japanese_vowels
import sktime
from sktime.registry import all_estimators
from sktime.classification.kernel_based._svc import TimeSeriesSVC
from sktime.transformations.panel.padder import PaddingTransformer
from sktime.transformations.panel.compose import ColumnConcatenator
from sktime.classification.ensemble import BaggingClassifier
import copy
from sklearn.gaussian_process.kernels import RBF
from sktime.dists_kernels import AggrDist

from pandas.core.common import SettingWithCopyWarning
warnings.simplefilter(action="ignore", category=SettingWithCopyWarning)

# Creating an instance of label Encoder.
le = LabelEncoder()

### activate the seeds only for cross-validation and final reproducibility,
### but not for testing the model and code: could lead to accidentally
### picking very good dataset
def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True

setup_seed(123) ### original seed: 123

# https://towardsdatascience.com/a-simple-guide-to-command-line-arguments-with-argparse-6824c30ab1c3
# Create the parser
parser = argparse.ArgumentParser()
# Add an argument
parser.add_argument('--target_aoi', type=str, required=True)
parser.add_argument('--indices', type=str, required=True)
# parser.add_argument('--train_baseline', type=bool, required=True)
# Parse the argument
args = parser.parse_args()

### argparse stuff, implement when everything else is finished
# def Config():
#     parser = argparse.ArgumentParser()
#     # Required parameters
#     parser.add_argument(
#         "--file_path",
#         default=None,
#         type=str,
#         required=False,
#         help="The input data path.",
#     )
#     parser.add_argument(
#         "--pretrain_path",
#         default=None,
#         type=str,
#         required=False,
#         help="The storage path of the pre-trained model.",
#     )
#     parser.add_argument(
#         "--finetune_path",
#         default=None,
#         type=str,
#         required=False,
#         help="The output directory where the fine-tuning checkpoints will be written.",
#     )
#     parser.add_argument(
#         "--valid_rate",
#         default=0.03,
#         type=float,
#         help="")
#     parser.add_argument(
#         "--max_length",
#         default=256,
#         type=int,
#         help="The maximum length of input time series. Sequences longer "
#              "than this will be truncated, sequences shorter will be padded.",
#     )
#     parser.add_argument(
#         "--num_features",
#         default=10,
#         type=int,
#         help="",
#     )
#     parser.add_argument(
#         "--num_classes",
#         default=2,
#         type=int,
#         help="",
#     )
#     parser.add_argument(
#         "--epochs",
#         default=100,
#         type=int,
#         help="",
#     )
#     parser.add_argument(
#         "--batch_size",
#         default=64,
#         type=int,
#         help="",
#     )
#     parser.add_argument(
#         "--hidden_size",
#         default=256,
#         type=int,
#         help="",
#     )
#     parser.add_argument(
#         "--layers",
#         default=3,
#         type=int,
#         help="",
#     )
#     parser.add_argument(
#         "--attn_heads",
#         default=8,
#         type=int,
#         help="",
#     )
#     parser.add_argument(
#         "--learning_rate",
#         default=2e-5,
#         type=float,
#         help="",
#     )
#     parser.add_argument(
#         "--dropout",
#         default=0.1,
#         type=float,
#         help="",
#     )
#     return parser.parse_args()


### all timeseries visualization convenience functions are adopted from:
### https://github.com/pytorch/captum/blob/010f76dc36a68d62cc7ad59fc6582cfaa1d19008/captum/attr/_utils/visualization.py#L3
### only slight changes are made, credits and many thanks to "smaeland"
class TimeseriesVisualizationMethod_cs(Enum):
    overlay_individual = 1
    overlay_combined = 2
    colored_graph = 3

class VisualizeSign_cs(Enum):
    positive = 1
    absolute_value = 2
    negative = 3
    all = 4

def _cumulative_sum_threshold_cs(values: ndarray, percentile: Union[int, float]):
    # given values should be non-negative
    assert percentile >= 0 and percentile <= 100, (
        "Percentile for thresholding must be " "between 0 and 100 inclusive."
    )
    sorted_vals = np.sort(values.flatten())
    cum_sums = np.cumsum(sorted_vals)
    threshold_id = np.where(cum_sums >= cum_sums[-1] * 0.01 * percentile)[0][0]
    return sorted_vals[threshold_id]

def _normalize_scale_cs(attr: ndarray, scale_factor: float):
    assert scale_factor != 0, "Cannot normalize by scale factor = 0"
    if abs(scale_factor) < 1e-5:
        warnings.warn(
            "Attempting to normalize by value approximately 0, visualized results"
            "may be misleading. This likely means that attribution values are all"
            "close to 0."
        )
    attr_norm = attr / scale_factor
    return np.clip(attr_norm, -1, 1)
def _normalize_attr_cs(
    attr: ndarray,
    sign: str,
    outlier_perc: Union[int, float] = 2,
    reduction_axis: Optional[int] = None,
):
    attr_combined = attr
    if reduction_axis is not None:
        attr_combined = np.sum(attr, axis=reduction_axis)

    # Choose appropriate signed values and rescale, removing given outlier percentage.
    if VisualizeSign_cs[sign] == VisualizeSign_cs.all:
        threshold = _cumulative_sum_threshold_cs(np.abs(attr_combined), 100 - outlier_perc)
    elif VisualizeSign_cs[sign] == VisualizeSign_cs.positive:
        attr_combined = (attr_combined > 0) * attr_combined
        threshold = _cumulative_sum_threshold_cs(attr_combined, 100 - outlier_perc)
    elif VisualizeSign_cs[sign] == VisualizeSign_cs.negative:
        attr_combined = (attr_combined < 0) * attr_combined
        threshold = -1 * _cumulative_sum_threshold_cs(
            np.abs(attr_combined), 100 - outlier_perc
        )
    elif VisualizeSign_cs[sign] == VisualizeSign_cs.absolute_value:
        attr_combined = np.abs(attr_combined)
        threshold = _cumulative_sum_threshold_cs(attr_combined, 100 - outlier_perc)
    else:
        raise AssertionError("Visualize Sign type is not valid.")
    return _normalize_scale_cs(attr_combined, threshold)

def visualize_timeseries_attr_cs(
    attr: ndarray,
    data: ndarray,
    x_values: Optional[ndarray] = None,
    method: str = "individual_channels",
    sign: str = "absolute_value",
    channel_labels: Optional[List[str]] = None,
    channels_last: bool = True,
    plt_fig_axis: Union[None, Tuple[figure, axis]] = None,
    outlier_perc: Union[int, float] = 2,
    cmap: Union[None, str] = None,
    alpha_overlay: float = 0.7,
    show_colorbar: bool = False,
    title: Union[None, str] = None,
    fig_size: Tuple[int, int] = (6, 6),
    use_pyplot: bool = True,
    **pyplot_kwargs,
):
    r"""
    Visualizes attribution for a given timeseries data by normalizing
    attribution values of the desired sign (positive, negative, absolute value,
    or all) and displaying them using the desired mode in a matplotlib figure.
    Args:
        attr (numpy.ndarray): Numpy array corresponding to attributions to be
                    visualized. Shape must be in the form (N, C) with channels
                    as last dimension, unless `channels_last` is set to True.
                    Shape must also match that of the timeseries data.
        data (numpy.ndarray): Numpy array corresponding to the original,
                    equidistant timeseries data. Shape must be in the form
                    (N, C) with channels as last dimension, unless
                    `channels_last` is set to true.
        x_values (numpy.ndarray, optional): Numpy array corresponding to the
                    points on the x-axis. Shape must be in the form (N, ). If
                    not provided, integers from 0 to N-1 are used.
                    Default: None
        method (str, optional): Chosen method for visualizing attributions
                    overlaid onto data. Supported options are:
                    1. `overlay_individual` - Plot each channel individually in
                        a separate panel, and overlay the attributions for each
                        channel as a heat map. The `alpha_overlay` parameter
                        controls the alpha of the heat map.
                    2. `overlay_combined` - Plot all channels in the same panel,
                        and overlay the average attributions as a heat map.
                    3. `colored_graph` - Plot each channel in a separate panel,
                        and color the graphs according to the attribution
                        values. Works best with color maps that does not contain
                        white or very bright colors.
                    Default: `overlay_individual`
        sign (str, optional): Chosen sign of attributions to visualize.
                    Supported options are:
                    1. `positive` - Displays only positive pixel attributions.
                    2. `absolute_value` - Displays absolute value of
                        attributions.
                    3. `negative` - Displays only negative pixel attributions.
                    4. `all` - Displays both positive and negative attribution
                        values.
                    Default: `absolute_value`
        channel_labels (list[str], optional): List of labels
                    corresponding to each channel in data.
                    Default: None
        channels_last (bool, optional): If True, data is expected to have
                    channels as the last dimension, i.e. (N, C). If False, data
                    is expected to have channels first, i.e. (C, N).
                    Default: True
        plt_fig_axis (tuple, optional): Tuple of matplotlib.pyplot.figure and axis
                    on which to visualize. If None is provided, then a new figure
                    and axis are created.
                    Default: None
        outlier_perc (float or int, optional): Top attribution values which
                    correspond to a total of outlier_perc percentage of the
                    total attribution are set to 1 and scaling is performed
                    using the minimum of these values. For sign=`all`, outliers
                    and scale value are computed using absolute value of
                    attributions.
                    Default: 2
        cmap (str, optional): String corresponding to desired colormap for
                    heatmap visualization. This defaults to "Reds" for negative
                    sign, "Blues" for absolute value, "Greens" for positive sign,
                    and a spectrum from red to green for all. Note that this
                    argument is only used for visualizations displaying heatmaps.
                    Default: None
        alpha_overlay (float, optional): Alpha to set for heatmap when using
                    `blended_heat_map` visualization mode, which overlays the
                    heat map over the greyscaled original image.
                    Default: 0.7
        show_colorbar (bool): Displays colorbar for heat map below
                    the visualization.
        title (str, optional): Title string for plot. If None, no title is
                    set.
                    Default: None
        fig_size (tuple, optional): Size of figure created.
                    Default: (6,6)
        use_pyplot (bool): If true, uses pyplot to create and show
                    figure and displays the figure after creating. If False,
                    uses Matplotlib object oriented API and simply returns a
                    figure object without showing.
                    Default: True.
        pyplot_kwargs: Keyword arguments forwarded to plt.plot, for example
                    `linewidth=3`, `color='black'`, etc
    Returns:
        2-element tuple of **figure**, **axis**:
        - **figure** (*matplotlib.pyplot.figure*):
                    Figure object on which visualization
                    is created. If plt_fig_axis argument is given, this is the
                    same figure provided.
        - **axis** (*matplotlib.pyplot.axis*):
                    Axis object on which visualization
                    is created. If plt_fig_axis argument is given, this is the
                    same axis provided.
    """

    # Check input dimensions
    assert len(attr.shape) == 2, "Expected attr of shape (N, C), got {}".format(
        attr.shape
    )
    assert len(data.shape) == 2, "Expected data of shape (N, C), got {}".format(
        attr.shape
    )

    # Convert to channels-first
    if channels_last:
        attr = np.transpose(attr)
        data = np.transpose(data)

    num_channels = attr.shape[0]
    timeseries_length = attr.shape[1]

    if num_channels > timeseries_length:
        warnings.warn(
            "Number of channels ({}) greater than time series length ({}), "
            "please verify input format".format(num_channels, timeseries_length)
        )

    num_subplots = num_channels
    if (
        TimeseriesVisualizationMethod_cs[method]
        == TimeseriesVisualizationMethod_cs.overlay_combined
    ):
        num_subplots = 1
        attr = np.sum(attr, axis=0)  # Merge attributions across channels

    ### the next bit contains a small bug fix by myself
    if x_values is not None:
        if channels_last:
            assert (
                x_values.shape[0] == timeseries_length
            ), "x_values must have same length as data"
        else:
            assert (
                    x_values.shape[0] == num_channels
            ), "x_values must have same length as data"
    else:
        x_values = np.arange(timeseries_length)

    # Create plot if figure, axis not provided
    if plt_fig_axis is not None:
        plt_fig, plt_axis = plt_fig_axis
    else:
        if use_pyplot:
            plt_fig, plt_axis = plt.subplots(
                figsize=fig_size, nrows=num_subplots, sharex=True
            )
        else:
            plt_fig = Figure(figsize=fig_size)
            plt_axis = plt_fig.subplots(nrows=num_subplots, sharex=True)

    if not isinstance(plt_axis, ndarray):
        plt_axis = np.array([plt_axis])

    norm_attr = _normalize_attr_cs(attr, sign, outlier_perc, reduction_axis=None)

    # Set default colormap and bounds based on sign.
    if VisualizeSign_cs[sign] == VisualizeSign_cs.all:
        default_cmap = LinearSegmentedColormap.from_list(
            "RdWhGn", ["red", "white", "green"]
        )
        vmin, vmax = -1, 1
    elif VisualizeSign_cs[sign] == VisualizeSign_cs.positive:
        default_cmap = "Greens"
        vmin, vmax = 0, 1
    elif VisualizeSign_cs[sign] == VisualizeSign_cs.negative:
        default_cmap = "Reds"
        vmin, vmax = 0, 1
    elif VisualizeSign_cs[sign] == VisualizeSign_cs.absolute_value:
        default_cmap = "Blues"
        vmin, vmax = 0, 1
    else:
        raise AssertionError("Visualize Sign type is not valid.")
    cmap = cmap if cmap is not None else default_cmap
    cmap = cm.get_cmap(cmap)
    cm_norm = colors.Normalize(vmin, vmax)

    def _plot_attrs_as_axvspan(attr_vals, x_vals, ax):

        half_col_width = (x_values[1] - x_values[0]) / 2.0
        for icol, col_center in enumerate(x_vals):
            left = col_center - half_col_width
            right = col_center + half_col_width
            ax.axvspan(
                xmin=left,
                xmax=right,
                facecolor=(cmap(cm_norm(attr_vals[icol]))),
                edgecolor=None,
                alpha=alpha_overlay,
            )

    if (
        TimeseriesVisualizationMethod_cs[method]
        == TimeseriesVisualizationMethod_cs.overlay_individual
    ):

        for chan in range(num_channels):

            plt_axis[chan].plot(x_values, data[chan, :], **pyplot_kwargs)
            if channel_labels is not None:
                plt_axis[chan].set_ylabel(channel_labels[chan])

            _plot_attrs_as_axvspan(norm_attr[chan], x_values, plt_axis[chan])

        plt.subplots_adjust(hspace=0)

    elif (
        TimeseriesVisualizationMethod_cs[method]
        == TimeseriesVisualizationMethod_cs.overlay_combined
    ):

        # Dark colors are better in this case
        # green, re1, re2 and so on = cm.Dark2.colors # unpacking the tuple, then cols = (green, re1, re2, and so on...)
        cycler = plt.cycler("color", cm.Dark2.colors)
        plt_axis[0].set_prop_cycle(cycler)

        for chan in range(num_channels):
            label = channel_labels[chan] if channel_labels else None
            plt_axis[0].plot(x_values, data[chan, :], label=label, **pyplot_kwargs)

        _plot_attrs_as_axvspan(norm_attr, x_values, plt_axis[0])

        ### legend position changed because of bad experience with "best"
        plt_axis[0].legend(loc="upper left")

    elif (
        TimeseriesVisualizationMethod_cs[method]
        == TimeseriesVisualizationMethod_cs.colored_graph
    ):

        for chan in range(num_channels):

            points = np.array([x_values, data[chan, :]]).T.reshape(-1, 1, 2)
            segments = np.concatenate([points[:-1], points[1:]], axis=1)

            lc = LineCollection(segments, cmap=cmap, norm=cm_norm, **pyplot_kwargs)
            lc.set_array(norm_attr[chan, :])
            plt_axis[chan].add_collection(lc)
            plt_axis[chan].set_ylim(
                1.2 * np.min(data[chan, :]), 1.2 * np.max(data[chan, :])
            )
            if channel_labels is not None:
                plt_axis[chan].set_ylabel(channel_labels[chan])

        plt.subplots_adjust(hspace=0)

    else:
        raise AssertionError("Invalid visualization method: {}".format(method))

    plt.xlim([x_values[0], x_values[-1]])

    if show_colorbar:
        axis_separator = make_axes_locatable(plt_axis[-1])
        colorbar_axis = axis_separator.append_axes("bottom", size="5%", pad=0.4)
        colorbar_alpha = alpha_overlay
        if (
            TimeseriesVisualizationMethod_cs[method]
            == TimeseriesVisualizationMethod_cs.colored_graph
        ):
            colorbar_alpha = 1.0
        plt_fig.colorbar(
            cm.ScalarMappable(cm_norm, cmap),
            orientation="horizontal",
            cax=colorbar_axis,
            alpha=colorbar_alpha,
        )
    if title:
        plt_axis[0].set_title(title)

    if use_pyplot:
        plt.show()

    return plt_fig, plt_axis


class EarlyStopper:
    def __init__(self, patience=1, min_delta=0):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.min_validation_loss = np.inf

    def early_stop(self, validation_loss):
        if validation_loss < self.min_validation_loss:
            self.min_validation_loss = validation_loss
            self.counter = 0
        elif validation_loss > (self.min_validation_loss + self.min_delta):
            self.counter += 1
            if self.counter >= self.patience:
                return True
        return False

def get_binary_accuracy(y_true, y_prob):
    assert y_true.ndim == 1 and y_true.size() == y_prob.size()
    y_prob = y_prob > 0.5
    return (y_true == y_prob).sum().item() / y_true.size(0)


def normalize_array(arr):
    normalized_vector = arr / np.linalg.norm(arr)
    return normalized_vector

### modified from https://github.com/uchidalab/time_series_augmentation/blob/master/utils/augmentation.py
### modifications mainly necessary due to conversion to Pytorch from Keras/Tensorflow
### as we have only 1 sample here, while the original implementation considers a whole batch at once;
### but also because of use case in this study
def window_warp(x, window_ratio=0.1, scales=[0.5, 2.]):
    # https://halshs.archives-ouvertes.fr/halshs-01357973/document
    warp_scales = np.random.choice(scales)
    warp_size = np.ceil(window_ratio * x.shape[0]).astype(int)
    window_steps = np.arange(warp_size)

    # print(1 > (x.shape[0] - warp_size - 1))
    if (x.shape[0] - warp_size - 1) > 10:
        window_starts = np.random.randint(low=1, high=x.shape[0] - warp_size - 1, size=1).astype(int)
        window_ends = (window_starts + warp_size).astype(int)

        ret = np.zeros_like(x)
        for dim in range(x.shape[1]):
            pat = x[:, dim]
            start_seg = pat[:window_starts[0]]
            window_seg = np.interp(np.linspace(0, warp_size - 1, num=int(warp_size * warp_scales)), window_steps,
                                   pat[window_starts[0]:window_ends[0]])
            end_seg = pat[window_ends[0]:]
            warped = np.concatenate((start_seg, window_seg, end_seg))
            ret[:, dim] = np.interp(np.arange(x.shape[0]), np.linspace(0, x.shape[0] - 1., num=warped.size),
                                       warped).T
        ret[:, -1] = ret[:, -1].astype(int)

        # plt.plot(ret[:, 0])
        # plt.plot(x[:, 0])
        # plt.show()

        return ret
    else:
        return x



# class PretrainDataset(Dataset):
#     """Time Series Disturbance dataset."""
#
#     def __init__(self, labels_ohe, root_dir, feature_num, seq_len, collist, classifier, data_aug, indices, only_indices): # frac_con,
#         """
#         Args:
#             labels_ohe (string): dataframe with integer class labels
#             root_dir (string): Directory with all the time series files.
#         """
#         self.labels = labels_ohe
#         self.root_dir = root_dir
#         # self.transform = transform
#         # self.rescale = rescale
#         self.collist = collist
#         self.seq_len = seq_len
#         self.dimension = feature_num
#         self.classifier = classifier
#         self.data_aug = data_aug
#         self.indices = indices
#         self.only_indices = only_indices
#         # self.frac_con = frac_con
#
#     def __len__(self):
#         return self.labels.shape[0]  # number of samples in the dataset
#
#     def __getitem__(self, idx):
#         if torch.is_tensor(idx):
#             idx = idx.tolist()
#
#         X_name = os.path.join(self.root_dir,
#                               self.labels.iloc[idx, 0] +
#                               '.csv')
#         # col_list specifies the dataframe columns to use as model input
#         X = pd.read_csv(X_name, sep=',', usecols=self.collist)
#
#         ### include indices?
#         if (self.indices.lower() == 'true') or self.only_indices:
#             # BLUE = B02, GREEN = B03, RED = B04, RE1 = B05, RE2 = B06, RE3 = B07,
#             # NIR = B8A (= Narrow NIR) = 865nm, BNIR = B08 = 832nm, SWIR1 = B11, SWIR2 = B12
#             X['CRSWIR'] = X['SW1_mean'] / (X['NIR_mean'] + ((X['SW2_mean'] - X['NIR_mean'] )/(2185.7 - 864)) * (1610.4 - 864))
#             EVI = 2.5 * (X['BNR_mean'] - X['RED_mean']) / ((X['BNR_mean'] + 6 * X['RED_mean'] - 7.5 * X['BLU_mean']) + 1)
#             X['NBR'] = (X['BNR_mean'] - X['SW2_mean']) / (X['BNR_mean'] + X['SW2_mean'])
#             X['TCW'] = 0.1509 * X['BLU_mean'] + 0.1973 * X['GRN_mean'] + 0.3279 * X['RED_mean'] + 0.3406 * X['BNR_mean'] - 0.7112 * X['SW1_mean'] - 0.4572 * X['SW2_mean']
#             TCG = -0.2848 * X['BLU_mean'] - 0.2435 * X['GRN_mean'] - 0.5436 * X['RED_mean'] + 0.7243 * X['BNR_mean'] + 0.084 * X['SW1_mean'] - 0.18 * X['SW2_mean']
#             TCB = 0.3037 * X['BLU_mean'] + 0.2793 * X['GRN_mean'] + 0.4743 * X['RED_mean'] + 0.5585 * X['BNR_mean'] + 0.5082 * X['SW1_mean'] + 0.1863 * X['SW2_mean']
#             X['TCD'] = TCB - (TCG + X['TCW'])
#             X['NDVI'] = (X['BNR_mean'] - X['RED_mean']) / (X['BNR_mean'] + X['RED_mean'])
#             X['NDWI'] = (X['NIR_mean'] - X['SW1_mean']) / (X['NIR_mean'] + X['SW1_mean'])
#             X['NDMI'] = (X['BNR_mean'] - X['SW1_mean']) / (X['BNR_mean'] + X['SW1_mean'])
#             # https://kaflekrishna.com.np/blog-detail/retrieving-leaf-area-index-lai-sentinel-2-image-google-earth-engine-gee/
#             X['LAI'] = (3.618 * EVI) - .118
#             X['MSI'] = X['SW1_mean'] / X['BNR_mean']
#             X['NDRE'] = (X['BNR_mean'] - X['RE1_mean']) / (X['BNR_mean'] + X['RE1_mean'])
#             # X['CRE'] = X['NIR_mean'] / X['RE1_mean'] - 1.0
#
#             # put X['DOY'] at the end of the dataframe
#             X = X.reindex(columns=[col for col in X.columns if col != 'DOY'] + ['DOY'])
#
#             ### replace inf values by max of column
#             for col in X.columns.tolist():
#                 max_value = np.nanmax(X[col][X[col] != np.inf])
#                 X[col].replace([np.inf, -np.inf], max_value, inplace=True)
#             ### replace NA by 0 values
#             X.fillna(0, inplace=True)
#
#             # if X.isnull().sum().sum() > 0:
#             #     print('NA value found!')
#             #     print(X.isnull().sum().sum())
#             #     print(X.columns.to_series()[np.isnan(X).any()])
#             #     print(X.index[np.isnan(X).any(1)])
#             # if np.isinf(X).values.sum() > 0:
#             #     print('X contains Inf values: N = ')
#             #     print(np.isinf(X).values.sum())
#             #     print(X.columns.to_series()[np.isinf(X).any()])
#             #     print(X.index[np.isinf(X).any(1)])
#
#         ### convert to SITS-BERT input format
#
#         if self.only_indices:
#             X = X.drop(labels=self.collist[0:10], axis=1)
#
#         X = X.unstack().to_frame().T
#         X = X.reset_index(drop=True)
#         line_data = X.values.flatten().tolist()
#         line_data = np.array(line_data, dtype=float)
#
#         ### extract and reshape time series
#         ts = np.reshape(line_data, (self.dimension + 1, -1)).T
#
#         ### implement data augmentation by randomly changing sequence length (= window slicing)
#         if self.data_aug:
#             # get max doy
#             max_doy = ts[-1, -1]
#
#             # start_or_end = bool(random.getrandbits(1))
#
#             # randomly choose to cut at beginning or end of sequence
#             if bool(random.getrandbits(1)):  # beginning
#
#                 # get value for 3/4*max_doy but min 1 year (=doy 365)
#                 if max_doy > 365:
#                     # keep at least one year in the time series
#                     latest_doy = int(max_doy - 365)
#
#                     # get (= count) number of (non-zero!) observations until the latter value
#                     obs = (ts[:, ts.shape[1]-1] < latest_doy).sum()
#
#                     if (
#                             # at least 10 observations before latest_doy
#                             (obs > 9) &
#                             # at least 10 non-zero observations before latest_doy
#                             ((np.count_nonzero(ts[ts[:, ts.shape[1]-1] < latest_doy, :ts.shape[1]-1]))
#                              / (ts.shape[1]-1) > 9)
#                     ):
#                         # pick random integer number between 0 and the latter value
#                         rdm = randrange(0, obs)
#
#                         # remove this number of observations starting at first observation
#                         ts = ts[rdm:, :]
#
#                         # subtract 365 from all doy values until first doy value is smaller than 365 (while-loop)
#                         # this way, the time series always starts in 'first' year
#                         while ts[0, ts.shape[1]-1] > 365:
#                             ts[:, ts.shape[1]-1] = ts[:, ts.shape[1]-1] - 365
#
#             if bool(random.getrandbits(1)): # cut at the end of sequence
#
#                 # get updated max doy
#                 max_doy = ts[-1, -1]
#
#                 # the idea is to enable early warnings:
#                 # only if the disturbance itself is not part of the sequence (e.g. RGB change from green to brown)
#                 # the model can focus on detecting it before (e.g. changes in infrared due to water content in tissue)
#                 if max_doy > 547: # 547 (365+182) # if sequence is longer than 1.5 years (547), cut a max of .5 years at the end
#                     # get (= count) number of observations in last .5 years of sequence
#                     obs = (ts[:, ts.shape[1] - 1] > (max_doy-182)).sum()
#
#                     if (
#                             # at least 10 observations before latest_doy
#                             (obs > 9) &
#                             # at least 10 non-zero observations before latest_doy
#                             ((np.count_nonzero(ts[ts[:, ts.shape[1]-1] < (max_doy-182), :ts.shape[1]-1])) / (ts.shape[1]-1) > 9)
#                     ):
#                         # pick random integer number between 0 and the latter value
#                         rdm = randrange(0, obs)
#
#                         if rdm != 0:
#                             # remove this number of observations at end of sequence
#                             ts = ts[:-rdm, :] # returns empty array if rdm==0
#
#             ### data augmentation 2: window warping
#             ts = window_warp(ts)
#
#
#         # get number of observations for further processing
#         ts_length = ts.shape[0]
#
#         # ### if self.seq_len < ts_length
#         # ### use truncating instead of padding
#         # if self.seq_len < ts_length:
#         #     ts_length = self.seq_len
#
#         ### we always take the LATEST seq_len observations,
#         ### (in the original paper/code, it has been the FIRST seq_len observations)
#         ### since they are considered most important
#         ### padding or truncating
#         bert_mask = np.zeros((self.seq_len,), dtype=int)
#         bert_mask[:ts_length] = 1
#
#         ### day of year
#         doy = np.zeros((self.seq_len,), dtype=int)
#
#         # BOA reflectances
#         ts_origin = np.zeros((self.seq_len, self.dimension))
#         if self.seq_len > ts_length:
#             ts_origin[:ts_length, :] = ts[:ts_length, :-1] / 10000.0 # divide by 10k dropped because of z-transform
#             doy[:ts_length] = np.squeeze(ts[:ts_length, -1])
#
#             ### apply z-transformation on each band individually
#             ### note that we can leave the DOY values as they are,
#             ### since they are being transformed later in PositionalEncoding
#             # nonzero_length = np.count_nonzero(ts[:, :ts.shape[1]-1]) / 10
#             ### apply it only for non-masked part
#             ts_origin[:ts_length, :] = (ts_origin[:ts_length, :] - ts_origin[:ts_length, :].mean(axis=0)) / (ts_origin[:ts_length, :].std(axis=0) + 1e-6)
#
#             if self.data_aug:
#
#                 ### data augmentation 3
#                 ### 1. slightly change DOY, but keep the range
#                 ### range [-5, 5] will make difference of one satellite scene
#                 ### this could also be dropped, since window warping
#                 ### induces a change of DOY already
#                 doy_noise = np.random.randint(-5, 5, doy[:ts_length].shape[0])
#                 minimum = doy[:ts_length].min()
#                 maximum = doy[:ts_length].max()
#                 doy[:ts_length] = doy[:ts_length] + doy_noise
#                 doy[:ts_length] = np.clip(doy[:ts_length], minimum, maximum)
#
#                 ### data augmentation 4
#                 ### 2. add a bit of noise to every value with respect to standard deviation
#                 noise = np.random.normal(0, .1, ts_origin[:ts_length, :].shape)
#
#                 # test = ts_origin[:ts_length, :] + noise
#                 # plt.plot(doy[:ts_length], test[:, 0])
#                 # plt.plot(doy[:ts_length], ts_origin[:ts_length, 0])
#                 # plt.show()
#
#                 ts_origin[:ts_length, :] = ts_origin[:ts_length, :] + noise
#
#         else:
#             ts_origin[:self.seq_len, :] = ts[:self.seq_len, :-1] / 10000.0 # divide by 10k dropped because of z-transform
#             doy[:self.seq_len] = np.squeeze(ts[:self.seq_len, -1])
#
#
#             ### apply z-transformation on each band individually
#             ### note that we can leave the DOY values as they are,
#             ### since they are being transformed later in PositionalEncoding
#             # nonzero_length = np.count_nonzero(ts[:, :ts.shape[1]-1]) / 10
#             ### apply it only for non-masked part
#             ts_origin[:self.seq_len, :] = (ts_origin[:self.seq_len, :] - ts_origin[:self.seq_len, :].mean(axis=0)) / (ts_origin[:self.seq_len, :].std(axis=0) + 1e-6)
#
#             if self.data_aug:
#
#                 ### data augmentation 3
#                 ### 1. slightly change DOY, but keep the range
#                 ### this could also be dropped, since window warping
#                 ### induces a change of DOY already
#                 doy_noise = np.random.randint(-3, 3, doy[:self.seq_len].shape[0])
#                 minimum = doy[:self.seq_len].min()
#                 maximum = doy[:self.seq_len].max()
#                 doy[:self.seq_len] = doy[:self.seq_len] + doy_noise
#                 doy[:self.seq_len] = np.clip(doy[:self.seq_len], minimum, maximum)
#
#                 ### data augmentation 4
#                 ### 2. add a bit of noise (here: 1/20 of std of signal) to every value with respect to standard deviation
#                 noise = np.random.normal(0, .1, ts_origin[:self.seq_len, :].shape)
#                 ts_origin[:self.seq_len, :] = ts_origin[:self.seq_len, :] + noise
#
#
#         ### get class label
#         if self.classifier:
#             class_label = np.array(self.labels.iloc[idx, 1:], dtype=int)
#         else:
#             class_label = np.array(self.labels.iloc[idx, 1:], dtype=float)
#
#
#         # ### get auxiliary information (fraction of coniferous forest)
#         # frac_con = np.array(self.frac_con.iloc[idx, 1:], dtype=float)
#
#
#
#         # y = self.labels.iloc[idx, 1:]
#         # y = np.array([y])
#         # y = y.astype('float')
#
#
#         output = {"bert_input": ts_origin,
#                   "bert_mask": bert_mask,
#                   "class_label": class_label,
#                   "time": doy,
#                   # "frac_con": frac_con
#                   }
#
#
#         return {key: torch.from_numpy(value) for key, value in output.items()}


### in the following version of PretrainDataset class, we DO NOT drop the end of the time series
### we want to double-check that against the model proposed in Paper 1
### also keep at least two years (instead of one year) of the time series
### this is to make very sure that the disturbance event is part of the time series
class PretrainDataset(Dataset):
    """Time Series Disturbance dataset."""

    def __init__(self, labels_ohe, root_dir, feature_num, seq_len, collist, classifier, data_aug, indices, only_indices): # frac_con,
        """
        Args:
            labels_ohe (string): dataframe with integer class labels
            root_dir (string): Directory with all the time series files.
        """
        self.labels = labels_ohe
        self.root_dir = root_dir
        # self.transform = transform
        # self.rescale = rescale
        self.collist = collist
        self.seq_len = seq_len
        self.dimension = feature_num
        self.classifier = classifier
        self.data_aug = data_aug
        self.indices = indices
        self.only_indices = only_indices
        # self.frac_con = frac_con

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

            # if X.isnull().sum().sum() > 0:
            #     print('NA value found!')
            #     print(X.isnull().sum().sum())
            #     print(X.columns.to_series()[np.isnan(X).any()])
            #     print(X.index[np.isnan(X).any(1)])
            # if np.isinf(X).values.sum() > 0:
            #     print('X contains Inf values: N = ')
            #     print(np.isinf(X).values.sum())
            #     print(X.columns.to_series()[np.isinf(X).any()])
            #     print(X.index[np.isinf(X).any(1)])

        ### convert to SITS-BERT input format

        if self.only_indices:
            X = X.drop(labels=self.collist[0:10], axis=1)

        X = X.unstack().to_frame().T
        X = X.reset_index(drop=True)
        line_data = X.values.flatten().tolist()
        line_data = np.array(line_data, dtype=float)

        ### extract and reshape time series
        ts = np.reshape(line_data, (self.dimension + 1, -1)).T

        ### implement data augmentation by randomly changing sequence length (= window slicing)
        if self.data_aug:
            # get max doy
            max_doy = ts[-1, -1]

            # start_or_end = bool(random.getrandbits(1))

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

        # ### if self.seq_len < ts_length
        # ### use truncating instead of padding
        # if self.seq_len < ts_length:
        #     ts_length = self.seq_len

        ### we always take the LATEST seq_len observations,
        ### (in the original paper/code, it has been the FIRST seq_len observations)
        ### since they are considered most important
        ### padding or truncating
        bert_mask = np.zeros((self.seq_len,), dtype=int)
        bert_mask[:ts_length] = 1

        ### day of year
        doy = np.zeros((self.seq_len,), dtype=int)

        # BOA reflectances
        ts_origin = np.zeros((self.seq_len, self.dimension))
        if self.seq_len > ts_length:
            ts_origin[:ts_length, :] = ts[:ts_length, :-1] / 10000.0 # divide by 10k dropped because of z-transform
            doy[:ts_length] = np.squeeze(ts[:ts_length, -1])

            ### apply z-transformation on each band individually
            ### note that we can leave the DOY values as they are,
            ### since they are being transformed later in PositionalEncoding
            # nonzero_length = np.count_nonzero(ts[:, :ts.shape[1]-1]) / 10
            ### apply it only for non-masked part
            ts_origin[:ts_length, :] = (ts_origin[:ts_length, :] - ts_origin[:ts_length, :].mean(axis=0)) / (ts_origin[:ts_length, :].std(axis=0) + 1e-6)

            if self.data_aug:

                ### data augmentation 3
                ### 1. slightly change DOY, but keep the range
                ### range [-5, 5] will make difference of one satellite scene
                ### this could also be dropped, since window warping
                ### induces a change of DOY already
                doy_noise = np.random.randint(-5, 5, doy[:ts_length].shape[0])
                minimum = doy[:ts_length].min()
                maximum = doy[:ts_length].max()
                doy[:ts_length] = doy[:ts_length] + doy_noise
                doy[:ts_length] = np.clip(doy[:ts_length], minimum, maximum)

                ### data augmentation 4
                ### 2. add a bit of noise to every value with respect to standard deviation
                noise = np.random.normal(0, .1, ts_origin[:ts_length, :].shape)

                # test = ts_origin[:ts_length, :] + noise
                # plt.plot(doy[:ts_length], test[:, 0])
                # plt.plot(doy[:ts_length], ts_origin[:ts_length, 0])
                # plt.show()

                ts_origin[:ts_length, :] = ts_origin[:ts_length, :] + noise

        else:
            ts_origin[:self.seq_len, :] = ts[:self.seq_len, :-1] / 10000.0 # divide by 10k dropped because of z-transform
            doy[:self.seq_len] = np.squeeze(ts[:self.seq_len, -1])


            ### apply z-transformation on each band individually
            ### note that we can leave the DOY values as they are,
            ### since they are being transformed later in PositionalEncoding
            # nonzero_length = np.count_nonzero(ts[:, :ts.shape[1]-1]) / 10
            ### apply it only for non-masked part
            ts_origin[:self.seq_len, :] = (ts_origin[:self.seq_len, :] - ts_origin[:self.seq_len, :].mean(axis=0)) / (ts_origin[:self.seq_len, :].std(axis=0) + 1e-6)

            if self.data_aug:

                ### data augmentation 3
                ### 1. slightly change DOY, but keep the range
                ### this could also be dropped, since window warping
                ### induces a change of DOY already
                doy_noise = np.random.randint(-3, 3, doy[:self.seq_len].shape[0])
                minimum = doy[:self.seq_len].min()
                maximum = doy[:self.seq_len].max()
                doy[:self.seq_len] = doy[:self.seq_len] + doy_noise
                doy[:self.seq_len] = np.clip(doy[:self.seq_len], minimum, maximum)

                ### data augmentation 4
                ### 2. add a bit of noise (here: 1/20 of std of signal) to every value with respect to standard deviation
                noise = np.random.normal(0, .1, ts_origin[:self.seq_len, :].shape)
                ts_origin[:self.seq_len, :] = ts_origin[:self.seq_len, :] + noise


        ### get class label
        if self.classifier:
            class_label = np.array(self.labels.iloc[idx, 1:], dtype=int)
        else:
            class_label = np.array(self.labels.iloc[idx, 1:], dtype=float)


        # ### get auxiliary information (fraction of coniferous forest)
        # frac_con = np.array(self.frac_con.iloc[idx, 1:], dtype=float)



        # y = self.labels.iloc[idx, 1:]
        # y = np.array([y])
        # y = y.astype('float')


        output = {"bert_input": ts_origin,
                  "bert_mask": bert_mask,
                  "class_label": class_label,
                  "time": doy,
                  # "frac_con": frac_con
                  }


        return {key: torch.from_numpy(value) for key, value in output.items()}

class PredictDataset(Dataset):
    """Time Series Disturbance dataset."""

    def __init__(self, root_dir, feature_num, seq_len, collist, classifier, data_aug):
        """
        Args:
            labels_ohe (string): dataframe with integer class labels
            root_dir (string): Directory with all the time series files.
        """
        self.root_dir = root_dir
        self.collist = collist
        self.seq_len = seq_len
        self.dimension = feature_num
        self.classifier = classifier
        self.data_aug = data_aug

    def __len__(self):
        return len(glob.glob(os.path.join(self.root_dir + '*.{}'.format('csv'))))  # number of samples in the dataset

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        X_name = glob.glob(os.path.join(self.root_dir + '*.{}'.format('csv')))[idx]

        # col_list specifies the dataframe columns to use as model input
        X = pd.read_csv(X_name, sep=',', usecols=self.collist)

        ### convert to SITS-BERT input format
        X = X.unstack().to_frame().T
        X = X.reset_index(drop=True)
        line_data = X.values.flatten().tolist()
        line_data = np.array(line_data, dtype=float)

        ### extract and reshape time series
        ts = np.reshape(line_data, (self.dimension + 1, -1)).T

        # get number of observations for further processing
        ts_length = ts.shape[0]

        # ### if self.seq_len < ts_length
        # ### use truncating instead of padding
        # if self.seq_len < ts_length:
        #     ts_length = self.seq_len

        ### we always take the LATEST seq_len observations,
        ### (in the original paper/code, it has been the FIRST seq_len observations)
        ### since they are considered most important
        ### padding or truncating
        bert_mask = np.zeros((self.seq_len,), dtype=int)
        bert_mask[-ts_length:] = 1

        ### day of year
        doy = np.zeros((self.seq_len,), dtype=int)

        # BOA reflectances
        ts_origin = np.zeros((self.seq_len, self.dimension))
        if self.seq_len > ts_length:
            ###
            ts_origin[-ts_length:, :] = ts[-ts_length:, :-1] / 10000.0 # divide by 10k dropped because of z-transform
            doy[-ts_length:] = np.squeeze(ts[-ts_length:, -1])
        else:
            ts_origin[-self.seq_len:, :] = ts[-self.seq_len:, :-1] / 10000.0 # divide by 10k dropped because of z-transform
            doy[-self.seq_len:] = np.squeeze(ts[-self.seq_len:, -1])


        ### apply z-transformation on each band individually
        ### note that we can leave the DOY values as they are,
        ### since they are being transformed later in PositionalEncoding
        # nonzero_length = np.count_nonzero(ts[:, :ts.shape[1]-1]) / 10
        ts_origin = (ts_origin - ts_origin.mean(axis=0)) / (ts_origin.std(axis=0) + 1e-6)

        # y = self.labels.iloc[idx, 1:]
        # y = np.array([y])
        # y = y.astype('float')

        output = {"bert_input": ts_origin,
                  "bert_mask": bert_mask,
                  "time": doy
                  }

        return {key: torch.from_numpy(value) for key, value in output.items()}

class PositionalEncoding(nn.Module):

    # max_len = max days in time series = max value of DOY column
    # now arbitrarily set to 5 years (because DOY cannot be higher in our time series)
    def __init__(self, d_model, max_len=1825):
        super().__init__()

        # Compute the positional encodings once in log space.
        pe = torch.zeros(max_len+1, d_model).float()
        pe.require_grad = False

        position = torch.arange(0, max_len).float().unsqueeze(1)         # [max_len, 1]
        div_term = (torch.arange(0, d_model, 2).float() * -(math.log(10000.0) / d_model)).exp()  # [d_model/2,]

        pe[1:, 0::2] = torch.sin(position * div_term)   # broadcasting to [max_len, d_model/2]
        pe[1:, 1::2] = torch.cos(position * div_term)   # broadcasting to [max_len, d_model/2]

        # ### use last dimension of pe to implement a linear term as in Time2Vec
        # ### pseudo code:
        # #pe[last dimension] = torch.matmul(doy.float(), w0) + b0
        # # w0 dim [seq_len, 1], b0 dim [seq_len, 1]
        # self.w0 = nn.parameter.Parameter(torch.randn(max_len, 1))
        # self.b0 = nn.parameter.Parameter(torch.randn(max_len, 1))
        # # replace one sine and one cosine function result
        # # use linear term instead
        # min = torch.min(position.float().squeeze())
        # max = torch.max(position.float().squeeze())
        # pe[1:, d_model - 1] = torch.matmul(
        #     (position.float().squeeze() - min) / (max - min), self.w0.squeeze()) + self.b0.squeeze()
        # pe[1:, d_model - 2] = torch.matmul(
        #     (position.float().squeeze() - min) / (max - min), self.w0.squeeze()) + self.b0.squeeze()


        # ### this line was added from a blog post, might be false in our context
        # pe = pe.unsqueeze(0).transpose(0, 1)

        self.register_buffer('pe', pe)

    def forward(self, doy):
        return self.pe[doy, :]

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

class LayerNorm(nn.Module):
    "Construct a layernorm module (See citation for details)."

    def __init__(self, features, eps=1e-6):
        super(LayerNorm, self).__init__()
        self.a_2 = nn.Parameter(torch.ones(features))
        self.b_2 = nn.Parameter(torch.zeros(features))
        self.eps = eps

    def forward(self, x):
        mean = x.mean(-1, keepdim=True)
        std = x.std(-1, keepdim=True)
        return self.a_2 * (x - mean) / (std + self.eps) + self.b_2

class GELU(nn.Module):
    """
    Paper Section 3.4, last paragraph notice that BERT used the GELU instead of RELU
    """

    def forward(self, x):
        return 0.5 * x * (1 + torch.tanh(math.sqrt(2 / math.pi) * (x + 0.044715 * torch.pow(x, 3))))

class PositionwiseFeedForward(nn.Module):
    "Implements FFN equation."

    def __init__(self, d_model, d_ff, dropout=0.1):
        super(PositionwiseFeedForward, self).__init__()
        self.w_1 = nn.Linear(d_model, d_ff)
        self.w_2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)
        self.activation = GELU()

    def forward(self, x):
        return self.w_2(self.dropout(self.activation(self.w_1(x))))

class SublayerConnection(nn.Module):
    """
    A residual connection followed by a layer norm.
    Note for code simplicity the norm is first as opposed to last.
    """

    def __init__(self, size, dropout):
        super(SublayerConnection, self).__init__()
        # self.norm = LayerNorm(size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, sublayer):
        "Apply residual connection to any sublayer with the same size."
        # return x + self.dropout(sublayer(self.norm(x)))
        return x + self.dropout(sublayer(x))

class Attention(nn.Module):
    """
    Compute 'Scaled Dot Product Attention
    """

    def forward(self, query, key, value, mask=None, dropout=None):
        scores = torch.matmul(query, key.transpose(-2, -1)) \
                 / math.sqrt(query.size(-1))

        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)

        p_attn = F.softmax(scores, dim=-1)

        if dropout is not None:
            p_attn = dropout(p_attn)

        return torch.matmul(p_attn, value), p_attn

class MultiHeadedAttention(nn.Module):
    """
    Take in model size and number of heads.
    """

    def __init__(self, h, d_model, dropout=0.1):
        super().__init__()
        assert d_model % h == 0

        # We assume d_v always equals d_k
        self.d_k = d_model // h
        self.h = h

        self.linear_layers = nn.ModuleList([nn.Linear(d_model, d_model) for _ in range(3)])
        self.output_linear = nn.Linear(d_model, d_model)
        self.attention = Attention()

        self.dropout = nn.Dropout(p=dropout)

    def forward(self, query, key, value, mask=None):
        batch_size = query.size(0)

        # 1) Do all the linear projections in batch from d_model => h x d_k
        query, key, value = [l(x).view(batch_size, -1, self.h, self.d_k).transpose(1, 2)
                             for l, x in zip(self.linear_layers, (query, key, value))]

        # 2) Apply attention on all the projected vectors in batch.
        x, attn = self.attention(query, key, value, mask=mask, dropout=self.dropout)

        # 3) "Concat" using a view and apply a final linear.
        x = x.transpose(1, 2).contiguous().view(batch_size, -1, self.h * self.d_k)

        return self.output_linear(x)

class TransformerBlock(nn.Module):

    def __init__(self, hidden, attn_heads, feed_forward_hidden, dropout):
        """
        :param hidden: hidden size of transformer
        :param attn_heads: head sizes of multi-head attention
        :param feed_forward_hidden: feed_forward_hidden, usually 4*hidden_size
        :param dropout: dropout rate
        """

        super().__init__()
        self.attention = MultiHeadedAttention(h=attn_heads, d_model=hidden)
        self.feed_forward = PositionwiseFeedForward(d_model=hidden, d_ff=feed_forward_hidden, dropout=dropout)
        self.input_sublayer = SublayerConnection(size=hidden, dropout=dropout)
        self.output_sublayer = SublayerConnection(size=hidden, dropout=dropout)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x, mask):
        x = self.input_sublayer(x, lambda _x: self.attention.forward(_x, _x, _x, mask=mask))
        x = self.output_sublayer(x, self.feed_forward)
        return self.dropout(x)

class SBERT(nn.Module):

    def __init__(self, num_features, hidden, n_layers, attn_heads, dropout=0.1,
                 hidden_clfr_head=None, cnn_embedding=None):
        """
        :param num_features: number of input features
        :param hidden: hidden size of the SITS-BERT model
        :param n_layers: numbers of Transformer blocks (layers)
        :param attn_heads: number of attention heads
        :param dropout: dropout rate
        """

        super().__init__()
        self.hidden = hidden
        self.n_layers = n_layers
        self.attn_heads = attn_heads

        self.feed_forward_hidden = hidden * 4

        self.hidden_clfr_head = hidden_clfr_head
        self.cnn_embedding = cnn_embedding

        self.embedding = BERTEmbedding(num_features, int(hidden/2), self.cnn_embedding)

        self.transformer_blocks = nn.ModuleList(
            [TransformerBlock(hidden, attn_heads, hidden * 4, dropout) for _ in range(n_layers)])



    def forward(self, x, doy, mask):
        mask = (mask > 0).unsqueeze(1).repeat(1, x.size(1), 1).unsqueeze(1)

        x = self.embedding(input_sequence=x, doy_sequence=doy)

        for transformer in self.transformer_blocks:
            x = transformer.forward(x, mask)

        return x


class SBERTRegression(nn.Module):
    """
    Downstream task: Satellite Time Series Classification
    """

    def __init__(self, sbert: SBERT, num_classes, seq_len):
        super().__init__()
        self.sbert = sbert
        self.max_len = seq_len
        self.pooling = nn.MaxPool1d(self.max_len)
        self.linear = nn.Linear(self.sbert.hidden, num_classes)
        # Softmax activation: outputs in range [0, 1] and summing up to 1
        self.softmax = nn.Softmax(dim=1)
        # self.classification = MulticlassClassification(self.sbert.hidden, num_classes, seq_len)

    def forward(self, x, doy, mask):
        # dimensions of x: [batch_size, sequence_length, num_classes]
        x = self.sbert(x, doy, mask)
        x = self.pooling(x.permute(0, 2, 1)).squeeze()
        x = self.linear(x)
        x = self.softmax(x)
        return x
        # return self.classification(x, mask)
        # Dongshen version:
        # seq, batch = output.size(0), output.size(1)
        # output = output.view([seq, -1])
        # # output: [seq_len, batch_size * dim_embd]
        # fc = nn.Linear(self.d_model * batch, batch).to(device)
        # output = self.softmax(fc(output))
        # return output

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
        ###### the following (commented out) code is likely obsolete, because it was a fix
        ###### for a hard-coding issue in original code (solved by MaxPool1d(self.max_len))
        ###### still, we keep the code and comments for the time until all code is settled
        ### permute x again to get proper input dimensions for PyTorch linear layer
        ### the main issue here seems to be the way (the order) of inputs PyTorch layers accept
        ### from the documentation of nn.Linear:
        ### input = torch.randn(batch_size, hidden_layer_size, pooled_sequence_length)
        ### nn.Linear(input_size, output_size)
        ### results in torch.Size([batch_size, pooled_sequence_length, output_size(==num_classes))
        ### it seems like the output of the preceeding funtion (before MultiClassClassification)
        ### is sized just to meet the requirements of nn.Linear in MultiClassClassification
        # x = x.permute(0, 2, 1)
        if self.hidden_clfr_head:
            for layer in self.linear[:-1]:
                x = self.relu(layer(x))
            out = self.linear[-1](x) # with BCELoss: torch.sigmoid(self.linear[-1](x))
            # we can drop sigmoid function as last layer if we use BCEWithLogitsLoss
            return out
        else:
            ### original code:
            x = self.linear(x)
            return x

def kappa(matrix):
    n = np.sum(matrix)
    sum_po = 0
    sum_pe = 0
    for i in range(len(matrix[0])):
        sum_po += matrix[i][i]
        row = np.sum(matrix[i, :])
        col = np.sum(matrix[:, i])
        sum_pe += row * col
    po = sum_po / n
    pe = sum_pe / (n * n)
    return (po - pe) / (1 - pe)

def average_accuracy(matrix):
    correct = np.diag(matrix)
    all = matrix.sum(axis=0)
    accuracy = correct / all
    aa = np.average(accuracy)
    return aa

class SBERTFineTuner:
    def __init__(self, sbert: SBERT, num_classes: int, classifier: bool,
                 train_dataloader: DataLoader, valid_dataloader: DataLoader,
                 seq_len: int, criterion,
                 lr: float = 1e-5, with_cuda: bool = True,
                 cuda_devices=None, log_freq: int = 100):

        cuda_condition = torch.cuda.is_available() and with_cuda
        self.device = torch.device("cuda" if cuda_condition else "cpu")

        self.sbert = sbert
        self.classifier = classifier
        if self.classifier:
            self.model = SBERTClassification(sbert, num_classes, seq_len).to(self.device)
        else:
            self.model = SBERTRegression(sbert, num_classes, seq_len).to(self.device)
        self.num_classes = num_classes

        if with_cuda and torch.cuda.device_count() > 1:
            print("Using %d GPUs for model fine-tuning" % torch.cuda.device_count())
            self.model = nn.DataParallel(self.model, device_ids=cuda_devices)

        self.train_dataloader = train_dataloader
        self.valid_dataloader = valid_dataloader

        self.optim = Adam(self.model.parameters(), lr=lr)
        # # https://pytorch.org/docs/stable/generated/torch.optim.lr_scheduler.OneCycleLR.html#torch.optim.lr_scheduler.OneCycleLR
        # # alternative from Keras: https://towardsdatascience.com/the-time-series-transformer-2a521a0efad3
        # # but the latter operates over epochs
        # # or StepLR in pytorch: https://stackoverflow.com/questions/60050586/pytorch-change-the-learning-rate-based-on-number-of-epochs
        # # this: https://stackoverflow.com/questions/65343377/adam-optimizer-with-warmup-on-pytorch
        # # is very helpful as well, but I think also within (!) epochs (= Vaswani et al.),
        # # not executing the step after each epoch
        # self.scheduler = OneCycleLR(
        #     self.optim, max_lr=0.01,
        #     steps_per_epoch=len(self.train_dataloader),
        #     epochs=NUM_EPOCHS)
        self.criterion = criterion

        self.log_freq = log_freq

    def train(self, epoch):
        train_loss = 0.0
        counter = 0

        if self.classifier:
            total_correct = 0
            total_element = 0
            ### the +1 is necessary in binary classification
            ### remove in case of multiclass classification
            matrix = np.zeros([self.num_classes+1, self.num_classes+1])
            for data in self.train_dataloader:
                data = {key: value.to(self.device) for key, value in data.items()}

                classification = self.model(data["bert_input"].float(),
                                            data["time"].long(),
                                            data["bert_mask"].long())

                ### original code line:
                # loss = self.criterion(classification, data["class_label"].squeeze().long())
                ### leads to error: RuntimeError: Expected target size [32, 1], got [32]
                ### therefore, .squeeze() was removed (in case of binary classification)
                ### in case of three classes: classification tensor has size (32,4,3), size needed is (32,3)
                ### the "4" is pooled_sequence_length -> how to get rid of this?
                ### see MultiClassClassification for the solution (hard-coded number 64 in original code)
                # print(data["class_label"].long().shape)
                ### added '.squeeze()' to get the dimension correctly
                # loss = self.criterion(classification, data["class_label"].long().squeeze())
                loss = self.criterion(classification, data["class_label"].float())
                self.optim.zero_grad()
                loss.backward()
                # ### pushing model to cuda is only necessary because of self-made time2vec linear implementation
                # self.model.to(self.device)
                self.optim.step()
                train_loss += loss.item()

                # classification_result = [torch.ones(1) if pred >= 0.5 else torch.zeros(1) for pred in classification]

                # # https://discuss.pytorch.org/t/calculate-accuracy-in-binary-classification/91758
                # # this code works fine, but is not optimal in our use case
                # # keep it commented out for the time
                # train_OA = get_binary_accuracy(data['class_label'].squeeze(), classification.squeeze())

                # classification_result = classification # .argmax(dim=-1)
                classification_result = classification.squeeze()
                classification_result = classification_result > 0.5  # gives boolean output
                classification_target = data["class_label"].squeeze()
                ### boolean output can still be compared to 0 and 1 from targets
                correct = classification_result.eq(classification_target).sum().item()

                total_correct += correct
                total_element += data["class_label"].nelement()
                for row, col in zip(classification_result, classification_target):
                    matrix[row, col] += 1

                # # https://pytorch.org/docs/stable/generated/torch.optim.lr_scheduler.OneCycleLR.html#torch.optim.lr_scheduler.OneCycleLR
                # self.scheduler.step()

                counter += 1

            train_loss /= counter
            train_OA = total_correct / total_element * 100
            train_kappa = kappa(matrix)

            valid_loss, valid_OA, valid_kappa = self._validate()

            print("EP%d, train_OA=%.2f, train_Kappa=%.3f, validate_OA=%.2f, validate_Kappa=%.3f"
                  % (epoch, train_OA, train_kappa, valid_OA, valid_kappa))

            return train_OA, train_kappa, valid_OA, valid_kappa, train_loss, valid_loss

        else:
            for data in self.train_dataloader:
                data = {key: value.to(self.device) for key, value in data.items()}

                regression = self.model(data["bert_input"].float(),
                                            data["time"].long(),
                                            data["bert_mask"].long())
                loss = self.criterion(regression, data["class_label"].float()) # .long().squeeze())
                self.optim.zero_grad()
                loss.backward()
                self.optim.step()
                train_loss += loss.item()

                counter += 1

            train_loss /= counter

            valid_loss = self._validate()

            print("EP%d, train_loss=%.5f, validate_loss=%.5f"
                  % (epoch, train_loss, valid_loss))

            return train_loss, valid_loss

    def _validate(self):
        with torch.no_grad():
            self.model.eval()

            valid_loss = 0.0
            counter = 0

            if self.classifier:
                total_correct = 0
                total_element = 0
                matrix = np.zeros([self.num_classes+1, self.num_classes+1])
                for data in self.valid_dataloader:
                    data = {key: value.to(self.device) for key, value in data.items()}

                    classification = self.model(data["bert_input"].float(),
                                                data["time"].long(),
                                                data["bert_mask"].long())

                    loss = self.criterion(classification, data["class_label"].float())
                    ### squeeze has to be dropped for new implementation of binary classification
                    # loss = self.criterion(classification, data["class_label"].squeeze().long())
                    valid_loss += loss.item()

                    classification_result = classification.squeeze()
                    classification_result = classification_result > 0.5  # gives boolean output
                    classification_target = data["class_label"].squeeze()
                    ### boolean output can still be compared to 0 and 1 from targets
                    correct = classification_result.eq(classification_target).sum().item()

                    correct = classification_result.eq(classification_target).sum().item()
                    total_correct += correct
                    total_element += data["class_label"].nelement()
                    for row, col in zip(classification_result, classification_target):
                        matrix[row, col] += 1

                    counter += 1

                valid_loss /= counter
                valid_OA = total_correct / total_element * 100
                valid_kappa = kappa(matrix)

            else:
                for data in self.valid_dataloader:
                    data = {key: value.to(self.device) for key, value in data.items()}

                    regression = self.model(data["bert_input"].float(),
                                            data["time"].long(),
                                            data["bert_mask"].long())

                    loss = self.criterion(regression, data["class_label"].float()) # .squeeze().long()
                    valid_loss += loss.item()

                    counter += 1

                valid_loss /= counter

            self.model.train()

            if self.classifier:
                return valid_loss, valid_OA, valid_kappa
            else:
                return valid_loss

    def test(self, data_loader):
        with torch.no_grad():
            self.model.eval()

            # ### testing done on cpu?
            # self.device -> torch.device(‚cpu‘)

            if self.classifier:
                total_correct = 0
                total_element = 0
                matrix = np.zeros([self.num_classes+1, self.num_classes+1])

                ### intialize dict that contains the labels and predictions for later analyses
                test_result = dict(label=[], prediction=[], raw_output=[])  # plotID=[],
                test_labels = []
                test_preds = []
                test_raw_output = []

                for data in data_loader:
                    data = {key: value.to(self.device) for key, value in data.items()}

                    ### do we need to apply sigmoid here because of BCEWithLogitsLoss?
                    # https://discuss.pytorch.org/t/playing-with-bcewithlogitsloss/82673/2
                    ### I think using torch.sigmoid here does not make sense,
                    ### since output is already in [0, 1] range (= probabilities)
                    # https://discuss.pytorch.org/t/bceloss-vs-bcewithlogitsloss/33586/27
                    classification = torch.sigmoid(self.model(data["bert_input"].float(),
                                        data["time"].long(),
                                        data["bert_mask"].long()
                                        ))

                    classification_result = classification.squeeze()
                    classification_class = classification_result > 0.5  # gives boolean output
                    classification_target = data["class_label"].squeeze()
                    ### boolean output can still be compared to 0 and 1 from targets
                    correct = classification_class.eq(classification_target).sum().item()

                    ### save label and prediction
                    # get only the tensor values, not the attributes (such as device information)
                    # https://stackoverflow.com/questions/57727372/how-do-i-get-the-value-of-a-tensor-in-pytorch
                    test_labels.append(classification_target.cpu().numpy())
                    test_preds.append(classification_class.cpu().numpy())
                    test_raw_output.append(classification_result.cpu().numpy())

                    # correct = classification_result.eq(classification_target).sum().item()

                    total_correct += correct
                    total_element += data["class_label"].nelement()
                    ## an error is thrown here in case of batch_size = 1
                    ## so we keep batch_size = BATCH_SIZE, cause the results are essentially the same
                    for row, col in zip(classification_class, classification_target):
                        matrix[row, col] += 1

                test_OA = total_correct * 100.0 / total_element
                test_kappa = kappa(matrix)
                test_AA = average_accuracy(matrix)

                test_result['label'] = [j for i in test_labels for j in i]
                test_result['prediction'] = [j for i in test_preds for j in i]
                test_result['raw_output'] = [j for i in test_raw_output for j in i]
            else:
                test_loss = 0.0
                counter = 0

                ### intialize dict that contains the labels and predictions for later analyses
                test_result = dict(label=[], prediction=[])  # plotID=[],
                test_labels = []
                test_preds = []

                for data in data_loader:
                    data = {key: value.to(self.device) for key, value in data.items()}

                    regression = self.model(data["bert_input"].float(),
                                        data["time"].long(),
                                        data["bert_mask"].long())


                    regression_result = regression # .argmax(dim=-1)
                    regression_target = data["class_label"] # .squeeze()

                    ### save label and prediction
                    # get only the tensor values, not the attributes (such as device information)
                    # https://stackoverflow.com/questions/57727372/how-do-i-get-the-value-of-a-tensor-in-pytorch
                    test_labels.append(regression_target.cpu().numpy().tolist())
                    test_preds.append(regression_result.cpu().numpy().tolist())

                    test_result['label'] = [j for i in test_labels for j in i]
                    test_result['prediction'] = [j for i in test_preds for j in i]

                    loss = self.criterion(regression, data["class_label"].float()) # .squeeze().long()
                    test_loss += loss.item()

                    counter += 1

                test_loss /= counter

        self.model.train()

        if self.classifier:
            return test_OA, test_kappa, test_AA, matrix, test_result
        else:
            return test_loss, test_result

    # def explain(self, data_loader):
    #     with torch.enable_grad():
    #         self.model.eval()
    #         self.model.zero_grad()
    #
    #         if self.classifier:
    #
    #             for data in data_loader:
    #                 data = {key: value.to(self.device) for key, value in data.items()}
    #
    #                 ### note that the embedding layer is apparently called:
    #                 ### self.model.sbert.embedding
    #                 ### this will be important for interpretable embedding layer
    #                 interpretable_emb = configure_interpretable_embedding_layer(self.model,
    #                                                                             'sbert.embedding')
    #
    #                 ### do NOT pass the two inputs input_sequence and doy_sequence as tuple,
    #                 ### because the FORWARD function in BERTEmbedding takes two separate inputs
    #                 ### it has nothing to do with the constructor (lesson learned)
    #                 input_emb = interpretable_emb.indices_to_embeddings(
    #                     data["bert_input"].float()[1, :, :].unsqueeze(axis=0),
    #                      data["time"].long()[1, :].unsqueeze(axis=0))
    #
    #                 ### initialize IntegratedGradients
    #                 ig = IntegratedGradients(self.model, multiply_by_inputs=False)
    #
    #                 ### allow_unused=True because one of the tensors is not being used in the graph?
    #                 ### other (better?) idea:
    #                 ### use additional_forward_args argument, because
    #                 ### 1. bert_mask is not relevant for the computation of the graph,
    #                 ### since it is only the mask for relevant data
    #                 ### 2. data['time'] is part of input_emb already,
    #                 ### so we do not need it for the following attention layers anymore
    #                 ### still, all the inputs are needed so that the model has all the information
    #                 ### only the inputs to input_emb (input_sequence = satellite time series values
    #                 ### and doy_sequence = dates of observations) are relevant for the comp. graph
    #                 ### note that in this binary classification, passing a target is not necessary
    #                 attribution = ig.attribute(input_emb,
    #                                            additional_forward_args=
    #                                            (data["bert_mask"].long()[1, :].unsqueeze(axis=0),
    #                                             data["time"].long()[1, :].unsqueeze(axis=0)))
    #
    #                 ### I think this shows that it works, but as explaind here:
    #                 ### https://github.com/pytorch/captum/issues/439#issuecomment-680950985
    #                 ### we only have the attributions for the EMBEDDINGS now,
    #                 ### not for the input features,
    #                 ### we have to go for LayerIntegratedGradients for more details, I guess
    #
    #
    #                 ### now reverting back to this tutorial:
    #                 # https://gist.github.com/smaeland/f132ae58db4aa709d92d49bf2bf19d58
    #                 ### this approach is likely able to give ATTRIBUTIONS FOR TIME STEPS
    #                 ### but NOT for SATELLITE CHANNELS
    #                 ### for the channels, we probably have to use LayerIntegratedGradients
    #                 ### maybe we can combine the information later?
    #
    #                 ### in this example, try to sum over time steps,
    #                 ### then plot the attribution score for each time step
    #
    #                 ### - for visualization, we need the time series of the 10 channels as x
    #                 ### - sum all attribution values over each time step
    #                 ###   (think about dismissing the doy_sequence attributions,
    #                 ###   since they do not refer to actual input data)
    #                 ### - broadcast the result (that has only 1 "channel")
    #                 ###   to num_channels
    #                 ### - overlay the same summed attributions over all individual channels
    #                 ### -> this way, we get information about WHICH TIMESTEP is IMPORTANT
    #                 ###    but NOT which CHANNEL is IMPORTANT
    #
    #                 # Remove batch dimension for plotting
    #                 attribution = torch.squeeze(attribution, dim=0)
    #                 # ### slice the attribution tensor to only first half of second dimension
    #                 # ### this means deliberately excluding the PositionalEncoding from attribution
    #                 # attribution = attribution[:, :128]  # [seq_length, embedding_dim]
    #                 # we do not do this for now, since I think the positional embeddings
    #                 # have a meaning themselves as well, and we should consider them
    #
    #                 ### sum over second dimension (first are time steps)
    #                 ### we will use the absolute values, so that pos and neg attributions
    #                 ### do not average out
    #                 attribution = torch.sum(torch.abs(attribution), dim=1, keepdim=True)
    #                 attribution = attribution / torch.norm(attribution)  # added and not tested on 2023-05-09
    #                 # attribution = torch.sum(attribution, dim=1, keepdim=True)
    #
    #                 ### duplicate 10 times for plotting each channel
    #                 attribution = attribution.repeat(1, 10)
    #
    #                 ### convert to numpy
    #                 attribution = attribution.detach().cpu().numpy()
    #
    #                 ### get input sequence
    #                 x = data["bert_input"].float()[1, :, :].detach().cpu().numpy()
    #
    #                 ### get positions for plotting
    #                 dates = data["time"].long()[1, :].detach().cpu().numpy()
    #
    #                 ### cut all data by dates[dates == 0]
    #                 # mask = data["bert_mask"].long()[1, :].detach().cpu().numpy()
    #                 arraymin = np.amin(np.nonzero(dates))
    #                 ### slice all arrays
    #                 attribution = attribution[arraymin:, :]
    #                 x = x[arraymin:, :]
    #                 dates = dates[arraymin:]
    #                 # dates = np.repeat(dates[:, np.newaxis], 10, axis=1)
    #
    #                 ### now visualize the result
    #                 viz.visualize_timeseries_attr_cs(
    #                     attribution,
    #                     x,
    #                     # # this checks dates.shape[0] == attribution.shape[1]
    #                     # # which is currently nonzero seq_length = 52 = num_channels = 10
    #                     # # think about what v_values is actually supposed to do
    #                     # x_values=dates,
    #                     method="overlay_combined",
    #                     sign='absolute_value',
    #                     channel_labels=['BLU', 'GRN', 'RED', 'RE1', 'RE2', 'RE3', 'BNR', 'NIR', 'SW1', 'SW2'],
    #                     channels_last=True,
    #                     show_colorbar=True,
    #                     title='Example plot for testing Captum IntegratedGradients across Embeddings',
    #                     alpha_overlay=0.5,
    #                     fig_size=(10, 5),
    #                 )
    #
    #
    #                 ### after finishing the interpretation, we need to remove
    #                 ### interpretable embedding layer with the following command
    #                 remove_interpretable_embedding_layer(self.model, interpretable_emb)
    #
    #                 # classification_result = self.model(data["bert_input"].float(),
    #                 #                     data["time"].long(),
    #                 #                     data["bert_mask"].long()
    #                 #                     ) # .argmax(dim=-1)
    #                 #
    #                 # # classification_result = result.argmax(dim=-1)
    #                 # # classification_target = data["class_label"].squeeze()
    #                 #
    #                 # # dl = Saliency(self.model)
    #                 # # ### deeplift is somewhat more complicated than other algorithms
    #                 # # ### and more prone to making mistakes
    #                 # # # see https://captum.ai/docs/faq#how-do-i-set-the-target-parameter-to-an-attribution-method
    #                 # # # question 'can my model use functional non-linearities...?'
    #                 # # dl = DeepLift(self.model, multiply_by_inputs=False)
    #                 # dl = IntegratedGradients(self.model, multiply_by_inputs=False)
    #                 # ### get attributions for this sample
    #                 # # x = [] ### list comprehension that adds unsqueeze(axis=0) to every value (= tensor) in dict
    #                 # # x = {key: value.to(device).unsqueeze(axis=0) for key, value in data.items()}
    #                 #
    #                 # ### idea: https://discuss.pytorch.org/t/how-the-pytorch-freeze-network-in-some-layers-only-the-rest-of-the-training/7088/13
    #                 # ### -> also unfreeze the optimizer!? or pass unfrozen layers to optimizer!?
    #                 #
    #                 # ### another idea: somehow get to unfreeze positional encoding
    #                 #
    #                 # ### this seems to work!?!?!?
    #                 # for param in self.model.parameters():
    #                 #     param.requires_grad = True
    #                 #     print(param)
    #                 #
    #                 # ### may it be that there are no gradients at all?
    #                 # ### can I see them?
    #                 #
    #                 # attrs = dl.attribute((data["bert_input"].float()[1, :, :].unsqueeze(axis=0),
    #                 #                     data["time"].long()[1, :].unsqueeze(axis=0),
    #                 #                     data["bert_mask"].long()[1, :].unsqueeze(axis=0)
    #                 #                       ))
    #                 # ### detach from GPU and convert to numpy array
    #                 # attrs = attrs.detach().numpy()
    #                 #
    #                 # ### now visualize

    def save(self, epoch, file_path, model_name):
        output_path = file_path + model_name + ".tar"
        torch.save({
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optim.state_dict(),
        }, output_path)

        print("EP:%d Model Saved on:" % epoch, output_path)
        return output_path

    def load(self, file_path, model_name):
        input_path = file_path + model_name + ".tar"

        checkpoint = torch.load(input_path)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optim.load_state_dict(checkpoint['optimizer_state_dict'])
        self.model.train()
        epoch = checkpoint['epoch']

        # ### make sure all layers are frozen:
        # for name, para in self.model.named_parameters():
        #     # print(name)
        #     # print(para)
        #     para.requires_grad = False
        #
        # ### alternatively, freeze only a part of the weights
        # ### e.g. the lower-level representations
        # for name, para in self.model.named_parameters():
        #     # print(name)
        #     # print(para)
        #     if (not name in 'sbert.transformer_blocks.2') and (not name in 'sbert.transformer_blocks.1'): #  and (not name in 'sbert.transformer_blocks.1')
        #         para.requires_grad = True
        #
        # ### unfreeze the last layer
        # self.model.classification.linear.bias.requires_grad = True
        # self.model.classification.linear.weight.requires_grad = True

        print("EP:%d Model loaded from:" % epoch, input_path)
        return input_path

    def predict(self, data_loader):
        # Disable grad
        with torch.no_grad():
            self.model.eval()

            pred_result = dict(prediction=[], raw_output=[])  # plotID=[],
            pred = []
            pred_raw_output = []

            for data in data_loader:
                data = {key: value.to(self.device) for key, value in data.items()}

                ### sigmoid added because:
                ### https://discuss.pytorch.org/t/playing-with-bcewithlogitsloss/82673/2
                classification = torch.sigmoid(
                    self.model(data["bert_input"].float(),
                               data["time"].long(),
                               data["bert_mask"].long()
                               ))

                classification_result = classification.squeeze()
                classification_class = classification_result > 0.5  # gives boolean output

                pred.append(classification_class.cpu().numpy())
                pred_raw_output.append(classification_result.cpu().numpy())

            pred_result['prediction'] = [j for i in pred for j in i]
            pred_result['raw_output'] = [j for i in pred_raw_output for j in i]

        return pred_result



def train_baseline(input_data_path, model_name, collist, datasets, n_jobs, indices, only_indices):
    # search for all classifiers which can handle unequal length data. This may give some
    # UserWarnings if soft dependencies are not installed.
    sktime.registry.all_tags()
    all_estimators(
        filter_tags={"capability:unequal_length": True, "capability:multivariate": True}, estimator_types="classifier"
    )

    # TimeSeriesSVC can handle all of our challenges:
    all_estimators(filter_tags={"capability:unequal_length": True,
                                "capability:multivariate": True},
                   estimator_types="classifier")
    ### options are: nested_univ, df-list, pd-long, alignment, alignment_loc, pd_DataFrame_Table
    sktime.datatypes.MTYPE_REGISTER

    ### load data
    traindat = pd.read_csv(os.path.join(input_data_path, 'split_data', 'train_labels_' + datasets[0] + '.csv'), sep=';')

    # ### sampling for testing the workflow -> remove this for real runs!
    # traindat = traindat.sample(n=1000, random_state=1)

    train_x = traindat['plotID']
    train_y = traindat['max_mort_int']

    testdat = pd.read_csv(os.path.join(input_data_path, 'split_data', 'test_labels_' + datasets[0] + '.csv'), sep=';')

    # ### sampling for testing the workflow -> remove this for real runs!
    # testdat = testdat.sample(n=1000, random_state=1)

    test_x = testdat['plotID']
    test_y = testdat['max_mort_int']

    # collist.remove('DOY')



    ######################################
    ##### TimeSeriesForestClassifier #####
    ######################################

    print('preparing random forest data')
    csv_ts_list = []
    # for csv_index, csv_path_i in enumerate(train_x):
    #     csv_df_i = pd.read_csv(os.path.join(input_data_path, csv_path_i + '.csv'), index_col=False, usecols=collist, parse_dates=False)
    #     csv_ts_i = convert_df_to_sktime(df=csv_df_i, index=csv_index)
    #     csv_ts_list.append(csv_ts_i)

    # ### end padding for the whole sequence after concatenation of the s2 bands
    # padding_size = 256 * 10
    # for csv_index, csv_path_i in enumerate(train_x):
    #     csv_df_i = pd.read_csv(os.path.join(input_data_path, csv_path_i + '.csv'), index_col=False, usecols=collist,
    #                            parse_dates=False)
    #     ls = []
    #     for iter in collist:
    #         ls.append(csv_df_i[iter])
    #     flat_list = [item for sublist in ls for item in sublist]
    #     flat_list = (flat_list + padding_size * [0])[:padding_size]
    #     flat_df = pd.DataFrame(flat_list)
    #     # csv_ts_i = convert_df_to_sktime(df=csv_df_i, index=csv_index)
    #     csv_ts_list.append(flat_df)
    # len(csv_ts_list)

    ### end padding for each s2 band separately
    padding_size = 256 # 256 observation max
    for csv_index, csv_path_i in enumerate(train_x):
        csv_df_i = pd.read_csv(os.path.join(input_data_path, csv_path_i + '.csv'), index_col=False, usecols=collist,
                               parse_dates=False)

        if (indices.lower() == 'true') or only_indices:
            # BLUE = B02, GREEN = B03, RED = B04, RE1 = B05, RE2 = B06, RE3 = B07,
            # NIR = B08, BNIR = B08A, SWIR1 = B11, SWIR2 = B12
            csv_df_i['CRSWIR'] = csv_df_i['SW1_mean'] / (
                        csv_df_i['BNR_mean'] + ((csv_df_i['SW2_mean'] - csv_df_i['BNR_mean']) / (2185.7 - 864)) * (1610.4 - 864))
            EVI = 2.5 * (csv_df_i['NIR_mean'] - csv_df_i['RED_mean']) / (
                        (csv_df_i['NIR_mean'] + 6 * csv_df_i['RED_mean'] - 7.5 * csv_df_i['BLU_mean']) + 1)
            csv_df_i['NBR'] = (csv_df_i['NIR_mean'] - csv_df_i['SW2_mean']) / (csv_df_i['NIR_mean'] + csv_df_i['SW2_mean'])
            csv_df_i['TCW'] = 0.1509 * csv_df_i['BLU_mean'] + 0.1973 * csv_df_i['GRN_mean'] + 0.3279 * csv_df_i['RED_mean'] + 0.3406 * csv_df_i[
                'NIR_mean'] - 0.7112 * csv_df_i['SW1_mean'] - 0.4572 * csv_df_i['SW2_mean']
            TCG = -0.2848 * csv_df_i['BLU_mean'] - 0.2435 * csv_df_i['GRN_mean'] - 0.5436 * csv_df_i['RED_mean'] + 0.7243 * csv_df_i[
                'NIR_mean'] + 0.084 * csv_df_i['SW1_mean'] - 0.18 * csv_df_i['SW2_mean']
            TCB = 0.3037 * csv_df_i['BLU_mean'] + 0.2793 * csv_df_i['GRN_mean'] + 0.4743 * csv_df_i['RED_mean'] + 0.5585 * csv_df_i[
                'NIR_mean'] + 0.5082 * csv_df_i['SW1_mean'] + 0.1863 * csv_df_i['SW2_mean']
            csv_df_i['TCD'] = TCB - (TCG + csv_df_i['TCW'])
            csv_df_i['NDVI'] = (csv_df_i['NIR_mean'] - csv_df_i['RED_mean']) / (csv_df_i['NIR_mean'] + csv_df_i['RED_mean'])
            csv_df_i['NDWI'] = (csv_df_i['BNR_mean'] - csv_df_i['SW1_mean']) / (csv_df_i['BNR_mean'] + csv_df_i['SW1_mean'])
            csv_df_i['NDMI'] = (csv_df_i['NIR_mean'] - csv_df_i['SW1_mean']) / (csv_df_i['NIR_mean'] + csv_df_i['SW1_mean'])
            # https://kaflekrishna.com.np/blog-detail/retrieving-leaf-area-index-lai-sentinel-2-image-google-earth-engine-gee/
            csv_df_i['LAI'] = (3.618 * EVI) - .118
            csv_df_i['MSI'] = csv_df_i['SW1_mean'] / csv_df_i['NIR_mean']
            csv_df_i['NDRE'] = (csv_df_i['NIR_mean'] - csv_df_i['RE1_mean']) / (csv_df_i['NIR_mean'] + csv_df_i['RE1_mean'])
            # csv_df_i['CRE'] = csv_df_i['NIR_mean'] / csv_df_i['RE1_mean'] - 1.0

            # put X['DOY'] at the end of the dataframe
            csv_df_i = csv_df_i.reindex(columns=[col for col in csv_df_i.columns if col != 'DOY'] + ['DOY'])

            ### replace inf values by max of column
            for col in csv_df_i.columns.tolist():
                max_value = np.nanmax(csv_df_i[col][csv_df_i[col] != np.inf])
                csv_df_i[col].replace([np.inf, -np.inf], max_value, inplace=True)
            ### replace NA by 0 values
            csv_df_i.fillna(0, inplace=True)

        ls = []
        if only_indices:
            for iter in csv_df_i.drop(collist, axis=1).columns.tolist():
                ls.append(csv_df_i[iter])
            ls = [(ls[cols].tolist() + padding_size * [0])[:padding_size] for cols in range(len(csv_df_i.drop(collist, axis=1).columns.tolist()))]
        else:
            for iter in csv_df_i.drop('DOY', axis=1).columns.tolist():
                ls.append(csv_df_i[iter])
            ls = [(ls[cols].tolist() + padding_size * [0])[:padding_size] for cols in range(len(csv_df_i.drop('DOY', axis=1).columns.tolist()))]
        flat_list = [item for sublist in ls for item in sublist]
        # flat_list = (flat_list + padding_size * [0])[:padding_size]
        flat_df = pd.DataFrame(flat_list)
        # csv_ts_i = convert_df_to_sktime(df=csv_df_i, index=csv_index)
        csv_ts_list.append(flat_df)
    len(csv_ts_list)

    # csv_ts = pd.concat(csv_ts_list)
    # csv_ts
    # csv_ts_i
    train_x_rf = csv_ts_list
    sktime.datatypes.check_is_mtype(train_x_rf, 'df-list', return_metadata=True)

    # print(train_x_rf)
    # print('fitting PaddingTransformer...')
    # start = time.time()
    # padding_transformer = PaddingTransformer(pad_length=256*10) # 10 channels
    # train_x_rf = padding_transformer.fit_transform(train_x_rf)
    # end = time.time()
    # print(train_x_rf)
    # print('fitting PaddingTransformer finished')
    # print(start - end)
    # print('seconds for padding')

    print('fitting TimeSeriesForestClassifier...')
    print('dataset: ')
    print(datasets[0])
    start = time.time()
    classifier = (# ColumnConcatenator() *
                  # PaddingTransformer() *
                  TimeSeriesForestClassifier(min_interval=10, n_estimators=500, n_jobs=n_jobs, random_state=1))
    # classifier = TimeSeriesForestClassifier(n_jobs=60)
    classifier.fit(train_x_rf, train_y)
    end = time.time()
    print('fitting TimeSeriesForestClassifier finished')
    print(end - start)
    print('seconds for model fitting')

    ### save the random forest model itself to disk
    filename = os.path.join(input_data_path, 'model', 'rf_model_' + model_name + '.pkl')
    # pickle.dump(classifier, open(filename, "wb"))
    ### load model
    classifier = pickle.load(open(filename, "rb"))

    csv_ts_list = []
    # for csv_index, csv_path_i in enumerate(train_x):
    #     csv_df_i = pd.read_csv(os.path.join(input_data_path, csv_path_i + '.csv'), index_col=False, usecols=collist, parse_dates=False)
    #     csv_ts_i = convert_df_to_sktime(df=csv_df_i, index=csv_index)
    #     csv_ts_list.append(csv_ts_i)

    # ### end padding for the whole sequence after concatenation of the s2 bands
    # padding_size = 256 * 10
    # for csv_index, csv_path_i in enumerate(test_x):
    #     csv_df_i = pd.read_csv(os.path.join(input_data_path, csv_path_i + '.csv'), index_col=False, usecols=collist,
    #                            parse_dates=False)
    #     ls = []
    #     for iter in collist:
    #         ls.append(csv_df_i[iter])
    #     flat_list = [item for sublist in ls for item in sublist]
    #     flat_list = (flat_list + padding_size * [0])[:padding_size]
    #     flat_df = pd.DataFrame(flat_list)
    #     # csv_ts_i = convert_df_to_sktime(df=csv_df_i, index=csv_index)
    #     csv_ts_list.append(flat_df)
    # len(csv_ts_list)

    ### end padding for each s2 band separately
    padding_size = 256 # 256 observation max
    for csv_index, csv_path_i in enumerate(test_x):
        csv_df_i = pd.read_csv(os.path.join(input_data_path, csv_path_i + '.csv'), index_col=False, usecols=collist,
                               parse_dates=False)

        if (indices.lower() == 'true') or only_indices:
            # BLUE = B02, GREEN = B03, RED = B04, RE1 = B05, RE2 = B06, RE3 = B07,
            # NIR = B08, BNIR = B08A, SWIR1 = B11, SWIR2 = B12
            csv_df_i['CRSWIR'] = csv_df_i['SW1_mean'] / (
                        csv_df_i['BNR_mean'] + ((csv_df_i['SW2_mean'] - csv_df_i['BNR_mean']) / (2185.7 - 864)) * (1610.4 - 864))
            EVI = 2.5 * (csv_df_i['NIR_mean'] - csv_df_i['RED_mean']) / (
                        (csv_df_i['NIR_mean'] + 6 * csv_df_i['RED_mean'] - 7.5 * csv_df_i['BLU_mean']) + 1)
            csv_df_i['NBR'] = (csv_df_i['NIR_mean'] - csv_df_i['SW2_mean']) / (csv_df_i['NIR_mean'] + csv_df_i['SW2_mean'])
            csv_df_i['TCW'] = 0.1509 * csv_df_i['BLU_mean'] + 0.1973 * csv_df_i['GRN_mean'] + 0.3279 * csv_df_i['RED_mean'] + 0.3406 * csv_df_i[
                'NIR_mean'] - 0.7112 * csv_df_i['SW1_mean'] - 0.4572 * csv_df_i['SW2_mean']
            TCG = -0.2848 * csv_df_i['BLU_mean'] - 0.2435 * csv_df_i['GRN_mean'] - 0.5436 * csv_df_i['RED_mean'] + 0.7243 * csv_df_i[
                'NIR_mean'] + 0.084 * csv_df_i['SW1_mean'] - 0.18 * csv_df_i['SW2_mean']
            TCB = 0.3037 * csv_df_i['BLU_mean'] + 0.2793 * csv_df_i['GRN_mean'] + 0.4743 * csv_df_i['RED_mean'] + 0.5585 * csv_df_i[
                'NIR_mean'] + 0.5082 * csv_df_i['SW1_mean'] + 0.1863 * csv_df_i['SW2_mean']
            csv_df_i['TCD'] = TCB - (TCG + csv_df_i['TCW'])
            csv_df_i['NDVI'] = (csv_df_i['NIR_mean'] - csv_df_i['RED_mean']) / (csv_df_i['NIR_mean'] + csv_df_i['RED_mean'])
            csv_df_i['NDWI'] = (csv_df_i['BNR_mean'] - csv_df_i['SW1_mean']) / (csv_df_i['BNR_mean'] + csv_df_i['SW1_mean'])
            csv_df_i['NDMI'] = (csv_df_i['NIR_mean'] - csv_df_i['SW1_mean']) / (csv_df_i['NIR_mean'] + csv_df_i['SW1_mean'])
            # https://kaflekrishna.com.np/blog-detail/retrieving-leaf-area-index-lai-sentinel-2-image-google-earth-engine-gee/
            csv_df_i['LAI'] = (3.618 * EVI) - .118
            csv_df_i['MSI'] = csv_df_i['SW1_mean'] / csv_df_i['NIR_mean']
            csv_df_i['NDRE'] = (csv_df_i['NIR_mean'] - csv_df_i['RE1_mean']) / (csv_df_i['NIR_mean'] + csv_df_i['RE1_mean'])
            # csv_df_i['CRE'] = csv_df_i['NIR_mean'] / csv_df_i['RE1_mean'] - 1.0

            # put X['DOY'] at the end of the dataframe
            csv_df_i = csv_df_i.reindex(columns=[col for col in csv_df_i.columns if col != 'DOY'] + ['DOY'])

            ### replace inf values by max of column
            for col in csv_df_i.columns.tolist():
                max_value = np.nanmax(csv_df_i[col][csv_df_i[col] != np.inf])
                csv_df_i[col].replace([np.inf, -np.inf], max_value, inplace=True)
            ### replace NA by 0 values
            csv_df_i.fillna(0, inplace=True)

        ls = []
        if only_indices:
            for iter in csv_df_i.drop(collist, axis=1).columns.tolist():
                ls.append(csv_df_i[iter])
            ls = [(ls[cols].tolist() + padding_size * [0])[:padding_size] for cols in range(len(csv_df_i.drop(collist, axis=1).columns.tolist()))]
        else:
            for iter in csv_df_i.drop('DOY', axis=1).columns.tolist():
                ls.append(csv_df_i[iter])
            ls = [(ls[cols].tolist() + padding_size * [0])[:padding_size] for cols in range(len(csv_df_i.drop('DOY', axis=1).columns.tolist()))]
        flat_list = [item for sublist in ls for item in sublist]
        # flat_list = (flat_list + padding_size * [0])[:padding_size]
        flat_df = pd.DataFrame(flat_list)
        # csv_ts_i = convert_df_to_sktime(df=csv_df_i, index=csv_index)
        csv_ts_list.append(flat_df)
    len(csv_ts_list)

    # csv_ts = pd.concat(csv_ts_list)
    # csv_ts
    # csv_ts_i
    test_x_rf = csv_ts_list

    print('predicting...')
    # pred_y = classifier.predict(test_x_rf)
    probs_y = classifier.predict_proba(test_x_rf)
    pred_y = [1 if (probs_y[x, 1] >= .5) else 0 for x in range(0, probs_y.shape[0])]
    accuracy = accuracy_score(test_y, pred_y)
    print("Accuracy:", accuracy)
    cm = confusion_matrix(test_y, pred_y)  # set labels argument for better visualization?
    print(cm)
    cr = classification_report(test_y, pred_y)
    print(cr)

    ### concatenate test_x_rf, test_y, pred_y and probs_y
    test = pd.DataFrame(copy.deepcopy(test_x))
    test['labels'] = test_y
    test['preds_rf'] = pred_y
    # test['prob_rf_class_0'] = probs_y[:, 0]
    test['probs_dist_rf'] = probs_y[:, 1]

    ### save to disk
    test.to_csv(os.path.join(input_data_path + 'model/' +
                             'rf_test_results_' + model_name + '.csv'), sep=';', index=False)

    ### save predictions and labels to disk
    # path = os.path.join(input_data_path + 'model/', 'rf_preds' + model_name)


    ### possible visualization?
    # https://www.datacamp.com/tutorial/random-forests-classifier-python
    # for i in range(3):
    #     tree = rf.estimators_[i]
    #     dot_data = export_graphviz(tree,
    #                                feature_names=X_train.columns,
    #                                filled=True,
    #                                max_depth=2,
    #                                impurity=False,
    #                                proportion=True)
    #     graph = graphviz.Source(dot_data)
    #     display(graph)

    # #########################
    # ##### TimeSeriesSVC #####
    # #########################
    #
    # print('preparing support vector machine data')
    # csv_ts_list = []
    # # for csv_index, csv_path_i in enumerate(train_x):
    # #     csv_df_i = pd.read_csv(os.path.join(input_data_path, csv_path_i + '.csv'), index_col=False, usecols=collist, parse_dates=False)
    # #     csv_ts_i = convert_df_to_sktime(df=csv_df_i, index=csv_index)
    # #     csv_ts_list.append(csv_ts_i)
    # for csv_index, csv_path_i in enumerate(train_x):
    #     csv_df_i = pd.read_csv(os.path.join(input_data_path, csv_path_i + '.csv'), index_col=False, usecols=collist, parse_dates=False)
    #     # csv_ts_i = convert_df_to_sktime(df=csv_df_i, index=csv_index)
    #     csv_ts_list.append(csv_df_i)
    # len(csv_ts_list)
    # # csv_ts = pd.concat(csv_ts_list)
    # # csv_ts
    # # csv_ts_i
    # train_x_svc = csv_ts_list
    # # csv_ts.loc[0, 'BLU_mean']
    #
    # sktime.datatypes.check_is_mtype(train_x_svc, 'df-list', return_metadata=True)
    #
    # print('fitting TimeSeriesSVC...')
    # start = time.time()
    # # BaggingClassifier for parallelization has absolutely no effect -> drop it!
    # mean_gaussian_tskernel = AggrDist(RBF())
    # classifier = BaggingClassifier(TimeSeriesSVC(kernel=mean_gaussian_tskernel,
    #                                              class_weight='balanced',
    #                                              verbose=True,
    #                                              random_state=1,
    #                                              max_iter=500),
    #                                n_samples=len(train_x_svc) / n_jobs,
    #                                n_estimators=n_jobs,
    #                                bootstrap=True,
    #                                bootstrap_features=True,
    #                                random_state=1)  # , n_jobs=70
    # # classifier = TimeSeriesSVC(class_weight='balanced', verbose=True, random_state=1)
    # classifier.fit(train_x_svc, train_y)
    # end = time.time()
    # print('fitting TimeSeriesSVC finished')
    # print(start - end)
    # print('seconds for model fitting')
    #
    # csv_ts_list = []
    # # for csv_index, csv_path_i in enumerate(train_x):
    # #     csv_df_i = pd.read_csv(os.path.join(input_data_path, csv_path_i + '.csv'), index_col=False, usecols=collist, parse_dates=False)
    # #     csv_ts_i = convert_df_to_sktime(df=csv_df_i, index=csv_index)
    # #     csv_ts_list.append(csv_ts_i)
    # for csv_index, csv_path_i in enumerate(test_x):
    #     csv_df_i = pd.read_csv(os.path.join(input_data_path, csv_path_i + '.csv'), index_col=False, usecols=collist, parse_dates=False)
    #     # csv_ts_i = convert_df_to_sktime(df=csv_df_i, index=csv_index)
    #     csv_ts_list.append(csv_df_i)
    # len(csv_ts_list)
    # # csv_ts = pd.concat(csv_ts_list)
    # # csv_ts
    # # csv_ts_i
    # test_x_svc = csv_ts_list
    #
    # print('predicting...')
    # # pred_y = classifier.predict(test_x_svc)
    # probs_y = classifier.predict_proba(test_x_svc)
    # pred_y = [1 if (probs_y[x, 1] >= .5) else 0 for x in range(0, probs_y.shape[0])]
    # accuracy = accuracy_score(test_y, pred_y)
    # print("Accuracy:", accuracy)
    # cm = confusion_matrix(test_y, pred_y)  # set labels argument for better visualization?
    # print(cm)
    # cr = classification_report(test_y, pred_y)
    # print(cr)
    #
    # ### concatenate test_x_rf, test_y, pred_y and probs_y
    # test = pd.DataFrame(copy.deepcopy(test_x))
    # test['labels'] = test_y
    # test['preds_svm'] = pred_y
    # # test['prob_rf_class_0'] = probs_y[:, 0]
    # # test['probs_dist_svm'] = probs_y[:, 1]
    #
    # ### save to disk
    # test.to_csv(os.path.join(input_data_path + 'model/' +
    #                          'svm_test_results_' + model_name + '.csv'), sep=';', index=False)
    #
    # ### save predictions and labels to disk
    # # path = os.path.join(input_data_path + 'model/', 'rf_preds' + model_name)
    # ### save the random forest model itself to disk
    # filename = os.path.join(input_data_path, 'model', 'svm_model_' + model_name + '.pkl')
    # pickle.dump(classifier, open(filename, "wb"))
    #
    #
    #
    # ######################################
    # ##### TimeSeriesForestClassifier #####
    # ##### with balanced classes ##########
    # ######################################
    #
    # ### balance classes by sampling amount of minority class from majority class
    # traindat = traindat.groupby('max_mort_int').apply(
    #     lambda x: x.sample(min(traindat.groupby('max_mort_int').size()))).reset_index(drop=True)
    # train_x = traindat['plotID']
    # train_y = traindat['max_mort_int']
    #
    # print('preparing random forest with balanced data')
    # csv_ts_list = []
    # # for csv_index, csv_path_i in enumerate(train_x):
    # #     csv_df_i = pd.read_csv(os.path.join(input_data_path, csv_path_i + '.csv'), index_col=False, usecols=collist, parse_dates=False)
    # #     csv_ts_i = convert_df_to_sktime(df=csv_df_i, index=csv_index)
    # #     csv_ts_list.append(csv_ts_i)
    # for csv_index, csv_path_i in enumerate(train_x):
    #     csv_df_i = pd.read_csv(os.path.join(input_data_path, csv_path_i + '.csv'), index_col=False, usecols=collist,
    #                            parse_dates=False)
    #     ls = []
    #     for iter in collist:
    #         ls.append(csv_df_i[iter])
    #     flat_list = [item for sublist in ls for item in sublist]
    #     flat_df = pd.DataFrame(flat_list)
    #     # csv_ts_i = convert_df_to_sktime(df=csv_df_i, index=csv_index)
    #     csv_ts_list.append(flat_df)
    # len(csv_ts_list)
    # # csv_ts = pd.concat(csv_ts_list)
    # # csv_ts
    # # csv_ts_i
    # train_x_rf = csv_ts_list
    # sktime.datatypes.check_is_mtype(train_x_rf, 'df-list', return_metadata=True)
    #
    # print('fitting TimeSeriesForestClassifier...')
    # print(time.time())
    # classifier = (  # ColumnConcatenator() *
    #         PaddingTransformer() *
    #         TimeSeriesForestClassifier(min_interval=10, n_estimators=100, n_jobs=n_jobs, random_state=1))
    # # classifier = TimeSeriesForestClassifier(n_jobs=60)
    # classifier.fit(train_x_rf, train_y)
    # print(time.time())
    # print('fitting TimeSeriesForestClassifier finished')
    #
    # csv_ts_list = []
    # # for csv_index, csv_path_i in enumerate(train_x):
    # #     csv_df_i = pd.read_csv(os.path.join(input_data_path, csv_path_i + '.csv'), index_col=False, usecols=collist, parse_dates=False)
    # #     csv_ts_i = convert_df_to_sktime(df=csv_df_i, index=csv_index)
    # #     csv_ts_list.append(csv_ts_i)
    # for csv_index, csv_path_i in enumerate(test_x):
    #     csv_df_i = pd.read_csv(os.path.join(input_data_path, csv_path_i + '.csv'), index_col=False, usecols=collist,
    #                            parse_dates=False)
    #     ls = []
    #     for iter in collist:
    #         ls.append(csv_df_i[iter])
    #     flat_list = [item for sublist in ls for item in sublist]
    #     flat_df = pd.DataFrame(flat_list)
    #     # csv_ts_i = convert_df_to_sktime(df=csv_df_i, index=csv_index)
    #     csv_ts_list.append(flat_df)
    # len(csv_ts_list)
    # # csv_ts = pd.concat(csv_ts_list)
    # # csv_ts
    # # csv_ts_i
    # test_x_rf = csv_ts_list
    #
    # print('predicting...')
    # # pred_y = classifier.predict(test_x_rf)
    # probs_y = classifier.predict_proba(test_x_rf)
    # pred_y = [1 if (probs_y[x, 1] >= .5) else 0 for x in range(0, probs_y.shape[0])]
    # accuracy = accuracy_score(test_y, pred_y)
    # print("Accuracy:", accuracy)
    #
    # ### concatenate test_x_rf, test_y, pred_y and probs_y
    # test = pd.DataFrame(copy.deepcopy(test_x))
    # test['labels'] = test_y
    # test['preds_rf'] = pred_y
    # # test['prob_rf_class_0'] = probs_y[:, 0]
    # test['probs_dist_rf'] = probs_y[:, 1]
    #
    # ### save to disk
    # test.to_csv(os.path.join(input_data_path + 'model/' +
    #                          'rf_test_results_' + model_name + '_balanced.csv'), sep=';', index=False)
    #
    # ### save predictions and labels to disk
    # # path = os.path.join(input_data_path + 'model/', 'rf_preds' + model_name)
    # ### save the random forest model itself to disk
    # filename = os.path.join(input_data_path, 'model', 'rf_model_' + model_name + 'balanced_.pkl')
    # pickle.dump(classifier, open(filename, "wb"))


def preprocess_datasets(input_data_path, classifier, threshold, train, datasets, remove_threshold):
    # set parameters
    # input_data_path = '/home/cangaroo/christopher/future_forest/forest_decline/data/4_dl_ready_time_series_doy/'
    # TEST_SPLIT = .1
    VAL_SPLIT = .2
    # LR = .0001

    # if classifier:
    #     INPUT_LABELS_SHAPE = (5,)
    #     NUM_CLASSES = 2  # set to len(something) for automatization later on
    # else:
    #     INPUT_LABELS_SHAPE = (9,)
    #     NUM_CLASSES = 9  # set to len(something) for automatization later on

    if not os.path.exists(os.path.join(input_data_path + 'split_data')):
        os.mkdir(os.path.join(input_data_path + 'split_data'))

    ### load metadata csv file
    # meta = pd.read_csv(os.path.join(input_data_path + 'meta/metadata.csv'), sep=';',
    #                    usecols=['plotID', 'date', 'mort_0', 'mort_1', 'mort_2', 'mort_3', 'mort_4',
    #                             'mort_5', 'mort_6', 'mort_7', 'mort_8', 'mort_9', 'dataset'])  # .drop(['Unnamed: 0'], axis=1)
    ### we use all columns,
    ### mainly because the metadata for test dataset will be accessible more easily then
    meta = pd.read_csv(os.path.join(input_data_path + 'meta/metadata.csv'), sep=';')

    ### change sax2 -> sax dataset
    meta.loc[meta['dataset'] == 'sax2', 'dataset'] = 'sax'


    ### datasets to use
    meta.dataset = meta.dataset.astype(str)
    meta = meta[meta['dataset'].isin(
        datasets
    )]

    # ### discard samples without frac_coniferous information
    # meta = meta[meta['frac_coniferous'].notna()]


    ### combine the values of mort_1 and mort_5, since mort_5 is ill-defined,
    ### so that mort_5 effectively gets kicked out for the time
    ### and added to harvest/logging class
    ### here, we even combine ALL the values of mortality (only for pre-training?)
    meta.loc[:, 'mort_1'] = meta['mort_1'] + meta['mort_2'] + meta['mort_3'] + meta['mort_4'] \
                            + meta['mort_5'] + meta['mort_6'] + meta['mort_7'] + meta['mort_8'] \
                            + meta['mort_9']
    # meta.loc[:, ['mort_2', 'mort_3', 'mort_4', 'mort_5', 'mort_6', 'mort_7', 'mort_8', 'mort_9']] = 0


    ### prepare class label
    # meta['max_mort'] = meta[
    #     ['mort_0', 'mort_1', 'mort_2', 'mort_3', 'mort_4', 'mort_5', 'mort_6', 'mort_7', 'mort_8', 'mort_9']
    # ].idxmax(axis=1)
    # ### prepare class label: mort_1 are all mortality reasons/severities combined
    # meta['max_mort'] = meta[
    #     ['mort_0', 'mort_1']
    # ].idxmax(axis=1)
    ### use a threshold for training, e.g. 10%
    # meta['max_mort'] = 'mort_1' if (meta['mort_1'] > 0.2) else 'mort_0'
    meta['max_mort'] = \
        ['mort_1' if (meta['mort_1'].iloc[x] > threshold) else 'mort_0' for x in range(0, len(meta))]

    # ######## IMPORTANT NOTE:
    # ######## since mort_5 means 'unknown mortality', it might make sense not to
    # ######## use this class, but rather incorporate it into mort_1
    # ######## (mort_1 is 'harvest, logging' and also somewhat ill-defined
    # ######## since reason for harvest might be other disturbance (e.g. bark bettle -> salvation logging?)
    meta.loc[(meta.max_mort == 'mort_5'), 'max_mort'] = 'mort_1'

    ### get number of classes automatically
    num_classes = len(meta.groupby('max_mort').size())
    # ### the following (commented out) code is likely obsolete and can be removed on github update
    # ### a binary classification problem takes num_classes = 1 as input,
    # ### because it only has to predict a continuous scale between 0 and 1
    if num_classes == 2:
        num_classes = 1


    ### remove duplicates (if any)
    meta.drop_duplicates(subset='plotID', keep="first", inplace=True)

    ### we need an indication if the pixel is covered merely by deciduous or coniferous trees
    meta['dec_con'] = \
        ['con' if (meta['frac_coniferous'].iloc[x] > .5) else 'dec' for x in range(0, len(meta))]

    ### first dataset in datasets list will be chosen as spatial hold-out
    testdat = meta.loc[(meta['dataset'].isin([datasets[0]]))]
    # testdat = meta.loc[(meta['dataset'].isin(["sax", "lux", "nrw", "rlp", "thu", "bb"]))]

    ### for testdat, we include mort_cleared and mort_soil to mortality
    ### to avoid false false positives in evaluation
    testdat['mort_soil'] = testdat['mort_soil'].fillna(0)
    ### drop soil pixels
    testdat = testdat.loc[~((testdat['mort_soil'].fillna(0) + testdat['mort_regrowth'].fillna(0)) > 0)]
    ### has been assigned using mort_1-9 already
    testdat['mort_1'] = testdat['mort_dec'] + testdat['mort_con'] + testdat['mort_cleared'] # + testdat['mort_soil']
    # clip to 0-1 range
    testdat['mort_1'] = testdat['mort_1'].clip(0,1)
    # re-compute mort_0 and max_mort
    testdat['mort_0'] = 1 - testdat['mort_1']
    # re-compute max_mort (we could even set another threshold for testing!?)
    testdat['max_mort'] =  ['mort_1' if (testdat['mort_1'].iloc[x] > 0) else 'mort_0' for x in range(0, len(testdat))]


    ### drop the spatial hold-out from training and validation dataset
    meta = meta.loc[~(meta['dataset'].isin([datasets[0]]))]
    # meta = meta.loc[~(meta['dataset'].isin(["sax", "lux", "nrw", "rlp", "thu", "bb"]))]

    if train:
        ##### prepare schiefer and schwarz datasets
        schiefer_schwarz = meta.loc[(meta['dataset'].isin(["schiefer", "schwarz"]))]
        schiefer_schwarz['mort_1'] = schiefer_schwarz['mort_1'].clip(0, 1)
        schiefer_schwarz['mort_0'] = 1 - schiefer_schwarz['mort_1']
        schiefer_schwarz = schiefer_schwarz.loc[~((schiefer_schwarz['mort_0'] > threshold) & (schiefer_schwarz['mort_0'] < 1))]
        # rest = meta.loc[(meta['dataset'].isin(['rlp', 'nrw', 'bb', 'sax2', 'thu', 'sax_2', 'lux', 'sax']))]

        ##### prepare fnews
        fnews = meta.loc[(meta['dataset'].isin(["fnews"]))]
        # rest = rest.loc[~(((rest['mort_con'] + rest['mort_dec']) < (rest['mort_cleared'] + rest['mort_soil']))) & (rest['mort_1'] > 0)]
        # if threshold > .5 this also implies (mort_con + mort_dec) > (mort_cleared + mort_soil)
        # rest = rest.loc[~((rest['mort_cleared'] + rest['mort_soil'].fillna(0)) > 0)]
        fnews = fnews.loc[~((fnews['mort_soil'].fillna(0)) > 0)]
        ### edge pixels of healthy polygons excluded
        fnews = fnews.loc[((fnews["healthy"] == 0) | (fnews["healthy"] == 1))]
        ### mort_1 has been defined above already
        # rest['mort_1'] = rest['mort_dec'] + rest['mort_con'] + rest['mort_cleared']
        ### "repair" mort_0 (there are some issues with mort_0 == 1 and simultaneously mort_cleared == 1
        fnews['mort_1'] = fnews['mort_1'].clip(0, 1)
        fnews['mort_0'] = 1 - fnews['mort_1']
        # rest = rest.loc[~((rest['mort_soil']) > 0)]

        ##### prepare 5 AOI's that have not been chosen as hold-out
        rest = meta.loc[~(meta['dataset'].isin(["fnews", "schiefer", "schwarz"]))]
        ### note that datasets[0] has been excluded beforehand (see above)
        rest = rest.loc[~((rest['mort_soil'].fillna(0) + rest['mort_regrowth'].fillna(0)) > 0)]
        rest['mort_1'] = rest['mort_dec'] + rest['mort_con'] + rest['mort_cleared']
        ### "repair" mort_0 (there are some issues with mort_0 == 1 and simultaneously mort_cleared == 1
        rest['mort_1'] = rest['mort_1'].clip(0, 1)
        rest['mort_0'] = 1 - rest['mort_1']

        ### combine the datasets for training
        meta = pd.concat([schiefer_schwarz, fnews, rest])

    # ### avoid bias caused by disturbance = coniferous trees, nondisturbance = broadleaved trees
    # ### idea: for each dataset (!!), get (roughly) equal number of coniferous and deciduous samples
    # ### do this only during training, not during testing
    # ### this code snippet can easily be improved!!!
    # if train:
    #     meta['dec_con'] = \
    #         ['con' if (meta['frac_coniferous'].iloc[x] > .5) else 'dec' for x in range(0, len(meta))]
    #     # meta['dec_con'].value_counts()
    #     # meta = meta.groupby('dec_con').apply(
    #     #     lambda x: x.sample(min(meta.groupby('dec_con').size()))).reset_index(drop=True)
    #     # meta.groupby(["dataset", "dec_con"]).size()
    #     # meta["dataset"].unique()
    #     # meta = [meta[meta["dataset"] == dataset].groupby('dec_con').apply(
    #     #     lambda x: x.sample(min(meta.groupby('dec_con').size()))).reset_index(drop=True)
    #     #         for dataset in meta["dataset"].unique()]
    #     ##### this is "cdeq"!!! #####
    #     meta_1 = meta[meta["max_mort"] == meta["max_mort"].unique()[0]]
    #     meta_1 = meta_1.groupby('dec_con').apply(
    #         lambda x: x.sample(min(meta_1.groupby('dec_con').size()))).reset_index(drop=True)
    #     for iter in range(1, len(meta["max_mort"].unique())):
    #         meta_2 = meta[meta["max_mort"] == meta["max_mort"].unique()[iter]]
    #         meta_2 = meta_2.groupby('dec_con').apply(
    #             lambda x: x.sample(min(meta_2.groupby('dec_con').size()))).reset_index(drop=True)
    #         meta_1 = pd.concat([meta_1, meta_2])
    #     meta_1.groupby(["max_mort", "dec_con"]).size()
    #     meta = meta_1

    ### remove disturbed samples below specific threshold to avoid confusing the model with fuzzy labels
    ### this is done only for training and validation datasets (see above)
    if remove_threshold:
        meta = meta.drop(meta[(meta.mort_1 < threshold) & (meta.mort_1 > 0)].index)
        print(meta.groupby('dataset').size())

    meta['max_mort'] = \
        ['mort_1' if (meta['mort_1'].iloc[x] > threshold) else 'mort_0' for x in range(0, len(meta))]

    ### this is the new OVERSAMPLING (therefore also "cdeq") and split code (avoid bias caused by "spruce always dieback" problem
    ### avoid bias caused by disturbance = coniferous trees, nondisturbance = broadleaved trees
    ### idea: for each dataset (!!), get (roughly) equal number of coniferous and deciduous samples
    ### do this only during training, not during testing
    ### this code snippet can easily be improved!!!
    if train:
        # # Identify the minority combination to oversample
        # minority_combination = 'mort_1 dec'

        ### since the amount of mort_1 dec is very small, we double it and then perform undersampling
        ### for mort_0: just perform undersampling of mort_0 con
        ### rest is done by class weights

        # Determine the number of samples needed to match the majority combination
        majority_combination_mort_1 = meta[meta['max_mort'] == "mort_1"].groupby('dec_con').size().idxmax()
        minority_combination_mort_1_cd = meta[meta['max_mort'] == "mort_1"].groupby('dec_con').size().idxmin()
        # minority_combination_mort_1 = meta[meta['max_mort'] == "mort_1"].groupby('dec_con').size().min()
        target_samples = meta[(meta['max_mort'] == "mort_1") & (meta['dec_con'] == minority_combination_mort_1_cd)].shape[0]

        # Oversample the minority combination
        minority_samples = meta[(meta['max_mort'] == "mort_1") & (meta['dec_con'] == minority_combination_mort_1_cd)]

        undersampled_majority_samples_mort_1 = meta[(meta["max_mort"].isin(["mort_1"]) & meta["dec_con"].isin([majority_combination_mort_1]))]
        minority_combination_mort_1 = target_samples * 2 # has been doubled
        undersampled_majority_samples_mort_1 = resample(undersampled_majority_samples_mort_1, replace=True,
                                        n_samples=minority_combination_mort_1,
                                        random_state=1)
        meta = meta[~(meta["max_mort"].isin(["mort_1"]) & meta["dec_con"].isin(["con"]))]

        ### undersampling mort_0 con
        undersampled_majority_samples_mort_0 = meta[(meta["max_mort"].isin(["mort_0"]) & meta["dec_con"].isin(["con"]))]
        minority_combination_mort_0 = meta[meta['max_mort'] == "mort_0"].groupby('dec_con').size().min()
        meta = meta[~(meta["max_mort"].isin(["mort_0"]) & meta["dec_con"].isin(["con"]))]
        undersampled_majority_samples_mort_0 = resample(undersampled_majority_samples_mort_0, replace=False,
                                                n_samples=minority_combination_mort_0,
                                                random_state=1)

        # # Concatenate the oversampled minority samples with the original dataframe
        # meta = pd.concat([meta, oversampled_minority_samples])

        # ##### also oversample the undisturbed class !!!!!
        # # Determine the number of samples needed to match the majority combination
        # majority_combination_mort_0 = meta[meta['max_mort'] == "mort_0"].groupby('dec_con').size().idxmax()
        # minority_combination_mort_0_cd = meta[meta['max_mort'] == "mort_0"].groupby('dec_con').size().idxmin()
        # minority_combination_mort_0 = meta[meta['max_mort'] == "mort_0"].groupby('dec_con').size().min()
        # target_samples = meta[(meta['max_mort'] == "mort_0") & (meta['dec_con'] == majority_combination_mort_0)].shape[0]
        #
        # # Oversample the minority combination
        # minority_samples = meta[(meta['max_mort'] == "mort_0") & (meta['dec_con'] == minority_combination_mort_0_cd)]
        # oversampled_minority_samples_mort_0 = resample(minority_samples, replace=True,
        #                                         n_samples=target_samples - minority_combination_mort_0,
        #                                         random_state=1)


        # Concatenate the oversampled minority samples with the original dataframe
        # meta = pd.concat([meta, oversampled_minority_samples_mort_0, oversampled_minority_samples_mort_1])
        meta = pd.concat([meta, undersampled_majority_samples_mort_0, minority_samples, undersampled_majority_samples_mort_1])



    traindat, valdat = train_test_split(meta,
                                            random_state=1,
                                            shuffle=True,
                                            test_size=VAL_SPLIT,
                                            stratify=meta['max_mort'])

    # traindat, valdat = train_test_split(trainvaldat, random_state=1, shuffle=True,
    #                                     test_size=VAL_SPLIT, stratify=trainvaldat['max_mort'])

    # if train:
    #     ##### this is "undersampling"!!! #####
    #     ### take a sample for each group so that mort classes are balanced
    #     traindat = traindat.groupby('max_mort').apply(
    #         lambda x: x.sample(min(traindat.groupby('max_mort').size()))).reset_index(drop=True)
    #     # traindat.groupby('max_mort').size()
    #     valdat = valdat.groupby('max_mort').apply(lambda x: x.sample(min(valdat.groupby('max_mort').size()))).reset_index(
    #         drop=True)
    #     # valdat.groupby('max_mort').size()
    #     # testdat = testdat.groupby('max_mort').apply(
    #     #     lambda x: x.sample(min(testdat.groupby('max_mort').size()))).reset_index(drop=True)
    #     # testdat.groupby('max_mort').size()

    print("\nTest Set Distribution:\n", testdat.groupby(['max_mort', 'dec_con']).size())
    print("\nTraining Set Distribution:\n", traindat.groupby(['max_mort', 'dec_con']).size())
    print("\nValidation Set Distribution:\n", valdat.groupby(['max_mort', 'dec_con']).size())

    # meta[meta['plotID'] == 'nrw_11169_2021_06_14']

    ### get integer value for mortality class
    labels_int_train = traindat[['plotID', 'max_mort']]
    label_array = le.fit_transform(labels_int_train['max_mort'])
    # labels_int_train.drop("max_mort", axis=1, inplace=True)
    max_mort = pd.Series(label_array)
    labels_int_train.reset_index(drop=True, inplace=True)
    labels_int_train = labels_int_train.assign(max_mort=max_mort)
    traindat.reset_index(drop=True, inplace=True)
    traindat['max_mort_int'] = labels_int_train['max_mort']

    ### get integer value for mortality class
    labels_int_val = valdat[['plotID', 'max_mort']]
    label_array = le.fit_transform(labels_int_val['max_mort'])
    # labels_int_val.drop("max_mort", axis=1, inplace=True)
    max_mort = pd.Series(label_array)
    labels_int_val.reset_index(drop=True, inplace=True)
    labels_int_val = labels_int_val.assign(max_mort=max_mort)
    valdat.reset_index(drop=True, inplace=True)
    valdat['max_mort_int'] = labels_int_val['max_mort']

    ### get integer value for mortality class
    labels_int_test = testdat[['plotID', 'max_mort']]
    label_array = le.fit_transform(labels_int_test['max_mort'])
    # labels_int_test.drop("max_mort", axis=1, inplace=True)
    max_mort = pd.Series(label_array)
    labels_int_test.reset_index(drop=True, inplace=True)
    labels_int_test = labels_int_test.assign(max_mort=max_mort)
    testdat.reset_index(drop=True, inplace=True)
    testdat['max_mort_int'] = labels_int_test['max_mort']

    ### explanation class weights in BCEWithLogitsLoss:
    ### https://discuss.pytorch.org/t/bcewithlogitsloss-and-class-weights/88837/3
    ### BCEWithLogitsLoss takes only a weight for class 1, not for class 0
    ### so we divide N(class_0) / N(class_1)
    class_weights = [(traindat.groupby('max_mort').size()[0] / traindat.groupby('max_mort').size()[1])]

    ### save training, validation and test data to disk
    traindat.to_csv(os.path.join(input_data_path + 'split_data/' +
                             'train_labels_' + datasets[0] + '.csv'), sep=';', index=False)
    valdat.to_csv(os.path.join(input_data_path + 'split_data/' +
                             'validation_labels_' + datasets[0] + '.csv'), sep=';', index=False)
    testdat.to_csv(os.path.join(input_data_path + 'split_data/' +
                             'test_labels_' + datasets[0] + '.csv'), sep=';', index=False)

    if not train:
        testdat = pd.concat([traindat, valdat, testdat], ignore_index=True)



    ### return labels for train/val/test split
    return labels_int_train, labels_int_val, labels_int_test, num_classes, testdat, class_weights


if __name__ == "__main__":
    # config = Config()

    # INPUT_DATA_PATH = \
    #     '/home/cangaroo/christopher/future_forest/forest_decline/data/4_dl_ready_time_series_all_data_distatend/'

    # INPUT_DATA_PATH = \
    #      '/mnt/storage2/forest_decline/data/4_dl_ger_lvl2_incl_schwarz/'

    # INPUT_DATA_PATH = \
    #     '/mnt/storage2/forest_decline/data/4_dl_ger_lvl2_all/'

    # INPUT_DATA_PATH = \
    #      '/mnt/storage2/forest_decline/data/4_dl_ger_lux_foresight_hypothesis/'

    INPUT_DATA_PATH = \
         '/mnt/storage2/forest_decline/data/4_dl_europe_lvl2_all_distatend_finetune_iter6_20240412/'

    ### prepare list of columns/bands to use as input for the model
    bands = ['BLU', 'GRN', 'RED', 'RE1', 'RE2', 'RE3', 'BNR', 'NIR', 'SW1', 'SW2']  # all channels, no indices
    # possible indices are: ['CRE', 'EVI', 'KNV', 'NBR', 'NDV', 'TCB', 'TCW', 'TCD', 'TCG']
    # bands = ['BLU', 'GRN', 'RED', 'RE1', 'RE2', 'RE3', 'BNR', 'NIR', 'SW1', 'SW2',
    #          'CRE', 'EVI', 'KNV', 'NBR', 'NDV', 'TCB', 'TCW', 'TCD', 'TCG']
    # add statistics with a preceeding underscore symbol, e.g. '_mean'
    # all possible statistics would be: ['_mean', '_min', '_max', '_count', '_q10', '_q25', '_q50', '_q75', '_q90']
    vals = ['_mean']
    COL_LIST = [[os.path.join(band + val) for band in bands] for val in vals]
    COL_LIST = [item for sublist in COL_LIST for item in sublist]
    COL_LIST.append('DOY')

    # DATASETS = ['landshut']
    ### first dataset is chosen as spatial hold-out!!!
    DATASETS = ['lux', 'thu', 'rlp', 'bb', 'nrw', 'sax', 'fnews', 'schiefer', 'schwarz']
    # DATASETS = ['lux', 'rlp', 'nrw', 'sax', 'schiefer', 'schwarz']
    DATASETS.insert(0, DATASETS.pop(DATASETS.index(args.target_aoi)))
    # DATASETS.insert(0, DATASETS.pop(DATASETS.index('lux')))
    print('datasets used (first one is test hold-out): ')
    print(DATASETS)
    # DATASETS = ['lux', 'nrw', 'bb', 'rlp', 'sax2', 'thu', 'schiefer', 'schwarz']
    # DATASETS = ['senf2', 'undstrbd', 'undisturbed', 'effis', 'forwind', 'thonfeld']
    # DATASETS = ['undstrbd', 'undisturbed', 'effis', 'forwind', 'thonfeld']
    # DATASETS = ['senf2', 'undstrbd', 'undisturbed', 'effis',
    #             'forwind', 'thonfeld', 'schwarz', 'schiefer']
    # pre-training: ['thonfeld', 'senf2', 'undstrbd', 'effis', 'caudullo']
    # fine-tuning:
    # DATASETS = ['schwarz', 'icp', 'schiefer', 'fva', 'icplvl2']
    # inference on specific dataset: ['schwarz']
    BATCH_SIZE = 126 # batch size for other models: 128
    # NUM_FEATURES = 10 # number of channels and/or indices -> derived from COL_LIST
    HIDDEN_SIZE = 128 # default 256; also called (and equals) embedding dim in the code
    HIDDEN_CLFR_HEAD = None # takes a list of number of hidden sizes, or None
    CNN_EMBEDDING = False
    N_LAYERS = 3 # default 3
    ATTN_HEADS = 8 # default 8; note that HIDDEN_SIZE % ATTN_HEADS must equal 0
    DROPOUT = 0.3 # default .1
    NUM_EPOCHS = 100
    MAX_LEN = 256
    # INDICES = 'False'
    INDICES = args.indices
    print('indices: ')
    print(INDICES)
    if INDICES.lower() == 'only':
        ONLY_INDICES = True
    else:
        ONLY_INDICES = False
    ### automated naming of the model according to parameters?
    if ONLY_INDICES:
        MODEL_NAME = os.path.join('mort50_ger_lvl2_distatend_h128_ah8_cdeq_cwdiv3_noundersamp_nosac_vi_only_pretrain_wocuttsatend_traininclfnews_cdeq_nocw_osus_inclcleared_iter5_lr15_' + DATASETS[0])
    else:
        MODEL_NAME = os.path.join('mort50_ger_lvl2_distatend_h128_ah8_cdeq_cwdiv3_noundersamp_nosac_vi_' + str(INDICES).lower() + '_pretrain_wocuttsatend_traininclfnews_cdeq_nocw_osus_inclcleared_iter5_lr15_' + DATASETS[0])
        # MODEL_NAME = "mort50_ger_lvl2_distatend_h128_ah8_cdeq_cwdiv3_noundersamp_nosac_vi_false_pretrain_wocuttsatend_trainonlyfnews_cdeq_nocw_osus_inclcleared_lr15_iter5"
        # mort50_ger_lvl2_distatend_h128_ah8_cdeq_cwdiv3_noundersamp_nosac_vi_false_pretrain_wocuttsatend_traininclfnews_cdeq_nocw_osus_inclcleared_iter5_lr15_lux
    MODEL_SAVE_PATH = os.path.join(INPUT_DATA_PATH + 'model/')
    TRAIN = True # include training, or skip training and jump to inference straight away?
    CLASSIFIER = True # False means Regression
    DATA_AUG = True # randomly remove observations at beginning of time series (only in training phase)
    # what is the percentage canopy cover (or tree number) loss above which to qualify as disturbed pixel?
    THRESHOLD = 0.50 # must be given as ratio, i.e. [0, 1] range, not as percentage; default: .1
    REMOVE_THRESHOLD = True
    PATIENCE = 5
    TEST = True
    EXPLAIN = True
    NUM_WORKERS = 25 # 40 to max 65 is optimal for high GPU usage
    DEFINE_TESTDAT = True
    ### remember to set learning rate 1e-4 for pretraining and 1e-5 for fine-tuning!!!
    PRETRAINED = True
    PREDICT = False
    BASELINE = False
    # BASELINE = args.train_baseline
    print('baseline: ')
    print(BASELINE)
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if ONLY_INDICES:
        INDICES = 'false' # because then we have 10 features (=INDICES=False), not 20 (INDICES=True)

    if PREDICT:
        TRAIN = False
        TEST = False
        EXPLAIN = False
        PRETRAINED = True
        DATA_AUG = False
        REMOVE_THRESHOLD = False

    if not os.path.exists(MODEL_SAVE_PATH):
        os.mkdir(MODEL_SAVE_PATH)


    if not PREDICT:
        ### prepare and load training, validation and test data
        # train_file = config.file_path + 'Train.csv'
        # valid_file = config.file_path + 'Validate.csv'
        # test_file = config.file_path + 'Test.csv'
        ### obviously, the following code has to be polished (e.g. input variables to function)...
        train_labels, val_labels, test_labels, NUM_CLASSES, testdat, CLASS_WEIGHTS = \
            preprocess_datasets(INPUT_DATA_PATH, CLASSIFIER, THRESHOLD, TRAIN, DATASETS, REMOVE_THRESHOLD)

    # print('class weights:')
    # print(CLASS_WEIGHTS)
    # CRITERION = nn.BCEWithLogitsLoss(pos_weight = torch.Tensor(CLASS_WEIGHTS).to(DEVICE)) # pos_weight = torch.Tensor(CLASS_WEIGHTS).to(DEVICE)
    CRITERION = nn.BCEWithLogitsLoss() # pos_weight = torch.Tensor(CLASS_WEIGHTS).to(DEVICE)

    if INDICES.lower() == 'true':
        FEATURE_NUM = len(COL_LIST) - 1 + 10
    else:
        FEATURE_NUM = len(COL_LIST) - 1

    if DEFINE_TESTDAT:
        testdat = pd.read_csv(os.path.join(INPUT_DATA_PATH + 'split_data/test_labels_' + DATASETS[0] + '.csv'), sep=';')

    if BASELINE:
        train_baseline(INPUT_DATA_PATH, MODEL_NAME, COL_LIST, DATASETS, NUM_WORKERS, INDICES, ONLY_INDICES)

    ### adjust how to input the datasets to FineTuneDataset (Dataset class)
    print("Loading Data sets...")
    if PREDICT:
        predict_dataset = PredictDataset(root_dir=INPUT_DATA_PATH, feature_num=len(COL_LIST)-1,
                                         seq_len=MAX_LEN, collist=COL_LIST,
                                         classifier=CLASSIFIER, data_aug=DATA_AUG)
        preddat = glob.glob(os.path.join(INPUT_DATA_PATH + '*.{}'.format('csv')))
        preddat = [os.path.basename(x) for x in preddat]
        preddat = [os.path.splitext(x)[0] for x in preddat]
        preddat = pd.DataFrame(preddat)
        preddat.rename(columns={ preddat.columns[0]: "plotID" }, inplace = True)
        print("prediction samples: %d" %
              (len(predict_dataset)))
    else:
        train_dataset = PretrainDataset(labels_ohe=train_labels, root_dir=INPUT_DATA_PATH,
                                        feature_num=FEATURE_NUM, seq_len=MAX_LEN, collist=COL_LIST,
                                        classifier=CLASSIFIER, data_aug=DATA_AUG, indices=INDICES, only_indices=ONLY_INDICES)
        valid_dataset = PretrainDataset(labels_ohe=val_labels, root_dir=INPUT_DATA_PATH,
                                        feature_num=FEATURE_NUM, seq_len=MAX_LEN, collist=COL_LIST,
                                        classifier=CLASSIFIER, data_aug=DATA_AUG, indices=INDICES, only_indices=ONLY_INDICES)
        test_dataset = PretrainDataset(labels_ohe=test_labels, root_dir=INPUT_DATA_PATH,
                                       feature_num=FEATURE_NUM, seq_len=MAX_LEN, collist=COL_LIST,
                                       classifier=CLASSIFIER, data_aug=False, indices=INDICES, only_indices=ONLY_INDICES)
        print("training samples: %d, validation samples: %d, testing samples: %d" %
              (len(train_dataset), len(valid_dataset), len(test_dataset)))
    # train_dataset = FinetuneDataset(train_labels, config.num_features, config.max_length)
    # valid_dataset = FinetuneDataset(val_labels, config.num_features, config.max_length)
    # test_dataset = FinetuneDataset(test_labels, config.num_features, config.max_length)

    if (not TRAIN) and (not DEFINE_TESTDAT) and TEST:
        test_dataset = ConcatDataset([train_dataset, valid_dataset, test_dataset])
        print("no training desired! test samples: %d" %(len(test_dataset)))

    # ### short sanity check
    # for i in range(0, 10): # len(train_dataset)
    #     sample = train_dataset[i]
    #     print(i,
    #           sample['bert_mask'].shape,
    #           sample['class_label'].shape,
    #           sample['time'].shape)

    print("Creating Dataloader...")
    if PREDICT:
        predict_data_loader = DataLoader(predict_dataset, shuffle=False, num_workers=NUM_WORKERS,
                                       batch_size=BATCH_SIZE, drop_last=False)
    else:
        train_data_loader = DataLoader(train_dataset, shuffle=True, num_workers=NUM_WORKERS,
                                       batch_size=BATCH_SIZE, drop_last=False)
        valid_data_loader = DataLoader(valid_dataset, shuffle=True, num_workers=NUM_WORKERS,
                                       batch_size=BATCH_SIZE, drop_last=False)
        test_data_loader = DataLoader(test_dataset, shuffle=False, num_workers=NUM_WORKERS,
                                      batch_size=BATCH_SIZE, drop_last=False)
    # train_data_loader = DataLoader(train_dataset, shuffle=True,
    #                                batch_size=config.batch_size, drop_last=False)
    # valid_data_loader = DataLoader(valid_dataset, shuffle=False,
    #                                batch_size=config.batch_size, drop_last=False)
    # test_data_loader = DataLoader(test_dataset, shuffle=False,
    #                               batch_size=config.batch_size, drop_last=False)

    print("Initializing SITS-BERT...")
    if (INDICES.lower() == 'true'):  # 10 indices used here
        sbert = SBERT(num_features=len(COL_LIST) - 1 + 10, hidden=HIDDEN_SIZE, n_layers=N_LAYERS,
                      attn_heads=ATTN_HEADS, dropout=DROPOUT, hidden_clfr_head=HIDDEN_CLFR_HEAD,
                      cnn_embedding=CNN_EMBEDDING)
    else:
        sbert = SBERT(num_features=len(COL_LIST) - 1, hidden=HIDDEN_SIZE, n_layers=N_LAYERS,
                      attn_heads=ATTN_HEADS, dropout=DROPOUT, hidden_clfr_head=HIDDEN_CLFR_HEAD,
                      cnn_embedding=CNN_EMBEDDING)

    if TRAIN or TEST:
        # sbert = SBERT(config.num_features, hidden=config.hidden_size, n_layers=config.layers,
        #               attn_heads=config.attn_heads, dropout=config.dropout)
        print("Creating Downstream Task Trainer...")
        trainer = SBERTFineTuner(sbert, NUM_CLASSES, seq_len=MAX_LEN,
                                 criterion=CRITERION, classifier=CLASSIFIER,
                                 train_dataloader=train_data_loader,
                                 valid_dataloader=valid_data_loader)
    # ### add this again if we are using a pretrained model
    # if config.pretrain_path is not None:
    #     print("Loading pre-trained model parameters...")
    #     sbert_path = config.pretrain_path + "checkpoint.bert.pth"
    #     sbert.load_state_dict(torch.load(sbert_path))

    if TRAIN:
        if PRETRAINED:
            trainer.load(MODEL_SAVE_PATH, MODEL_NAME)
        if CLASSIFIER:
            print("Training SITS-BERT Classifier...")
            early_stopper = EarlyStopper(patience=PATIENCE, min_delta=0)
            OAAccuracy = 0
            OALoss = 1000000
            history = dict(epoch=[], train_OA=[], train_Kappa=[], valid_OA=[], valid_Kappa=[], train_loss=[], valid_loss=[])
            for epoch in range(NUM_EPOCHS):
                train_OA, train_Kappa, valid_OA, valid_Kappa, train_loss, valid_loss = trainer.train(epoch)
                history['epoch'].append(epoch)
                history['train_OA'].append(train_OA)
                history['train_Kappa'].append(train_Kappa)
                history['valid_OA'].append(valid_OA)
                history['valid_Kappa'].append(valid_Kappa)
                history['train_loss'].append(train_loss)
                history['valid_loss'].append(valid_loss)

                ### plot intermediate loss and accuracy curves
                epochs_range = range(len(history['train_loss']))
                plt.figure(figsize=(12, 12))
                plt.subplot(2, 2, 1)
                plt.plot(epochs_range, history['train_OA'], label='Training Overall Accuracy')
                plt.plot(epochs_range, history['valid_OA'], label='Validation Overall Accuracy')
                plt.legend(loc='upper left')
                plt.title('Training and Validation Overall Accuracy')
                plt.ylim([0, 100])
                plt.subplot(2, 2, 2)
                plt.plot(epochs_range, history['train_loss'], label='Training Loss')
                plt.plot(epochs_range, history['valid_loss'], label='Validation Loss')
                plt.legend(loc='upper right')
                plt.title('Training and Validation Loss')
                plt.ylim([0, 1])
                plt.show()

                if OAAccuracy < valid_OA:
                    OAAccuracy = valid_OA
                    trainer.save(epoch, MODEL_SAVE_PATH, MODEL_NAME)
                ### implement early stopping
                if early_stopper.early_stop(valid_loss):
                    print('Training stopped due to early stopping criterion!')
                    break

            ### plot final results
            epochs_range = range(len(history['train_loss']))
            plt.figure(figsize=(12, 12))
            plt.subplot(2, 2, 1)
            plt.plot(epochs_range, history['train_OA'], label='Training Overall Accuracy')
            plt.plot(epochs_range, history['valid_OA'], label='Validation Overall Accuracy')
            plt.legend(loc='upper left')
            plt.title('Training and Validation Overall Accuracy')

            plt.subplot(2, 2, 2)
            plt.plot(epochs_range, history['train_loss'], label='Training Loss')
            plt.plot(epochs_range, history['valid_loss'], label='Validation Loss')
            plt.legend(loc='upper right')
            plt.title('Training and Validation Loss')
            # plt.show()
            plt.savefig(os.path.join(MODEL_SAVE_PATH, MODEL_NAME + '_train_val_acc_loss.pdf'),
                        bbox_inches='tight')

        else:
            print("Training SITS-BERT Regressor...")
            OALoss = 10000000
            history = dict(epoch=[], train_loss=[], valid_loss=[])
            for epoch in range(NUM_EPOCHS):
                train_loss, valid_loss = trainer.train(epoch)
                history['epoch'].append(epoch)
                history['train_loss'].append(train_loss)
                history['valid_loss'].append(valid_loss)
                if OALoss > valid_loss:
                    OALoss = valid_loss
                    trainer.save(epoch, MODEL_SAVE_PATH)

            ### plot results
            epochs_range = range(len(history['train_loss']))
            plt.figure(figsize=(15, 15))
            plt.plot(epochs_range, history['train_loss'], label='Training Loss')
            plt.plot(epochs_range, history['valid_loss'], label='Validation Loss')
            plt.legend(loc='upper right')
            plt.title('Training and Validation Loss')
            plt.savefig(os.path.join(MODEL_SAVE_PATH, MODEL_NAME + '_train_val_acc_loss_regression.pdf'),
                        bbox_inches='tight')

    if TEST:
        print("\n\n\n")
        print("Testing SITS-BERT...")
        trainer.load(MODEL_SAVE_PATH, MODEL_NAME)
        if CLASSIFIER:
            OA, Kappa, AA, _, test_result = trainer.test(test_data_loader)
            print('test_OA = %.2f, test_kappa = %.3f, test_AA (average acc) = %.3f' % (OA, Kappa, AA))

            ### put together test data including plotID, labels and predictions
            testdat['test_label'] = test_result['label']  # or pd.Series(test_result['label']).values
            testdat['prediction'] = pd.DataFrame(test_result['prediction']).astype(int)
            testdat['raw_output'] = pd.DataFrame(test_result['raw_output'])

            ### check if that worked (very important: shuffle=False in DataLoader)
            print('sanity check for label assigment: no errors?')
            print(testdat['max_mort_int'].equals(testdat['test_label'])) # if True, assigning labels worked fine

            ### get more information about classification results
            cm = confusion_matrix(testdat['test_label'], testdat['prediction'])
            print(cm)

            cr = classification_report(testdat['test_label'], testdat['prediction'])
            print(cr)

            # set normalize='true' if desired
            disp = ConfusionMatrixDisplay(confusion_matrix=cm)
            disp.plot()
            plt.savefig(os.path.join(MODEL_SAVE_PATH, MODEL_NAME + '_confusion_matrix.pdf'),
                        bbox_inches='tight')

            ### save test data and labels
            testdat.to_csv(os.path.join(MODEL_SAVE_PATH,
                                        'test_results_' + MODEL_NAME + '.csv'), sep=';', index=False)

        else: # if regression
            test_loss, test_result = trainer.test(test_data_loader)
            print('test_loss = %.2f' % test_loss)

            labels = pd.DataFrame(test_result['label'])
            labels.columns = ['label_mort_0', 'label_mort_1', 'label_mort_2', 'label_mort_3', 'label_mort_4',
                              'label_mort_5', 'label_mort_6', 'label_mort_7', 'label_mort_8', 'label_mort_9']
            preds= pd.DataFrame(test_result['prediction'])
            preds.columns = ['pred_mort_0', 'pred_mort_1', 'pred_mort_2', 'pred_mort_3', 'pred_mort_4',
                             'pred_mort_5', 'pred_mort_6', 'pred_mort_7', 'pred_mort_8', 'pred_mort_9']

            ### concat to test dataframe
            testdat = pd.concat([testdat.reset_index(drop=True), labels.reset_index(drop=True)], axis = 1)
            testdat = pd.concat([testdat.reset_index(drop=True), preds.reset_index(drop=True)], axis=1)

            ### drop label columns again (we can use the old column names from testdat)
            testdat = testdat.drop(
                ['label_mort_0', 'label_mort_1', 'label_mort_2', 'label_mort_3', 'label_mort_4',
                 'label_mort_5', 'label_mort_6', 'label_mort_7', 'label_mort_8', 'label_mort_9']
                , axis=1)

            ### scatterplots for mortality labels vs. predictions
            plt.figure(figsize=(15, 15))
            # plt.subplot(2, 2, 2)
            plt.scatter(testdat['pred_mort_0'], testdat['mort_0']) 
            plt.xlabel("Predictions")
            plt.ylabel("Labels")
            plt.title('Non-Disturbance: Predictions vs. Targets')
            # fit linear regression via least squares with numpy.polyfit
            # It returns an slope (b) and intercept (a)
            # deg=1 means linear fit (i.e. polynomial of degree 1)
            b, a = np.polyfit(testdat['pred_mort_0'], testdat['mort_0'], deg=1)

            # Create sequence of 100 numbers from 0 to 1
            xseq = np.linspace(0, 1, num=100)

            # Plot regression line
            plt.plot(xseq, a + b * xseq, color="k", lw=2.5)
            # plt.show()
            plt.savefig(os.path.join(MODEL_SAVE_PATH, MODEL_NAME + '_scatterplot_mort_0.pdf'),
                        bbox_inches='tight')

            ### mort_1
            plt.figure(figsize=(15, 15))
            plt.scatter(testdat['pred_mort_1'], testdat['mort_1'])  # , label='Non-Disturbance Predictions vs. Targets'
            plt.xlabel("Predictions")
            plt.ylabel("Labels")
            plt.title('Disturbance: Predictions vs. Targets')
            # Fit linear regression via least squares with numpy.polyfit
            # It returns an slope (b) and intercept (a)
            # deg=1 means linear fit (i.e. polynomial of degree 1)
            b, a = np.polyfit(testdat['pred_mort_1'], testdat['mort_1'], deg=1)

            # Create sequence of 100 numbers from 0 to 1
            xseq = np.linspace(0, 1, num=100)

            # Plot regression line
            plt.plot(xseq, a + b * xseq, color="k", lw=2.5)
            # plt.show()
            plt.savefig(os.path.join(MODEL_SAVE_PATH, MODEL_NAME + '_scatterplot_mort_1.pdf'),
                        bbox_inches='tight')
            
    ### only tested for binary classification, not for multiclass or regression
    if EXPLAIN:
        print('Explaining...')

        ### load test data incl. predictions and labels
        testdat = pd.read_csv(os.path.join(MODEL_SAVE_PATH,
                                    'test_results_' + MODEL_NAME + '.csv'), sep=';')
        testdat['prediction'] = testdat['prediction'].astype(int)

        ### assign device (cuda)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        ### initialize and load
        model = SBERTClassification(sbert, NUM_CLASSES, MAX_LEN)
        checkpoint = torch.load(os.path.join(MODEL_SAVE_PATH, MODEL_NAME + '.tar'))
        model.load_state_dict(checkpoint['model_state_dict'])

        with ((torch.enable_grad())):
            model.eval()
            model.to(device)
            model.zero_grad()

            # now loop over test data (each sample one by one)
            for iter in range(len(test_dataset)):

                data = test_dataset[iter]

                ##### feature importance (feature ablation) #####


                ### define feature mask to get feature importance for satellite bands
                # https://pytorch.org/docs/stable/tensors.html
                ### create empty array
                feature_mask = np.ones(shape=[MAX_LEN, FEATURE_NUM])
                ### now loop through columns/sentinel-2 bands to assign group values
                for npiter in range(feature_mask.shape[1]):
                    feature_mask[:, npiter] = feature_mask[:, npiter] * npiter
                
                ### convert to pytorch tensor
                feature_mask = torch.tensor(feature_mask).long()

                ### initialize Feature Ablation algorithm
                ablator = FeatureAblation(model)

                ### compute attributions for sentinel-2 bands/indices
                ### .unsqueeze(axis=0) is necessary for all tensors
                ### to replace batch_size
                attribution = ablator.attribute(
                    inputs=data["bert_input"].float().unsqueeze(axis=0).to(device),
                    baselines=None,
                    target=None,
                    additional_forward_args=(
                        data["bert_mask"].long().unsqueeze(axis=0).to(device),
                        data["time"].long().unsqueeze(axis=0).to(device)),
                    feature_mask=feature_mask.unsqueeze(axis=0).to(device),
                    perturbations_per_eval=NUM_WORKERS,
                    show_progress=False
                )

                attribution = attribution.squeeze()
                attribution = pd.DataFrame(attribution.detach().cpu().numpy())


                ### column names:
                df_cols = [os.path.join(band) for band in bands]
                if INDICES: # INDICES=True
                    df_cols = df_cols + ['CRSWIR', 'NBR', 'TCW', 'TCD', 'NDVI', 'NDWI', 'NDMI', 'LAI', 'MSI', 'NDRE']
                    attribution.columns = df_cols
                else:
                    attribution.columns = df_cols

                ### only first row is relevant, all other rows are duplicates
                attribution = attribution.head(1)

                if not os.path.exists(os.path.join(INPUT_DATA_PATH, 'model', 'attr_' + MODEL_NAME, 'feature_ablation')):
                    os.mkdir(os.path.join(INPUT_DATA_PATH, 'model', 'attr_' + MODEL_NAME, 'feature_ablation'))

                ### save dataframe to disk
                attribution.to_csv(os.path.join(MODEL_SAVE_PATH, 'attr_' + MODEL_NAME, 'feature_ablation',
                                        str(testdat["plotID"].iloc[iter]) + '_attr_label_' +
                                        str(testdat["test_label"].iloc[iter]) + '_pred_' +
                                        str(testdat["prediction"].iloc[iter]) +
                                        str(np.where((testdat["mort_1"].iloc[iter] > 0) & (testdat["prediction"].iloc[iter] == 1)
                                                    or (testdat["mort_1"].iloc[iter] == 0) & (testdat["prediction"].iloc[iter] == 0),
                                                    '_correct', '_false')) +
                                        '_extent_' + str(int(testdat["mort_1"].iloc[iter] * 100)) +
                                        '_featabl.csv'),
                            sep=';', index=False)


                ################################
                ##### time step importance #####
                ##### (Integrated Gradients) ###
                ################################

                ### ignore UserWarnings (it is known and clear to me
                ### the original embedding layer must be set back after
                ### model interpretation is finished
                warnings.filterwarnings("ignore", category=UserWarning)

                ### configure interpretable embedding layer
                interpretable_emb = configure_interpretable_embedding_layer(model,'sbert.embedding')

                ### do NOT pass the two inputs input_sequence and doy_sequence as tuple,
                ### because the FORWARD function in BERTEmbedding takes two separate inputs
                ### it has nothing to do with the constructor (lesson learned)
                input_emb = interpretable_emb.indices_to_embeddings(
                    data["bert_input"].float().unsqueeze(axis=0).to(device),
                    data["time"].long().unsqueeze(axis=0).to(device))

                ### initialize IntegratedGradients
                ig = IntegratedGradients(model, multiply_by_inputs=False)

                ### get attribution
                attribution = ig.attribute(input_emb.to(device),
                                            additional_forward_args=
                                            (data["bert_mask"].long().unsqueeze(axis=0).to(device),
                                            data["time"].long().unsqueeze(axis=0).to(device)))

                # Remove batch dimension for plotting
                attribution = torch.squeeze(attribution, dim=0)
                attribution_abs = torch.sum(torch.abs(attribution), dim=1, keepdim=True)
                attribution_abs = attribution_abs / torch.norm(attribution_abs)  
                attribution = torch.sum(attribution, dim=1, keepdim=True)
                attribution = attribution / torch.norm(attribution) 

                ### get dataframe for saving to disk
                df = pd.DataFrame(attribution.detach().cpu().numpy())
                df.columns = ['attr_sum']
                df_abs = pd.DataFrame(attribution_abs.detach().cpu().numpy())
                df_abs.columns = ['attr_sum_abs_norm']

                ### duplicate 10 times for plotting each channel
                attribution = attribution.repeat(1, 10)

                ### convert to numpy
                attribution = attribution.detach().cpu().numpy()

                ### get input sequence
                x = data["bert_input"].float().detach().cpu().numpy()

                ### get positions for plotting
                dates = data["time"].long().detach().cpu().numpy()

                ### add dates and band values to dataframe
                df['doy'] = dates
                df2 = pd.DataFrame(x)
                ### column names:
                df_cols = [os.path.join(band) for band in bands]
                if INDICES: 
                    df_cols = df_cols + ['CRSWIR', 'NBR', 'TCW', 'TCD', 'NDVI', 'NDWI', 'NDMI', 'LAI', 'MSI', 'NDRE']
                    df2.columns = df_cols
                else:
                    df2.columns = df_cols
                df = pd.concat([df.reset_index(drop=True), df_abs.reset_index(drop=True), df2.reset_index(drop=True)], axis=1)

                ### cut all data by dates[dates == 0]
                arraymin = np.amin(np.nonzero(dates))
                ### slice all arrays
                attribution = attribution[arraymin:, :]
                x = x[arraymin:, :]
                dates = dates[arraymin:]

                ### finish the dataframe
                df = df.iloc[arraymin:, :]

                ### prepare directory
                if not os.path.exists(os.path.join(INPUT_DATA_PATH, 'model', 'attr_' + MODEL_NAME, 'integrated_gradients')):
                    os.mkdir(os.path.join(INPUT_DATA_PATH, 'model', 'attr_' + MODEL_NAME, 'integrated_gradients'))

                ### save all relevant data to disk
                df.to_csv(os.path.join(MODEL_SAVE_PATH, 'attr_' + MODEL_NAME, 'integrated_gradients',
                                        str(testdat["plotID"].iloc[iter]) + '_attr_label_' +
                                        str(testdat["test_label"].iloc[iter]) + '_pred_' +
                                        str(testdat["prediction"].iloc[iter]) +
                                        str(np.where((testdat["mort_1"].iloc[iter] > 0) & (testdat["prediction"].iloc[iter] == 1)
                                                    or (testdat["mort_1"].iloc[iter] == 0) & (testdat["prediction"].iloc[iter] == 0),
                                                    '_correct', '_false')) +
                                        '_extent_' + str(int(testdat["mort_1"].iloc[iter] * 100)) +
                                        '_intgrad.csv'),
                            sep=';', index=False)

                ### after finishing the interpretation, we need to remove
                ### interpretable embedding layer with the following command
                remove_interpretable_embedding_layer(model, interpretable_emb)