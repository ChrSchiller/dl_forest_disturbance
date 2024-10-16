import torch
from torch.utils.data import DataLoader
import torch.nn.functional as F
import torch.nn as nn
from torch.optim import Adam
from captum.attr import (
    IntegratedGradients,
    FeatureAblation,
    configure_interpretable_embedding_layer,
    remove_interpretable_embedding_layer
)
from model import bert as SBERT
from trainer import pretrain as SBERTPreTrainer
from dataset import pretrain_dataset as PretrainDataset

import numpy as np
import random
import os
import matplotlib.pyplot as plt
from matplotlib import cm, pyplot as plt
import pandas as pd
import warnings
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, classification_report
from utils.early_stopper import EarlyStopper
from utils.preprocess_datasets_pretraining import preprocess_datasets_pretraining

### set seed
def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True

setup_seed(123)



if __name__ == "__main__":
    INPUT_DATA_PATH = \
        'path/to/input/data/'

    ### prepare list of columns/bands to use as input for the model
    bands = ['BLU', 'GRN', 'RED', 'RE1', 'RE2', 'RE3', 'BNR', 'NIR', 'SW1', 'SW2']  # all channels, no indices
    ### prepare list to use when reading the time series files
    vals = ['_mean']
    COL_LIST = [[os.path.join(band + val) for band in bands] for val in vals]
    COL_LIST = [item for sublist in COL_LIST for item in sublist]
    COL_LIST.append('DOY')

    ### prepare list of datasets to be used in this step
    DATASETS = ['senf', 'undisturbed', 'thonfeld', 'forwind']

    ## define parameters needed for the training and testing
    BATCH_SIZE = 128
    HIDDEN_SIZE = 128 # == embedding_dim
    HIDDEN_CLFR_HEAD = None # takes a list of number of hidden sizes, or None
    CNN_EMBEDDING = False
    N_LAYERS = 3 
    ATTN_HEADS = 8 # note that HIDDEN_SIZE % ATTN_HEADS must equal 0
    DROPOUT = 0.3
    NUM_EPOCHS = 100
    MAX_LEN = 256
    MODEL_NAME = 'model_name'
    MODEL_SAVE_PATH = os.path.join(INPUT_DATA_PATH + 'model/')
    TRAIN = True # include training, or skip training and jump to inference directly?
    CLASSIFIER = True # False means Regression
    DATA_AUG = True # randomly remove observations at beginning of time series (only in training phase)
    # what is the percentage canopy cover (or tree number) loss above which to qualify as disturbed pixel?
    THRESHOLD = 0.5 # must be given as ratio, i.e. [0, 1] range, not as percentage
    REMOVE_THRESHOLD = True
    PATIENCE = 10 # early stopping
    TEST = True # also test model on test dataset?
    EXPLAIN = True # use explainable AI?
    NUM_WORKERS = 65
    PRETRAINED = False # use pre-trained model?

    INDICES = False # False means only S2 bands, True means S2 bands + indices
    ONLY_INDICES = False # if INDICES = True, ONLY_INDICES means only indices and no bands are used

    ### use cuda if available
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if ONLY_INDICES:
        INDICES = False # because then we have 10 features (=INDICES=False), not 20 (INDICES=True)

    ### create model save path
    if not os.path.exists(MODEL_SAVE_PATH):
        os.mkdir(MODEL_SAVE_PATH)

    if EXPLAIN: # if xAI desired, create subfolder for explanations
        if not os.path.exists(os.path.join(MODEL_SAVE_PATH, 'attr_' + MODEL_NAME)):
            os.mkdir(os.path.join(MODEL_SAVE_PATH, 'attr_' + MODEL_NAME))

    ### prepare and load training, validation and test datasets
    train_labels, val_labels, test_labels, NUM_CLASSES, testdat, CLASS_WEIGHTS = \
        preprocess_datasets_pretraining(INPUT_DATA_PATH, CLASSIFIER, THRESHOLD, TRAIN, DATASETS, REMOVE_THRESHOLD)

    print('weight for class 1: ')
    print(CLASS_WEIGHTS)
    CRITERION = nn.BCEWithLogitsLoss(pos_weight = torch.Tensor(CLASS_WEIGHTS).to(DEVICE))

    ### define correct number of input features (considering doy as well)
    if INDICES:
        FEATURE_NUM = len(COL_LIST) - 1 + 10
    else:
        FEATURE_NUM = len(COL_LIST) - 1

    print("Loading Data sets...")
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

    print("Creating Dataloader...")
    train_data_loader = DataLoader(train_dataset, shuffle=True, num_workers=NUM_WORKERS,
                                    batch_size=BATCH_SIZE, drop_last=False)
    valid_data_loader = DataLoader(valid_dataset, shuffle=True, num_workers=NUM_WORKERS,
                                    batch_size=BATCH_SIZE, drop_last=False)
    test_data_loader = DataLoader(test_dataset, shuffle=False, num_workers=NUM_WORKERS,
                                    batch_size=BATCH_SIZE, drop_last=False)

    print("Initializing SITS-BERT...")
    sbert = SBERT(num_features=FEATURE_NUM, hidden=HIDDEN_SIZE, n_layers=N_LAYERS,
                  attn_heads=ATTN_HEADS, dropout=DROPOUT, hidden_clfr_head=HIDDEN_CLFR_HEAD,
                  cnn_embedding=CNN_EMBEDDING)

    if TRAIN or TEST:
        # sbert = SBERT(config.num_features, hidden=config.hidden_size, n_layers=config.layers,
        #               attn_heads=config.attn_heads, dropout=config.dropout)
        print("Creating Downstream Task Trainer...")
        trainer = SBERTPreTrainer(sbert, NUM_CLASSES, seq_len=MAX_LEN,
                                 criterion=CRITERION, classifier=CLASSIFIER,
                                 train_dataloader=train_data_loader,
                                 valid_dataloader=valid_data_loader)

    
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