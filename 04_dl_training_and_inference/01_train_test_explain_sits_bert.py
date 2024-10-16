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
from model.bert import SBERT
from trainer.trainer import SBERTTrainer
from dataset.pretrain_dataset import PretrainDataset
from dataset.finetune_dataset import FinetuneDataset
from model.classification_model import SBERTClassification

import numpy as np
import argparse
import random
import os
import matplotlib.pyplot as plt
from matplotlib import cm, pyplot as plt
import pandas as pd
import warnings
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, classification_report
from utils.early_stopper import EarlyStopper
from utils.preprocess_datasets_pretraining import preprocess_datasets_pretraining
from utils.preprocess_datasets_finetuning import preprocess_datasets_finetuning

### set seed
def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True

setup_seed(123)

# create the parser
parser = argparse.ArgumentParser()

### experimental setup and processing
parser.add_argument('--train', type=bool, default=True)
parser.add_argument('--test', type=bool, default=True)
parser.add_argument('--explain', type=bool, default=True)
parser.add_argument('--target_aoi', type=str, required=True, default='lux')
parser.add_argument('--indices', type=str, required=True)
parser.add_argument('--num_workers', type=int, default=65)
parser.add_argument('--pretrained', type=bool, default=False)
parser.add_argument('--indices', type=bool, default=False)
parser.add_argument('--only_indices', type=bool, default=False)

### general info
parser.add_argument('--input_data_path', type=str, required=True, default='/path/to/data')
parser.add_argument('--model_name', type=str, default='model_name')

### hyperparameters
parser.add_argument('--batch_size', type=int, default=128)
parser.add_argument('--hidden_size', type=int, default=128)
parser.add_argument('--hidden_clfr_head', type=str, default=None)
parser.add_argument('--cnn_embedding', type=bool, default=False)
parser.add_argument('--n_layers', type=int, default=3)
parser.add_argument('--attn_heads', type=int, default=8)
parser.add_argument('--dropout', type=float, default=0.3)
parser.add_argument('--num_epochs', type=int, default=100)
parser.add_argument('--max_len', type=int, default=256)
parser.add_argument('--data_aug', type=bool, default=True)
parser.add_argument('--threshold', type=float, default=0.5)
parser.add_argument('--remove_threshold', type=bool, default=True)
parser.add_argument('--patience', type=int, default=10)

### parse the argument
args = parser.parse_args()

if __name__ == "__main__":

    ### prepare list of columns/bands to use as input for the model
    bands = ['BLU', 'GRN', 'RED', 'RE1', 'RE2', 'RE3', 'BNR', 'NIR', 'SW1', 'SW2']  # all channels, no indices
    ### prepare list to use when reading the time series files
    vals = ['_mean']
    COL_LIST = [[os.path.join(band + val) for band in bands] for val in vals]
    COL_LIST = [item for sublist in COL_LIST for item in sublist]
    COL_LIST.append('DOY')

    MODEL_SAVE_PATH = os.path.join(args.input_data_path, 'model')

    ### check if indices and only_indices arguments are contradictive
    assert not (not args.indices and args.only_indices), 'if args.indices is False, args.only_indices must be False as well'

    ### prepare list of datasets to be used in this step
    if args.pretrained:
        DATASETS = ['lux', 'thu', 'rlp', 'bb', 'nrw', 'sax', 'fnews', 'schiefer', 'schwarz']
        DATASETS.insert(0, DATASETS.pop(DATASETS.index(args.target_aoi)))
        print('datasets used (first one is test hold-out): ')
        print(DATASETS)
    else:
        DATASETS = ['senf', 'undisturbed', 'thonfeld', 'forwind']


    ### use cuda if available
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if args.only_indices:
        args.indices = False # because then we have 10 features (=args.indices=False), not 20 (args.indices=True)

    ### create model save path
    if not os.path.exists(MODEL_SAVE_PATH):
        os.mkdir(MODEL_SAVE_PATH)

    if args.explain: # if xAI desired, create subfolder for explanations
        if not os.path.exists(os.path.join(MODEL_SAVE_PATH, 'attr_' + args.model_name)):
            os.mkdir(os.path.join(MODEL_SAVE_PATH, 'attr_' + args.model_name))

    ### prepare and load training, validation and test datasets
    if args.pretrained:
        train_labels, val_labels, test_labels, NUM_CLASSES, testdat, CLASS_WEIGHTS = \
            preprocess_datasets_finetuning(args.input_data_path, args.threshold, args.train, DATASETS, args.remove_threshold)
    else:
        train_labels, val_labels, test_labels, NUM_CLASSES, testdat, CLASS_WEIGHTS = \
            preprocess_datasets_pretraining(args.input_data_path, args.threshold, args.train, DATASETS, args.remove_threshold)

    print('weight for class 1: ')
    print(CLASS_WEIGHTS)
    if args.pretrained: 
        ### in the study, we did not need class weights in finetuning
        CRITERION = nn.BCEWithLogitsLoss()
    else:
        ### apply class weights in pre-training
        CRITERION = nn.BCEWithLogitsLoss(pos_weight = torch.Tensor(CLASS_WEIGHTS).to(DEVICE))

    ### define correct number of input features (considering doy as well)
    if args.indices:
        FEATURE_NUM = len(COL_LIST) - 1 + 10
    else:
        FEATURE_NUM = len(COL_LIST) - 1

    print("Loading Data sets...")
    if args.pretrained: 
        train_dataset = FinetuneDataset(labels_ohe=train_labels, root_dir=args.input_data_path,
                                    feature_num=FEATURE_NUM, seq_len=args.max_len, collist=COL_LIST, 
                                    data_aug=args.data_aug, indices=args.indices, only_indices=args.only_indices)
        valid_dataset = FinetuneDataset(labels_ohe=val_labels, root_dir=args.input_data_path,
                                    feature_num=FEATURE_NUM, seq_len=args.max_len, collist=COL_LIST,
                                    data_aug=args.data_aug, indices=args.indices, only_indices=args.only_indices)
        test_dataset = FinetuneDataset(labels_ohe=test_labels, root_dir=args.input_data_path,
                                    feature_num=FEATURE_NUM, seq_len=args.max_len, collist=COL_LIST,
                                    data_aug=False, indices=args.indices, only_indices=args.only_indices)
    else:
        train_dataset = PretrainDataset(labels_ohe=train_labels, root_dir=args.input_data_path,
                                        feature_num=FEATURE_NUM, seq_len=args.max_len, collist=COL_LIST,
                                        data_aug=args.data_aug, indices=args.indices, only_indices=args.only_indices)
        valid_dataset = PretrainDataset(labels_ohe=val_labels, root_dir=args.input_data_path,
                                        feature_num=FEATURE_NUM, seq_len=args.max_len, collist=COL_LIST,
                                        data_aug=args.data_aug, indices=args.indices, only_indices=args.only_indices)
        test_dataset = PretrainDataset(labels_ohe=test_labels, root_dir=args.input_data_path,
                                        feature_num=FEATURE_NUM, seq_len=args.max_len, collist=COL_LIST,
                                        data_aug=False, indices=args.indices, only_indices=args.only_indices)
    print("training samples: %d, validation samples: %d, testing samples: %d" %
            (len(train_dataset), len(valid_dataset), len(test_dataset)))

    print("Creating Dataloader...")
    train_data_loader = DataLoader(train_dataset, shuffle=True, num_workers=args.num_workers,
                                    batch_size=args.batch_size, drop_last=False)
    valid_data_loader = DataLoader(valid_dataset, shuffle=True, num_workers=args.num_workers,
                                    batch_size=args.batch_size, drop_last=False)
    test_data_loader = DataLoader(test_dataset, shuffle=False, num_workers=args.num_workers,
                                    batch_size=args.batch_size, drop_last=False)

    print("Initializing SITS-BERT...")
    sbert = SBERT(num_features=FEATURE_NUM, hidden=args.hidden_size, n_layers=args.n_layers,
                  attn_heads=args.attn_heads, dropout=args.dropout, hidden_clfr_head=args.hidden_clfr_head,
                  cnn_embedding=args.cnn_embedding)

    if args.train or args.test:
        print("Creating Downstream Task Trainer...")
        trainer = SBERTTrainer(sbert, NUM_CLASSES, seq_len=args.max_len,
                                 criterion=CRITERION,
                                 train_dataloader=train_data_loader,
                                 valid_dataloader=valid_data_loader)

    if args.train:
        if args.pretrained:
            trainer.load(MODEL_SAVE_PATH, args.model_name)
        print("Training SITS-BERT Classifier...")
        early_stopper = EarlyStopper(patience=args.patience, min_delta=0)
        OAAccuracy = 0
        OALoss = 1000000
        history = dict(epoch=[], train_OA=[], train_Kappa=[], valid_OA=[], valid_Kappa=[], train_loss=[], valid_loss=[])
        for epoch in range(args.num_epochs):
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
                trainer.save(epoch, MODEL_SAVE_PATH, args.model_name)
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
        plt.savefig(os.path.join(MODEL_SAVE_PATH, args.model_name + '_train_val_acc_loss.pdf'),
                    bbox_inches='tight')

    if args.test:
        print("\n\n\n")
        print("Testing SITS-BERT...")
        trainer.load(MODEL_SAVE_PATH, args.model_name)

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
        plt.savefig(os.path.join(MODEL_SAVE_PATH, args.model_name + '_confusion_matrix.pdf'),
                    bbox_inches='tight')

        ### save test data and labels
        testdat.to_csv(os.path.join(MODEL_SAVE_PATH,
                                    'test_results_' + args.model_name + '.csv'), sep=';', index=False)

    ### only tested for binary classification
    if args.explain:
        print('Explaining...')

        ### load test data incl. predictions and labels
        testdat = pd.read_csv(os.path.join(MODEL_SAVE_PATH,
                                    'test_results_' + args.model_name + '.csv'), sep=';')
        testdat['prediction'] = testdat['prediction'].astype(int)

        ### assign device (cuda)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        ### initialize and load
        model = SBERTClassification(sbert, NUM_CLASSES, args.max_len)
        checkpoint = torch.load(os.path.join(MODEL_SAVE_PATH, args.model_name + '.tar'))
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
                feature_mask = np.ones(shape=[args.max_len, FEATURE_NUM])
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
                    perturbations_per_eval=args.num_workers,
                    show_progress=False
                )

                attribution = attribution.squeeze()
                attribution = pd.DataFrame(attribution.detach().cpu().numpy())


                ### column names:
                df_cols = [os.path.join(band) for band in bands]
                if args.indices: # args.indices=True
                    df_cols = df_cols + ['CRSWIR', 'NBR', 'TCW', 'TCD', 'NDVI', 'NDWI', 'NDMI', 'LAI', 'MSI', 'NDRE']
                    attribution.columns = df_cols
                else:
                    attribution.columns = df_cols

                ### only first row is relevant, all other rows are duplicates
                attribution = attribution.head(1)

                if not os.path.exists(os.path.join(args.input_data_path, 'model', 'attr_' + args.model_name, 'feature_ablation')):
                    os.mkdir(os.path.join(args.input_data_path, 'model', 'attr_' + args.model_name, 'feature_ablation'))

                ### save dataframe to disk
                attribution.to_csv(os.path.join(MODEL_SAVE_PATH, 'attr_' + args.model_name, 'feature_ablation',
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
                if args.indices: 
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
                if not os.path.exists(os.path.join(args.input_data_path, 'model', 'attr_' + args.model_name, 'integrated_gradients')):
                    os.mkdir(os.path.join(args.input_data_path, 'model', 'attr_' + args.model_name, 'integrated_gradients'))

                ### save all relevant data to disk
                df.to_csv(os.path.join(MODEL_SAVE_PATH, 'attr_' + args.model_name, 'integrated_gradients',
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