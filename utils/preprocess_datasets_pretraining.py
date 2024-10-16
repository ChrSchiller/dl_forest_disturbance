import pandas as pd
import os
from sklearn.model_selection import train_test_split
from sklearn.utils import resample
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()

def preprocess_datasets(input_data_path, threshold, train, datasets, remove_threshold):

    TEST_SPLIT = .1
    VAL_SPLIT = .2

    if not os.path.exists(os.path.join(input_data_path + 'pretrain_data')):
        os.mkdir(os.path.join(input_data_path + 'pretrain_data'))

    ### load metadata csv file
    meta = pd.read_csv(os.path.join(input_data_path + 'meta/metadata.csv'), sep=';')

    ### use only thonfeld, undisturbed and senf2 for the time
    meta.dataset = meta.dataset.astype(str)
    meta = meta[meta['dataset'].isin(
        datasets
    )]

    ### combine all mortality values for binary classification
    meta.loc[:, 'mort_1'] = meta['mort_1'] + meta['mort_2'] + meta['mort_3'] + meta['mort_4'] \
                            + meta['mort_5'] + meta['mort_6'] + meta['mort_7'] + meta['mort_8'] \
                            + meta['mort_9']
    meta.loc[:, ['mort_2', 'mort_3', 'mort_4', 'mort_5', 'mort_6', 'mort_7', 'mort_8', 'mort_9']] = 0

    ### remove disturbed samples below specific threshold to avoid confusing the model with fuzzy labels
    if remove_threshold:
        meta = meta.drop(meta[(meta.mort_1 < threshold) & (meta.mort_1 > 0)].index)

    ### prepare class label
    meta['max_mort'] = \
        ['mort_1' if (meta['mort_1'].iloc[x] > threshold) else 'mort_0' for x in range(0, len(meta))]

    ### get number of classes
    num_classes = len(meta.groupby('max_mort').size())
    ### a binary classification problem takes num_classes = 1 as input,
    ### because it only has to predict a continuous scale between 0 and 1
    if num_classes == 2:
        num_classes = 1

    ### avoid bias caused by disturbance = coniferous trees, nondisturbance = broadleaved trees
    ### idea: get (roughly) equal number of coniferous and deciduous samples
    ### do this only during training, not during testing
    if train:
        meta['dec_con'] = \
            ['con' if (meta['frac_coniferous'].iloc[x] > .5) else 'dec' for x in range(0, len(meta))]

        # Split the dataset into train, validation, and test sets
        meta, testdat = train_test_split(meta[['plotID', 'mort_0', 'mort_1', 'mort_2', 'mort_3', 'mort_4',
                                                      'mort_5', 'mort_6', 'mort_7', 'mort_8', 'mort_9', 'max_mort',
                                                      'dec_con']],
                                                test_size=TEST_SPLIT,
                                                stratify=meta[['max_mort', 'dec_con']], random_state=42)

        # Determine the number of samples needed to match the majority combination
        majority_combination_mort_1 = meta[meta['max_mort'] == "mort_1"].groupby('dec_con').size().idxmax()
        minority_combination_mort_1 = meta[meta['max_mort'] == "mort_1"].groupby('dec_con').size().min()
        target_samples = meta[(meta['max_mort'] == "mort_1") & (meta['dec_con'] == majority_combination_mort_1)].shape[0]

        # Oversample the minority combination
        minority_samples = meta[(meta['max_mort'] == "mort_1") & (meta['dec_con'] == 'dec')]
        oversampled_minority_samples = resample(minority_samples, replace=True, n_samples=target_samples - minority_combination_mort_1,
                                                random_state=1)

        # Concatenate the oversampled minority samples with the original dataframe
        meta = pd.concat([meta, oversampled_minority_samples])

        print("\nTest Set Distribution:\n", testdat.groupby(['max_mort', 'dec_con']).size())

    # Split dataset into validation and training sets
    traindat, valdat = train_test_split(meta[['plotID', 'mort_0', 'mort_1', 'mort_2', 'mort_3', 'mort_4',
                                                      'mort_5', 'mort_6', 'mort_7', 'mort_8', 'mort_9', 'max_mort',
                                                      'dec_con']],
                                        test_size=VAL_SPLIT, stratify=meta[['max_mort', 'dec_con']],
                                       random_state=42)

    # Print the distribution of each set
    print("Training Set Distribution:\n", traindat.groupby(['max_mort', 'dec_con']).size())
    print("\nValidation Set Distribution:\n", valdat.groupby(['max_mort', 'dec_con']).size())

    ### get integer value for mortality class
    labels_int_train = traindat[['plotID', 'max_mort']]
    label_array = le.fit_transform(labels_int_train['max_mort'])
    max_mort = pd.Series(label_array)
    labels_int_train.reset_index(drop=True, inplace=True)
    labels_int_train = labels_int_train.assign(max_mort=max_mort)
    traindat.reset_index(drop=True, inplace=True)
    traindat['max_mort_int'] = labels_int_train['max_mort']

    ### get integer value for mortality class
    labels_int_val = valdat[['plotID', 'max_mort']]
    label_array = le.fit_transform(labels_int_val['max_mort'])
    max_mort = pd.Series(label_array)
    labels_int_val.reset_index(drop=True, inplace=True)
    labels_int_val = labels_int_val.assign(max_mort=max_mort)
    valdat.reset_index(drop=True, inplace=True)
    valdat['max_mort_int'] = labels_int_val['max_mort']

    ### get integer value for mortality class
    labels_int_test = testdat[['plotID', 'max_mort']]
    label_array = le.fit_transform(labels_int_test['max_mort'])
    max_mort = pd.Series(label_array)
    labels_int_test.reset_index(drop=True, inplace=True)
    labels_int_test = labels_int_test.assign(max_mort=max_mort)
    testdat.reset_index(drop=True, inplace=True)
    testdat['max_mort_int'] = labels_int_test['max_mort']

    ### get class weights for training with imbalanced classes
    class_weights = [(traindat.groupby('max_mort').size()[0] / traindat.groupby('max_mort').size()[1]) / 3]

    ### save training, validation and test data to disk
    traindat.to_csv(os.path.join(input_data_path + 'pretrain_data/' +
                             'train_labels.csv'), sep=';', index=False)
    valdat.to_csv(os.path.join(input_data_path + 'pretrain_data/' +
                             'validation_labels.csv'), sep=';', index=False)
    testdat.to_csv(os.path.join(input_data_path + 'pretrain_data/' +
                             'test_labels.csv'), sep=';', index=False)

    ### if no training desired, we concatenate all datasets for testing
    if not train:
        testdat = pd.concat([traindat, valdat, testdat], ignore_index=True)

    ### return labels for train/val/test split
    return labels_int_train, labels_int_val, labels_int_test, num_classes, testdat, class_weights