import pandas as pd
import os
from sklearn.model_selection import train_test_split
from sklearn.utils import resample
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()


def preprocess_datasets_finetuning(input_data_path, threshold, train, datasets, remove_threshold):
    VAL_SPLIT = .2
    if not os.path.exists(os.path.join(input_data_path + 'split_data')):
        os.mkdir(os.path.join(input_data_path + 'split_data'))

    ### load metadata csv file
    meta = pd.read_csv(os.path.join(input_data_path + 'meta/metadata.csv'), sep=';')

    ### datasets to use
    meta.dataset = meta.dataset.astype(str)
    meta = meta[meta['dataset'].isin(
        datasets
    )]

    ### combine all mortality values for binary classification
    meta.loc[:, 'mort_1'] = meta['mort_1'] + meta['mort_2'] + meta['mort_3'] + meta['mort_4'] \
                            + meta['mort_5'] + meta['mort_6'] + meta['mort_7'] + meta['mort_8'] \
                            + meta['mort_9']


    ### prepare class label
    meta['max_mort'] = \
        ['mort_1' if (meta['mort_1'].iloc[x] > threshold) else 'mort_0' for x in range(0, len(meta))]

    ### get number of classes
    num_classes = len(meta.groupby('max_mort').size())
    ### a binary classification problem takes num_classes = 1 as input,
    ### because it only has to predict a continuous scale between 0 and 1
    if num_classes == 2:
        num_classes = 1

    ### we need an indication if the pixel is covered merely by deciduous or coniferous trees
    meta['dec_con'] = \
        ['con' if (meta['frac_coniferous'].iloc[x] > .5) else 'dec' for x in range(0, len(meta))]

    ### first dataset in datasets list will be chosen as spatial hold-out
    testdat = meta.loc[(meta['dataset'].isin([datasets[0]]))]

    ### prepare testdat
    testdat['mort_soil'] = testdat['mort_soil'].fillna(0)
    ### drop soil pixels (unclear if disturbance happened, or if it is just a natural canopy opening or undetected non-forest)
    testdat = testdat.loc[~(testdat['mort_soil'].fillna(0) > 0)]
    ### mort_1 is mort_dec + mort_con + mort_cleared
    testdat['mort_1'] = testdat['mort_dec'] + testdat['mort_con'] + testdat['mort_cleared']
    # clip to 0-1 range
    testdat['mort_1'] = testdat['mort_1'].clip(0,1)
    # re-compute mort_0 and max_mort
    testdat['mort_0'] = 1 - testdat['mort_1']
    # re-compute max_mort
    testdat['max_mort'] =  ['mort_1' if (testdat['mort_1'].iloc[x] > 0) else 'mort_0' for x in range(0, len(testdat))]

    ### drop the spatial hold-out from training and validation dataset
    meta = meta.loc[~(meta['dataset'].isin([datasets[0]]))]

    if train:
        ##### prepare schiefer and schwarz datasets
        schiefer_schwarz = meta.loc[(meta['dataset'].isin(["schiefer", "schwarz"]))]
        schiefer_schwarz['mort_1'] = schiefer_schwarz['mort_1'].clip(0, 1)
        schiefer_schwarz['mort_0'] = 1 - schiefer_schwarz['mort_1']
        schiefer_schwarz = schiefer_schwarz.loc[~((schiefer_schwarz['mort_0'] > threshold) & (schiefer_schwarz['mort_0'] < 1))]

        ##### prepare fnews
        fnews = meta.loc[(meta['dataset'].isin(["fnews"]))]
        fnews = fnews.loc[~((fnews['mort_soil'].fillna(0)) > 0)]
        ### edge pixels of healthy polygons excluded
        fnews = fnews.loc[((fnews["healthy"] == 0) | (fnews["healthy"] == 1))]
        ### mort_1 has been defined above already
        fnews['mort_1'] = fnews['mort_1'].clip(0, 1)

        ##### prepare 5 AOI's that have not been chosen as hold-out
        rest = meta.loc[~(meta['dataset'].isin(["fnews", "schiefer", "schwarz"]))]
        ### note that datasets[0] has been excluded beforehand (see above)
        rest = rest.loc[~(rest['mort_soil'].fillna(0) > 0)]
        rest['mort_1'] = rest['mort_dec'] + rest['mort_con'] + rest['mort_cleared']
        rest['mort_1'] = rest['mort_1'].clip(0, 1)

        ### combine the datasets for training
        meta = pd.concat([schiefer_schwarz, fnews, rest])

    ### remove disturbed samples below specific threshold to avoid confusing the model with fuzzy labels
    ### this is done only for training and validation datasets (see above)
    if remove_threshold:
        meta = meta.drop(meta[(meta.mort_1 < threshold) & (meta.mort_1 > 0)].index)
        print(meta.groupby('dataset').size())

    ### assign mortality class
    meta['max_mort'] = \
        ['mort_1' if (meta['mort_1'].iloc[x] > threshold) else 'mort_0' for x in range(0, len(meta))]

    ### oversampling of minority class
    ### avoid bias caused by disturbance = coniferous trees, nondisturbance = broadleaved trees
    if train:
        ### since the amount of mort_1 dec is very small, we double it and then conduct undersampling
        ### for mort_0: just conduct undersampling of mort_0 con
        ### rest is done by class weights

        # determine the number of samples needed to match the majority combination
        majority_combination_mort_1 = meta[meta['max_mort'] == "mort_1"].groupby('dec_con').size().idxmax()
        minority_combination_mort_1_cd = meta[meta['max_mort'] == "mort_1"].groupby('dec_con').size().idxmin()
        target_samples = meta[(meta['max_mort'] == "mort_1") & (meta['dec_con'] == minority_combination_mort_1_cd)].shape[0]

        ### oversampling the minority combination
        minority_samples = meta[(meta['max_mort'] == "mort_1") & (meta['dec_con'] == minority_combination_mort_1_cd)]

        ### undersampling mort_1 con
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

        ### concatenate the dataframes to obtain a more balanced dataset
        meta = pd.concat([meta, undersampled_majority_samples_mort_0, minority_samples, undersampled_majority_samples_mort_1])

    ### Split the dataset into training and validation sets
    traindat, valdat = train_test_split(meta,
                                            random_state=1,
                                            shuffle=True,
                                            test_size=VAL_SPLIT,
                                            stratify=meta['max_mort'])

    print("\nTest Set Distribution:\n", testdat.groupby(['max_mort', 'dec_con']).size())
    print("\nTraining Set Distribution:\n", traindat.groupby(['max_mort', 'dec_con']).size())
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

    ### get class weights
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