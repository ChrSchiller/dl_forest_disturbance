### this script takes as input the force time series retrieved from script previous scripts (all in one folder)
### and outputs the ready-to-go time series for the deep learning model
### including the number of observations per time series
### in a combined metafile
### optionally, the time series can be interpolated

### imports
# set seed
import random
random.seed(123)

import pandas as pd
import geopandas as gpd
import numpy as np
import os
import glob
import matplotlib.pyplot as plt
from scipy import signal
import matplotlib.dates as mdates
myFmt = mdates.DateFormatter('%y-%m')
import copy
from multiprocessing import Pool
from pandas.tseries.offsets import DateOffset
pd.options.mode.chained_assignment = None

### main function: preparing the datasets
### looping over the time series files
def create_interpolate_datasets(main_iterator):
    ### read the dataframe containing the time esries
    df = pd.read_csv(
        os.path.join(WORKDIR + INPUT_DATA_DIR, meta['plotID'].iloc[main_iterator] + '.csv'), sep=',')

    ### drop the plotID column if it exists
    if 'plotID' in df.columns:
        df.drop('plotID', axis=1, inplace=True)

    ### convert date to datetime format
    df['date'] = pd.to_datetime(df['date'], format='%Y%m%d')

    ### sort dataframe by date
    df = df.sort_values(by='date')

    ### get desired end date of the time series
    endDate = pd.to_datetime(meta['date'].iloc[main_iterator], format='%Y-%m-%d')

    ### define start date of the time series as 4 years before the end date
    ### resulting in time series of length 4 years
    startDate = endDate - DateOffset(days=1460)  # 1460 = 4 years

    ### filter the dataframe by the desired time frame
    df = df[
        (df['date'] >= startDate) &
        (df['date'] <= endDate)
        ]

    # get day of the year of startDate
    doy_diff = df['date'].iloc[0].timetuple().tm_yday

    ### convert date to index column
    df = df.set_index('date', drop=False)

    ### prepare list of columns
    vals = ['_mean']
    col_list = [[os.path.join(band + val) for band in bands] for val in vals]
    col_list_flat = [item for sublist in col_list for item in sublist]

    ## add if statement: only continue if file contains all desired bands
    ### also check if there is at least one non-NA value
    if set(col_list_flat).issubset(df.columns) & (not df[col_list_flat[0]].isnull().values.all()):

        ### if resampling is not desired, continue here
        ### (this is the case in the presented study)
        if not RESAMP:
            ### this is to avoid issues later on (mainly to enable relative num_obs)
            df_resamp = df[col_list_flat].resample('1D', origin='end').mean()

            ### assign date of the year (well, across multiple years it is a bit different, but same concept)
            ### adding doy_diff ensures same DOY for same date across input time series
            df_resamp['DOY'] = list(range(doy_diff, len(df_resamp) + doy_diff))

            ### get number of observations in the time series for metafile
            num_obs = (len(df_resamp) - df_resamp['BLU_mean'].isna().sum())

            ### remove empty rows
            df_resamp.dropna(inplace=True)

            # sometimes, there are no observations or some band values missing (for whatever reason)
            # we need at least two values so that std is defined for normalization
            if (len(df_resamp)) > 1 & (df_resamp.isna().sum().sum() == 0):

                ### sometimes first non-NA value is in second year, so we can remove first year's DOY
                if df_resamp['DOY'].iloc[0] > 365:
                    df_resamp['DOY'] = df_resamp['DOY'] - 365

                ### assign last doy value for meta information
                last_doy = df_resamp['DOY'].iloc[-1]

                ### write file to disk
                df_resamp.to_csv(os.path.join(WORKDIR + OUTPUT_DATA_DIR,
                                              meta.iloc[main_iterator, meta.columns.get_loc('plotID')] + '.csv'),
                                 sep=','
                                 )

            else:
                last_doy = np.nan
                num_obs = np.nan
        ### if resampling is desired, continue here
        else:
            # other rules are possible, such as '2W', 'M', ...
            df_resamp = df[col_list_flat].resample('10d', origin='end').mean()

            ### get count of observations per resampled period
            count_obs = df[os.path.join(bands[0] + '_mean')].resample('10d', origin='end').count()

            ### add this information to the dataframe
            df_resamp['count_obs'] = count_obs

            ### check if dataframe contains only NA values
            ### use this later in if condition before writing
            df_empty = len(df_resamp.dropna(inplace=False))

            ### get number of observations for meta file
            num_obs = count_obs.sum()

            # now we can fill potential data gabs with placeholder value
            # (cause DL cannot accept NA values)
            # the idea is that the model still understands that it is an NA value
            # because of the value count column
            # assign 0 to all NA values
            df_resamp = df_resamp.fillna(0)

            ### doy
            # get day of the year of startDate
            doy_diff = df['date'].iloc[0].timetuple().tm_yday

            ### assign date of the year (well, across multiple years it is a bit different, but same concept)
            ### adding doy_diff ensures same DOY for same date across input time series
            df_resamp['DOY'] = np.nan
            doy = df_resamp.index

            ### assign correct doy
            for iter, x in enumerate(doy):
                df_resamp['DOY'][iter] = x.timetuple().tm_yday

            ### now extend doy over multiple years
            for iter in range(1, len(df_resamp)):
                while df_resamp['DOY'][iter] < df_resamp['DOY'][iter-1]:
                    df_resamp['DOY'][iter] = df_resamp['DOY'][iter] + 365 # next year

            # sometimes, there are no observations or some band values missing (for whatever reason)
            # we need at least two values so that std is defined for normalization
            if ((df_empty > 1) & (df_resamp.isna().sum().sum() == 0)):

                ### sometimes first non-NA value is in second year, so we can remove first year's DOY
                if df_resamp['DOY'].iloc[0] > 365:
                    df_resamp['DOY'] = df_resamp['DOY'] - 365

                ### if interpolation is desired, continue here
                if INTERPOLATE:
                    df_resamp_interpol = copy.deepcopy(df_resamp)

                    ### interpolate each band
                    for band in col_list_flat:
                        df_resamp_interpol[band].interpolate(method='linear', axis=0, limit=15, limit_direction='both',
                                                             limit_area=None, inplace=True)

                    ### Savitzky Golay filter on top of interpolation
                    for col in df_resamp_interpol.columns[1:]:
                        # note that count column does not make sense anymore now
                        series = df_resamp_interpol[col]
                        new_series = signal.savgol_filter(series, 6, 3, mode="nearest")
                        df_resamp_interpol[col] = new_series

                    ### save to disk
                    df_resamp_interpol.to_csv(os.path.join(WORKDIR + OUTPUT_DATA_DIR,
                                         meta.iloc[main_iterator, meta.columns.get_loc('plotID')] + '.csv'),
                                              sep=','
                                              )
                ### if interpolation is not desired, continue here
                else:
                    ### save to disk
                    df_resamp.to_csv(os.path.join(WORKDIR + OUTPUT_DATA_DIR,
                                         meta.iloc[main_iterator, meta.columns.get_loc('plotID')] + '.csv'),
                                              sep=','
                                              )

        ### add information about time series 'completeness' to metadata
        return [num_obs, last_doy]


### entry point for the program
if __name__ == '__main__':

    global WORKDIR
    WORKDIR = '/'
    os.chdir(WORKDIR)

    global INPUT_DATA_DIR
    INPUT_DATA_DIR = 'path/to/input/data/folder'

    global OUTPUT_DATA_DIR
    OUTPUT_DATA_DIR = 'path/to/output/data/folder'

    ### filter for datasets needed in DL training
    DATASET_FILTER = ['sax', 'bb', 'rlp', 'nrw', 'thu', 'schwarz', 'schiefer', 'lux', 'fnews',
                      'undisturbed', 'forwind', 'senf2', 'thonfeld']

    ### interpolation desired?
    ### only applicable if resamp = True
    global INTERPOLATE
    INTERPOLATE = False

    ### RESAMP = True means build xy day composites
    global RESAMP
    RESAMP = False

    global COLLIST
    ### subset by list of columns that we think we need
    COLLIST = ["plotID", "date", "mort_0", "mort_1", "mort_2", "mort_3", "mort_4", "mort_5", "mort_6",
               "mort_7", "mort_8", "mort_9", "year", "dataset", "frac_coniferous", "mort_soil",
               "mort_dec", "mort_con", "mort_cleared", "healthy", "region", "NAME_LATN"]

    ### define number of workers (= number of cpu's used)
    parallel_processes = 60

    ### load metadata files
    meta_fls = glob.glob(os.path.join(WORKDIR + INPUT_DATA_DIR, 'meta/*.{}'.format('gpkg')))
    meta_fls = [path for path in meta_fls if any(word in os.path.basename(path) for word in DATASET_FILTER)]

    global meta
    ### read first meta file
    meta = gpd.read_file(meta_fls[0])
    # Filter COLLIST to include only columns that exist in the GeoDataFrame
    existing_columns = [col for col in COLLIST if col in meta.columns]
    meta = meta[existing_columns]
    meta = pd.DataFrame(meta)
    ### change column names if necessary
    if 'NAME_LATN' in meta.columns:
        meta.rename(columns={"NAME_LATN": "region"}, inplace = True)

    ### concatenate the other metadata files
    if len(meta_fls) > 1:
        for fl in meta_fls[1:]:
            print(fl)
            dat = gpd.read_file(fl)
            # Filter COLLIST to include only columns that exist in the GeoDataFrame
            existing_columns = [col for col in COLLIST if col in dat.columns]
            dat = dat[existing_columns]
            dat = pd.DataFrame(dat)
            if 'NAME_LATN' in dat.columns:
                dat.rename(columns={"NAME_LATN": "region"}, inplace = True)
            ### merge/concatenate all metadata dataframes
            meta = pd.concat([meta, dat], axis=0, ignore_index=True)

    ### filter the concatenated meta files by actually existing sample time series (in case we lost some time series on the way)
    global fls
    fls = glob.glob(os.path.join(WORKDIR + INPUT_DATA_DIR + '/*.{}'.format('csv')))
    filter = [os.path.basename(x) for x in fls]
    filter = [os.path.splitext(x)[0] for x in filter]
    meta = meta[meta['plotID'].isin(filter)]

    ### we have to filter for specific dates, since we only use Sentinel-2 time series
    ### so end date of the time series must be after 2017-07-01
    ### this filter is an additional safety precaution to avoid all-NA time series
    ### if filtering out end dates before 2017 was overseen in dataset preparation
    meta = meta[(meta['date'] >= '2017-07-01')]

    ### create folders
    if not os.path.exists(os.path.join(WORKDIR + OUTPUT_DATA_DIR)):
        os.mkdir(os.path.join(WORKDIR + OUTPUT_DATA_DIR))
    if not os.path.exists(os.path.join(WORKDIR + OUTPUT_DATA_DIR + '/plots')):
        os.mkdir(os.path.join(WORKDIR + OUTPUT_DATA_DIR + '/plots'))
    if not os.path.exists(os.path.join(WORKDIR + OUTPUT_DATA_DIR + '/meta')):
        os.mkdir(os.path.join(WORKDIR + OUTPUT_DATA_DIR + '/meta'))

    ### define relevant bands
    global bands
    bands = ['BLU', 'GRN', 'RED', 'RE1', 'RE2', 'RE3', 'BNR', 'NIR', 'SW1', 'SW2']

    # Create a sequence of numbers for the multiprocessing pool
    MAIN_ITERATOR = range(0, len(meta))

    # assign workers by Pool method
    pool = Pool(parallel_processes)

    ### execute main function create_interpolate_datasets
    RESULTS = pool.map(create_interpolate_datasets, MAIN_ITERATOR)

    ### next line is to avoid None values in resulting list of lists,
    ### since they will throw an error
    RESULTS = [[np.nan, np.nan] if v is None else v for v in RESULTS]
    NUM_OBS = [result[0] for result in RESULTS]
    LAST_DOY = [result[1] for result in RESULTS]

    # Close the pool
    pool.close()

    ### filter meta file for csv files that have actually been written
    finalfiles = glob.glob(os.path.join(WORKDIR + OUTPUT_DATA_DIR + '/*.{}'.format('csv')))
    filter = [os.path.basename(x) for x in finalfiles]
    filter = [os.path.splitext(x)[0] for x in filter]
    meta = meta[meta['plotID'].isin(filter)]
    ### write metafile to disk
    meta.to_csv(os.path.join(WORKDIR + OUTPUT_DATA_DIR + '/meta/' +
                                'metadata.csv'), sep =';', index=False)
    
    print("End of script")