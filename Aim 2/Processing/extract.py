if __name__ == '__main__':
    import pandas as pd
    import sqlite3
    import tsfresh
    import os, json

    # Get path and connection
    with open('cfg.json') as json_file:
        cfg = json.load(json_file)
    path_in = os.path.join('/home/idies/workspace/Storage/zmurphy3/PCM Team Emerald/Data/Processed/Merged.db')
    con = sqlite3.connect(path_in)

    insheet = pd.read_sql_query("SELECT * FROM timeseries_instantaneous", con)
    startstop = pd.read_sql_query("SELECT * FROM timeseries_startstop", con)
    static = pd.read_sql_query("SELECT * FROM static_predictors", con)
    outcomes = pd.read_sql_query("SELECT * FROM outcomes", con)

    insheet = insheet.append(startstop).sort_values('mrn_csn_pair')

    # Function to perform the feature extraction from time-series data
    # Window is size of window to use in minutes (eg window=1440 pulls the first 24 hours of data)
    def extract(window):
        # Get all unique patients so we can pull first 24 hours of data
        pats = insheet.sort_values('mrn_csn_pair')['mrn_csn_pair'].unique()

        first = insheet[(insheet['timestamp'] < window) & (insheet['timestamp'] >= 0)]
        first = first.sort_values('mrn_csn_pair').reset_index(drop=True)

        # Have to break dataset up to avoid memory issues, takes ~20 min to run
        extracted_flowsheet = pd.DataFrame()
        for i in range(8):
            split1 = int(pats.shape[0] / 8) * i
            split2 = int(pats.shape[0] / 8) * (i + 1)
            key1 = pats[split1]
            key2 = pats[split2]
            index1 = first[first['mrn_csn_pair'] == key1].index[0]
            index2 = first[first['mrn_csn_pair'] == key2].index[0] - 1
            if i == 7:
                index2 = first.shape[0] - 1
            temp = first.loc[index1 : index2, :]
            extracted_temp = tsfresh.extract_features(temp, column_id='mrn_csn_pair', column_sort='timestamp', column_kind='measure', column_value='value', n_jobs=6)
            extracted_flowsheet = extracted_flowsheet.append(extracted_temp, ignore_index=True)

        # Drop features that are only NaN
        extracted_flowsheet = extracted_flowsheet.dropna(axis=1, how='all')

        tsfresh.utilities.dataframe_functions.impute(extracted_flowsheet)
        # Add back the mrn_csn_pair
        extracted_flowsheet.insert(0, 'mrn_csn_pair', pats)

        return extracted_flowsheet

    # Function to consolidate extracted timeseries data with static features and perform feature selection
    # Window is the cutoff for a binary True for length of stay in minutes - eg 10080 is 7 days
    def select(flowsheet, window):
        # Need to change this if we are doing sliding model / changing the outcome variable (eg los > 8)
        los = (outcomes.sort_values('mrn_csn_pair')['time_in_hospital'] > window).astype(int).reset_index(drop=True)

        # Drop all columns that are only 0
        flowsheet = flowsheet.loc[:, (flowsheet != 0).any(axis=0)]

        # Do tsfresh feature filtering to dramatically reduce feature space
        feature_table = tsfresh.feature_selection.relevance.calculate_relevance_table(flowsheet.drop('mrn_csn_pair', axis=1), los,
                                                                                      n_jobs=6)
        # Concat data into one place, dropping irrelevant features
        complete = pd.concat([flowsheet.drop('mrn_csn_pair', axis=1).loc[:, feature_table['relevant']],
                              static.sort_values('mrn_csn_pair').reset_index(drop=True)], axis=1)

        # Insert LOS and  mrn_csn_pair to the data file
        complete.insert(0, 'LOS', los)
        pairs = complete['mrn_csn_pair']
        complete = complete.drop('mrn_csn_pair', axis=1)
        complete.insert(1, 'mrn_csn_pair', pairs)

        return complete

    extracted = extract(1440)
    # Can output just the extracted here before feature slection if want to run mulitple models using same window
    # of data but different hospital discharge dates
    #extracted.to_csv(os.path.join(cfg['WORKING_DATA_DIR'], 'Processed/complete_24h.csv'))
    df = select(extracted, 10080)
    print(df.shape)
    # Can output the agglomerated and feature selected databse here
    df.to_csv(os.path.join(cfg['WORKING_DATA_DIR'], 'Processed/complete_24h.csv'))


