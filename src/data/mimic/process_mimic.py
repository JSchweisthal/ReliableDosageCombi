import os
import pandas as pd
import numpy as np
os.chdir('dataset/mimic/preprocessing/')

# data available at: https://physionet.org/content/mimiciv/2.2/
# then do some preliminary filtering for the different tables
admissions = pd.read_parquet('raw/admissions.parquet')
chartevents = pd.read_parquet('raw/chartevents.parquet')
d_items = pd.read_parquet('raw/d_items.parquet')
d_labitems = pd.read_parquet('raw/d_labitems.parquet') 
icustays = pd.read_parquet('raw/icustays.parquet')
labevents = pd.read_parquet('raw/labevents.parquet')
patients = pd.read_parquet('raw/patients.parquet')

chartevents = chartevents.merge(icustays[['hadm_id', 'subject_id']], on='hadm_id')
temp = chartevents[chartevents['hadm_id'].notna()].groupby(['subject_id', 'charttime', 'itemid']).size()
temp = temp[temp>1].reset_index()[['subject_id', 'charttime']].drop_duplicates()
temp = chartevents.merge(temp, how='left', indicator=True)
temp = temp[temp['_merge'] == 'left_only'].drop('_merge', axis=1)
chartevents_wide = temp.pivot(index = ['subject_id', 'charttime'], columns='itemid', values='valuenum').reset_index()#.drop('itemid', axis=1)

resp_items = [220339, 223848, 223849, 224419, 224684, 224685, 224686, 224687,
       224695, 224696, 224697, 224700, 224701, 224702, 224705, 224706,
       224707, 224709, 224738, 224746, 224747, 224750, 226873, 227187, 220210, 224688, 224690]
chartevents_wide = chartevents_wide[chartevents_wide[resp_items].notna().any(1)]

chartevents['charttime'] = pd.to_datetime(chartevents['charttime'])
chartevents_wide['charttime'] = pd.to_datetime(chartevents_wide['charttime'])
df_merged = chartevents_wide.copy().sort_values(by=['charttime', 'subject_id'])
chartevents = chartevents.sort_values(by=['charttime', 'subject_id'])
for item in chartevents.itemid.unique(): 
    df_merged = pd.merge_asof(df_merged, chartevents.loc[chartevents['itemid']==item, ['subject_id', 'charttime', 'valuenum']], on = 'charttime', by='subject_id',
                 tolerance=pd.Timedelta('7d'), direction='backward') #
    print(item)
    df_merged[item] = np.where(df_merged[item].isna(), df_merged['valuenum'], df_merged[item])
    df_merged.drop('valuenum', axis=1, inplace=True)

# df_merged.to_pickle('mimic_processed_ventilated.pkl')
df_ventilated = df_merged
pred_items = [220045, 220048, 220050, 220051, 220052, 220179, 220180, 220181, 220210, 220277,
              220283, 220739, 220765, 223761, 223835, 223848, 223900, 223901, 224419, 224685,
              224686, 224687, 224690, 224695, 224696, 224746, 224747, 226512, 226707, 226873,
              227066, 228096, 228640]
# exclude: PEEP, temp Cels., ventilator mode, tidal volume (set), resp rate (set),
# peak inspir pressure, plateu pressure, mean airway pressure, total peep level, psv level,
# al aprv, inspiratory time, Nitric Oxide Tank Pressure, Pinsp (Draeger only), Pinsp (Draeger only)

# include selected items
df_processed = df_ventilated.loc[:, df_ventilated.columns.isin(['subject_id', 'charttime'] + pred_items)]

patients['gender'] = np.where(patients['gender']=='M', 1, 0)
df_processed = df_processed.merge(patients[['subject_id', 'gender', 'anchor_age']], on = 'subject_id', how = 'left')
col_pct = df_processed.apply(lambda x: x.notna().sum() / len(x), axis=0)
# Filter the dataframe to keep only the columns where at least 20% of the entries are not NAs
df_processed = df_processed.loc[:, col_pct >= 0.2]
# keep only rows where all entries are not NAs
df_modeling = df_processed[df_processed.notna().all(1)]
# Sort dataframe by subject and timestamp
df_modeling = df_modeling.sort_values(['subject_id', 'charttime'])
# Get index of most recent timestamp for each subject
most_recent = df_modeling.groupby('subject_id')['charttime'].idxmax()
# Select only the rows with most recent timestamp for each subject
df_modeling = df_modeling.loc[most_recent]
df_modeling = df_modeling.iloc[:, 2:]
df_modeling = df_modeling.clip(df_modeling.quantile(0.01, axis=0), df_modeling.quantile(0.99, axis=0), axis=1)
df_modeling = (df_modeling - df_modeling.min(0)) / df_modeling.max(0)

# # scale to 1 for each patient
df_modeling = df_modeling/df_modeling.sum(1).values.reshape(-1,1)

df_modeling.to_pickle('../df_mimic_x.pkl')