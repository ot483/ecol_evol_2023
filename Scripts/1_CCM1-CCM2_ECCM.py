#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: ofir
"""




#from auto_shap.auto_shap import generate_shap_values, produce_shap_values_and_summary_plots
#import shap
import numpy as np
import pandas as pd
from datetime import date
import pickle 
import matplotlib.pyplot as plt
import seaborn as sns
import pyEDM
from sklearn.preprocessing import MinMaxScaler
from scipy.signal import argrelmax, argrelmin
import scipy
import skccm as ccm
from skccm.utilities import train_test_split
from scipy import stats
import bnlearn as bn
import networkx as nx
from itertools import groupby
from operator import itemgetter
import pydot
import os
import networkx as nx
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, accuracy_score
import itertools
from random import randrange
from sklearn.ensemble import RandomForestRegressor
from statsmodels.tsa.stattools import adfuller




def amplifyData(df, subSetLength=600, jumpN=30):
    allDfs = []
    for i in list(range(1, len(df)-subSetLength, jumpN)):
        tmp = df.iloc[i:i+subSetLength]
        allDfs.append(tmp)
    return allDfs

def build_colsDict(df):
    dd = {}
    counter = 0
    for i in df.columns:
        counter=counter+1
        dd[i] = "col_"+str(counter)
        dd["col_"+str(counter)] = i
    return dd



Today = '2021-01-01'
BaseFolder = "./"


targetlist = ['2-Microcystis flos-aquae']

environmentalCols = ['Nitrit',
 'Nitrate',
 'NH4',
 'Oxygen',
 'Norg_par',
 'Norg',
 'Cl',
 'TSS',
 'PTD',
 'Norg_dis',
 'Port',
 'Turbidity',
 'PH',
 ]

confounders = ['Temperature']

Dict_groups = {
    '2' : 'Cyanobacteria',
    '3' : 'Diatomaceae',
    '4' : 'Chlorophyta',
    '5' : 'Dinoflagellate',
    '6' : 'Cryptophytes',
    '7' : 'Prasinophyte',
    '9' :'Haptophytes',
   }


taxa_groups = ['Prasinophyte',
 'Chlorophyta',
 'Diatomaceae',
 'Dinoflagellate',
 'Cyanobacteria',
 "Cryptophytes",
 'Haptophytes']

concated_ = pd.read_csv(BaseFolder+"dataset.csv")
concated_['Date'] = pd.to_datetime(concated_['Date'])
concated_ = concated_.set_index('Date')

Full_cols = environmentalCols +  targetlist + confounders + taxa_groups
concated_[taxa_groups+targetlist] = concated_[taxa_groups+targetlist].replace(0, np.nan)

# Full_cols = environmentalCols +  targetlist + confounders 
# concated_[targetlist] = concated_[targetlist].replace(0, np.nan)




#Interpolated not normalized for categorization
df_interpolatedNotNormalized = concated_.copy()

#Normalize 0-1
df_upsampled_normalized = pd.DataFrame(index = concated_.index)
#df_upsampled_normalized = df_concated_smoothed.copy()
AllScalersDict = {}
for i in concated_.columns:
    scaler = MinMaxScaler((0,1))
    scaled_data = scaler.fit_transform(concated_[i].values.reshape(-1, 1))
    df_upsampled_normalized[i] = [j[0] for j in scaled_data]
    AllScalersDict[i] = scaler

df_concated_fixed_outlayers = df_upsampled_normalized.copy()

#fix outlayers
for i in df_concated_fixed_outlayers.columns:
    mask = (np.abs(stats.zscore(df_concated_fixed_outlayers[i])) > 3)
    df_concated_fixed_outlayers[i] = df_concated_fixed_outlayers[i].mask(mask).interpolate()

for i in df_interpolatedNotNormalized.columns:
    mask = (np.abs(stats.zscore(df_interpolatedNotNormalized[i])) > 3)
    df_interpolatedNotNormalized[i] = df_interpolatedNotNormalized[i].mask(mask).interpolate(method='linear')
    
df_interpolatedNotNormalized = df_interpolatedNotNormalized.resample('5D').interpolate(method='linear') #its already 7 days, this interpolation is for the case there are missing values
df_interpolatedNotNormalized[df_interpolatedNotNormalized < 0] = 0

##Figure -2a microcystis 21 year, weekly 
plt.figure()
plt.rcParams['xtick.labelsize'] = 25
plt.rcParams['ytick.labelsize'] = 25
g = df_interpolatedNotNormalized[targetlist].plot(legend=False, rot=0, figsize=(15,10))
plt.xlabel("Date", fontsize=25)
plt.ylabel("Biomass, µg/ml", fontsize=25)
plt.ylim(0, 2)
plt.savefig(BaseFolder+'Microcystis21y.png', bbox_inches='tight' , transparent=True )
plt.close()

##Figure -S2 microcystis 7 year, weekly 
plt.figure()
plt.rcParams['xtick.labelsize'] = 25
plt.rcParams['ytick.labelsize'] = 25
g = df_interpolatedNotNormalized[targetlist].loc['2014-01-01':'2021-01-01'].plot(legend=False, rot=0, figsize=(15,10))
plt.xlabel("Date", fontsize=25)
plt.ylabel("Biomass, µg/ml", fontsize=25)
plt.ylim(0, 2)
plt.savefig(BaseFolder+'Microcystis7y.png', bbox_inches='tight' , transparent=True )
plt.close()

##Figure -2b microcystis 12 months, monthly average 
df_monthly_comparison_ = concated_.copy()
df_monthly_comparison_['Month'] = pd.DatetimeIndex(df_monthly_comparison_.reset_index()['Date']).month
df_monthly_comparison = df_monthly_comparison_.groupby('Month')[targetlist].agg('mean')
df_monthly_comparison_std = df_monthly_comparison_.groupby('Month')[targetlist].agg('std')/np.sqrt(21) #years

l = ["J", "F", "M", "A", "M", "J", "J", "A", "S", "O", "N", "D"]

plt.figure()
plt.rcParams['xtick.labelsize'] = 25
plt.rcParams['ytick.labelsize'] = 25
g = df_monthly_comparison.plot(kind='bar', yerr=df_monthly_comparison_std, legend=False, rot=0, figsize=(15,10))
plt.xticks([0, 1,2,3,4,5,6,7,8,9,10,11], labels=l)
plt.xlabel("Month", fontsize=25)
plt.ylabel("Biomass, Mean µg/ml", fontsize=25)
plt.savefig(BaseFolder+'Monthly_mean.png', bbox_inches='tight' , transparent=True )
plt.close()
###############


df_concated_fixed_outlayers = df_concated_fixed_outlayers[df_concated_fixed_outlayers.index <= Today]
df_concated_fixed_outlayers = df_concated_fixed_outlayers[df_concated_fixed_outlayers.index >= '2000-01-01']
df_concated_fixed_outlayers = df_concated_fixed_outlayers.dropna()

#df_upsampled_normalized = df_upsampled_normalized[df_upsampled_normalized.index <= Today]
#df_upsampled_normalized = df_upsampled_normalized[df_upsampled_normalized.index >= '2000-01-01']











Full_cols  = list(set(Full_cols + targetlist))

###############################################

df_upsampled_proc = df_concated_fixed_outlayers.dropna().copy()
df_upsampled_proc = df_upsampled_proc.loc['2000-01-01':]


amplified_dfs = amplifyData(df_upsampled_proc, subSetLength=100, jumpN=10)
                
DictCols = build_colsDict(df_upsampled_proc)


#test for stationarity, if not stationary, create diff 
# Function to perform ADF test and difference if necessary
def make_stationary(column):
    adf_result = adfuller(column)
    p_value = adf_result[1]
    if p_value >= 0.05:  # If p-value is greater than or equal to 0.05, column is non-stationary
        diff_column = column.diff()  # Difference the column
        return diff_column
    else:
        return column


    
for i, vali in enumerate(amplified_dfs):
    vali.columns = [DictCols[i] for i in vali.columns]
    
    # Iterate over columns, perform test, and difference if necessary
    for col in vali.columns:
        vali[col] = make_stationary(vali[col])
    
    amplified_dfs[i] = vali


#save amplified df as pickle to be read by the external process
with open(BaseFolder+'ccm1_amplified_dfs.pickle', 'wb') as handle:
    pickle.dump(amplified_dfs, handle, protocol=pickle.HIGHEST_PROTOCOL)   

with open(BaseFolder+'ccm1_DictCols.pickle', 'wb') as handle:
    pickle.dump(DictCols, handle, protocol=pickle.HIGHEST_PROTOCOL)  

with open(BaseFolder+'ccm1_x1_x2_columns.pickle', 'wb') as handle:
    pickle.dump([Full_cols, targetlist], handle, protocol=pickle.HIGHEST_PROTOCOL)     

os.system('python '+BaseFolder+'ccm_multiproc_1.py '+ BaseFolder + ' ccm1_' )

with open(BaseFolder + 'All_ccm1_results.pickle', 'rb') as handle:
    All_CCM_dfs = pickle.load(handle)

#check convergence
for counti, i in enumerate(All_CCM_dfs):
    All_CCM_dfs[counti] = list(All_CCM_dfs[counti])
    df_Scores = i[1]
    try:
        l=int(len(df_Scores)/2)
        if ((df_Scores["x1_mean"][-5:].std() <= 0.2) == True) and \
            (df_Scores["x1_mean"][l:].mean() >= df_Scores["x2_mean"][:l].mean() and \
             (df_Scores["x1_mean"][-5:].mean() >= 0.01)):
            All_CCM_dfs[counti].append(True)
            print('true')
            print(All_CCM_dfs[counti][-2][-1][-4]+' ' +All_CCM_dfs[counti][-2][-1][-5])
        else:
            All_CCM_dfs[counti].append(False)
    except:
        All_CCM_dfs[counti].append(False)



# =======
plt.close()

CausalFeatures  = []

for i in All_CCM_dfs:
    if (len(i[2]) > 0):
        try:
            if (i[1]["x1_mean"][-5:].mean() >= 0.01) and (i[-1] == True):
                
            #if (i[-2] == True) and (i[-1] == True):
                i[1]["x1_mean"].plot()
                print(i[2][0][2] + ' ' + i[2][0][3])
                CausalFeatures.append([i[2][0][2], i[2][0][3],  i[1]["x1_mean"][-5:].mean()])
        except:
                xx=1

df_CausalFeatures = pd.DataFrame(data=CausalFeatures, columns=['x1', 'x2', 'Score'])
df_CausalFeatures.to_csv(BaseFolder+'CCM1_results.csv')

Features = list(df_CausalFeatures['x1'].unique()) + list(df_CausalFeatures['x2'].unique())
Features = list(set(Features))

#all causal variables vs themselvs
Features = Features + targetlist
Features = list(set(Features))
Features = [i for i in Features if i in list(concated_.columns)]

amplified_dfs = amplifyData(df_upsampled_proc[Features], subSetLength=100, jumpN=10)


DictCols = {}
DictCols = build_colsDict(df_upsampled_proc[Features])



for i, vali in enumerate(amplified_dfs):
    vali.columns = [DictCols[i] for i in vali.columns]
    
    for col in vali.columns:
        vali[col] = make_stationary(vali[col])
        
    amplified_dfs[i] = vali




#save amplified df as pickle to be read by the external process
with open(BaseFolder+'ccm2_amplified_dfs.pickle', 'wb') as handle:
    pickle.dump(amplified_dfs, handle, protocol=pickle.HIGHEST_PROTOCOL)   

with open(BaseFolder+'ccm2_DictCols.pickle', 'wb') as handle:
    pickle.dump(DictCols, handle, protocol=pickle.HIGHEST_PROTOCOL)  

with open(BaseFolder+'ccm2_x1_x2_columns.pickle', 'wb') as handle:
    pickle.dump([Features, Features], handle, protocol=pickle.HIGHEST_PROTOCOL)     

os.system('python '+BaseFolder+'ccm_multiproc_1.py ' + BaseFolder + ' ccm2_' )

#===========

with open(BaseFolder + 'All_ccm2_results.pickle', 'rb') as handle:
    All_causal_CCM_dfs = pickle.load(handle)


x=0
#check convergence
for counti, i in enumerate(All_causal_CCM_dfs):
    All_causal_CCM_dfs[counti] = list(All_causal_CCM_dfs[counti])
    df_Scores = i[1]
    
    try:
        l=int(len(df_Scores)/2)
        if  (df_Scores["x1_mean"][l:].mean() >= df_Scores["x2_mean"][:l].mean()): 
            All_causal_CCM_dfs[counti].append(True)
            print('true')  
            x = x+1
        else:
            All_causal_CCM_dfs[counti].append(False)       
    except:
        All_causal_CCM_dfs[counti].append(False)



CausalFeatures2  = []

plt.close()
for i in All_causal_CCM_dfs:
    if (len(i[2]) > 0):
        try:
            if (i[1]["x1_mean"][-10:].mean() >= 0.01) and (i[-1] == True):
                CausalFeatures2.append([i[2][0][2], i[2][0][3],  i[1]["x1_mean"][-10:].mean()])
                g = i[1]["x1_mean"][:55].plot(color='gray', alpha=0.25)
        except:
            xx=1

plt.savefig(BaseFolder+"CCM_results.png")
plt.close()

df_CausalFeatures2 = pd.DataFrame(data=CausalFeatures2, columns=['x1', 'x2', 'Score'])
df_CausalFeatures2["Score"] = df_CausalFeatures2["Score"].round(2)

df_CausalFeatures2.to_csv(BaseFolder+'CCM2_results.csv')

#filter by the 0.01 cutoff
#Cofounders are always x1

#Temperature
df_CausalFeatures2 =  df_CausalFeatures2[(df_CausalFeatures2['x2'] != 'Temperature')]

df_CausalFeatures2_temperature  = df_CausalFeatures2[(df_CausalFeatures2['Score'] > 0.01) & \
                                                     (df_CausalFeatures2['x1'] == 'Temperature')]

df_CausalFeatures2_Inflow  = df_CausalFeatures2[(df_CausalFeatures2['Score'] > 0.01) & \
                                                 (df_CausalFeatures2['x1'] == 'Inflow')]
    
#Environmental
df_CausalFeatures2_chem =  df_CausalFeatures2[((df_CausalFeatures2['x1'].isin(environmentalCols)) | \
                                         (df_CausalFeatures2['x2'].isin(environmentalCols))) & \
                                         (df_CausalFeatures2['Score'] >= 0.01)    ]

#targetlist
df_CausalFeatures2_cyano =  df_CausalFeatures2[((df_CausalFeatures2['x1'].isin(targetlist)) | \
                                         (df_CausalFeatures2['x2'].isin(targetlist))) & \
                                         (df_CausalFeatures2['Score'] > 0.01)    ]

    
df_CausalFeatures2_taxa =  df_CausalFeatures2[((df_CausalFeatures2['x1'].isin(list(Dict_groups.values()))) | \
                                         (df_CausalFeatures2['x2'].isin(list(Dict_groups.values())))) & \
                                         (df_CausalFeatures2['Score'] > 0.01)    ]

df_CausalFeatures2 = pd.concat([df_CausalFeatures2_temperature, df_CausalFeatures2_Inflow, df_CausalFeatures2_chem, df_CausalFeatures2_cyano], axis=0)
df_CausalFeatures2 = df_CausalFeatures2.drop_duplicates()

Features2 = list(df_CausalFeatures2['x1'].unique()) + list(df_CausalFeatures2['x2'].unique())
Features2 = list(set(Features2))

df_CausalFeatures2_G_piv = pd.pivot_table(df_CausalFeatures2, values='Score', index='x1', columns='x2')


#Causal Network
G = nx.DiGraph() 

for i in df_CausalFeatures2[["x1", "x2", "Score"]].values.tolist():
    G.add_edge(i[0], i[1], weight = abs(i[2])*10)

df_CausalFeatures2 = df_CausalFeatures2.assign(is_Valid=[np.nan]*len(df_CausalFeatures2))
df_CausalFeatures2 = df_CausalFeatures2.assign(timeToEffect=[np.nan]*len(df_CausalFeatures2))
df_CausalFeatures2 = df_CausalFeatures2.reset_index(drop=True)
df_CausalFeatures2.to_csv(BaseFolder+'CCM_ECCM.csv')


#ECCM ###############################################
df_CausalFeatures2 = pd.read_csv(BaseFolder+'CCM_ECCM.csv')

df_CausalFeatures2 = df_CausalFeatures2[df_CausalFeatures2["Score"] >= 0.01]


df_upsampled_proc = df_concated_fixed_outlayers.dropna().copy()
df_upsampled_proc = df_upsampled_proc.loc['2000-01-01':'2020-01-01']

df_upsampled_proc_diff = df_upsampled_proc.copy()

x1x2s = df_CausalFeatures2[['x1', 'x2']].values.tolist()
x1x2s = [(i[0], i[1]) for i in x1x2s]

with open(BaseFolder+'eccm_dataset.pickle', 'wb') as handle:
    pickle.dump(df_upsampled_proc_diff, handle, protocol=pickle.HIGHEST_PROTOCOL)      
   
with open(BaseFolder+'eccm_edges.pickle', 'wb') as handle:
    pickle.dump(x1x2s, handle, protocol=pickle.HIGHEST_PROTOCOL)   

os.system('python '+BaseFolder+'eccm_multiproc_1.py ' + BaseFolder  )




