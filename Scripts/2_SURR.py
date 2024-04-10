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






# Function to perform ADF test and difference if necessary
def make_stationary(column):
    adf_result = adfuller(column)
    p_value = adf_result[1]
    if p_value >= 0.05:  # If p-value is greater than or equal to 0.05, column is non-stationary
        diff_column = column.diff()  # Difference the column
        return diff_column
    else:
        return column


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



def varsToGroups(x):
    grp = ''
    if (x in targetlist):
        grp = 'Microcystis'
    elif (x in environmentalCols ):
        grp = 'Environmental factors'
    elif (x == 'Temperature'):
        grp = 'Environmental factors'
    elif (x == 'Inflow'):
        grp = 'Environmental factors'
    elif (x in Dict_groups.values()):
        grp = 'Taxonomic groups'
        
    return grp


def fixPair(x):
    fixed_labels = [0,0]
    k = x.split("_")
    if len(k[0].split('-')) > 1:
        fixed_labels[0] = k[0][2:]
    if k[0] == "NH4":
        fixed_labels[0] = "Ammonium"
    if k[0] == "Port":
        fixed_labels[0] = "Phosphate"
    if k[0] == "PH":
        fixed_labels[0] = "pH"
    if k[0] == "NH4":
        fixed_labels[0] = "Ammonium"                   
    if k[0] == "Nitrit":
        fixed_labels[0] = "Nitrite" 
    if k[0] == "Norg":
        fixed_labels[0] = "Nitrogen (organic)"
    if fixed_labels[0] == 0:
        fixed_labels[0] = k[0]   
    if len(k[1].split('-')) > 1:
        fixed_labels[1] = k[1][2:]
    if k[1] == "NH4":
        fixed_labels[1] = "Ammonium"
    if k[1] == "Port":
        fixed_labels[1] = "Phosphate"
    if k[1] == "PH":
        fixed_labels[1] = "pH"
    if k[1] == "NH4":
        fixed_labels[1] = "Ammonium"                   
    if k[1] == "Nitrit":
        fixed_labels[1] = "Nitrite" 
    if k[1] == "Norg":
        fixed_labels[1] = "Nitrogen (organic)"       
    if fixed_labels[1] == 0:
        fixed_labels[1] = k[1]
        
    return fixed_labels[0] + "_" + fixed_labels[1]

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
    '9' : 'Haptophytes'}

taxa_groups = ['Prasinophyte',
 'Chlorophyta',
 'Diatomaceae',
 'Dinoflagellate',
 'Cyanobacteria',
 "Haptophytes",
 "Cryptophytes"]

concated_ = pd.read_csv(BaseFolder+"dataset.csv")
concated_['Date'] = pd.to_datetime(concated_['Date'])
concated_ = concated_.set_index('Date')

Full_cols = environmentalCols +  targetlist + confounders + taxa_groups
concated_[taxa_groups+targetlist] = concated_[taxa_groups+targetlist].replace(0, np.nan)

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


for i in df_concated_fixed_outlayers.columns:
    mask = (np.abs(stats.zscore(df_concated_fixed_outlayers[i])) > 3)
    df_concated_fixed_outlayers[i] = df_concated_fixed_outlayers[i].mask(mask).interpolate(method='linear')
    
#df_concated_fixed_outlayers = df_concated_fixed_outlayers.resample('7D').interpolate(method='linear') #its already 7 days, this interpolation is for the case there are missing values
df_concated_fixed_outlayers[df_concated_fixed_outlayers < 0] = 0

df_concated_fixed_outlayers = df_concated_fixed_outlayers.dropna()


df_concated_fixed_outlayers = df_concated_fixed_outlayers[df_concated_fixed_outlayers.index <= Today]
df_concated_fixed_outlayers = df_concated_fixed_outlayers[df_concated_fixed_outlayers.index >= '2000-01-01']



#refine CCM network using eccm
#is_valid legend: 0-not valid; 1-both reponse to common strong force, at time 0 - synchrony; 2- x1 causes x2;
#extract time to effect

with open(BaseFolder + 'All_ccm2_results.pickle', 'rb') as handle:
    All_causal_CCM_dfs = pickle.load(handle)


#TODO - automatic eccm interpretation

df_CausalFeatures2 = pd.read_csv(BaseFolder+'CCM_ECCM_curated.csv')

df_CausalFeatures2 = df_CausalFeatures2[df_CausalFeatures2['is_Valid'] == 2]

df_CausalFeatures2 = df_CausalFeatures2[df_CausalFeatures2['timeToEffect'] <= 20]

##All_causal_CCM_dfs
ll = [len(i[2]) for i in All_causal_CCM_dfs]
sc = []
for i, vali in enumerate(All_causal_CCM_dfs):
    try:
        x1x2 =  [vali[2][0][2], vali[2][0][3]] 
        sc.append([x1x2, ll[i]])
    except:
        #break
        x=1

sc  = [i for i in sc if i[1] >= 10]
df_tmp = pd.DataFrame()

for i in sc:
    df_tmp = pd.concat([df_tmp, df_CausalFeatures2[(df_CausalFeatures2['x1'] == i[0][0]) & (df_CausalFeatures2['x2'] == i[0][1])] ], axis=0)

df_CausalFeatures2 = df_tmp.copy()
df_CausalFeatures2 = df_CausalFeatures2.drop_duplicates()

df_CausalFeatures2 = df_CausalFeatures2.drop_duplicates(['x1', 'x2'])

Features2 = list(df_CausalFeatures2['x1'].unique()) + list(df_CausalFeatures2['x2'].unique())
Features2 = list(set(Features2))

os.environ['MKL_NUM_THREADS'] = '1'

x1x2s = df_CausalFeatures2[['x1', 'x2']].values.tolist()
x1x2s = [(i[0], i[1]) for i in x1x2s]

pairs = x1x2s
pairs = [(i[0], i[1]) for i in pairs]
pairs = list(set(pairs))


x = 0

    
    
for p in pairs:
    try:
        with open(BaseFolder+'surr_results.pickle', 'rb') as handle:
            Dict_sur = pickle.load(handle)
    except:
        Dict_sur = {}   
        with open(BaseFolder+'surr_results.pickle', 'wb') as handle:
            pickle.dump(Dict_sur, handle, protocol=pickle.HIGHEST_PROTOCOL)      
        
    allKeys = Dict_sur.keys()
    if not p in allKeys:
        
        
            
        x=x+1
        df_sur_x1 = pyEDM.SurrogateData(dataFrame=df_concated_fixed_outlayers[[p[0]]][100:1000] ,column=p[0], method= 'ebisuzaki', numSurrogates = 33,alpha= df_concated_fixed_outlayers[[p[0]]][100:1000].std())#,smooth= 0.8,outputFile= None )
        df_sur_x2 = pyEDM.SurrogateData(dataFrame=df_concated_fixed_outlayers[[p[1]]][100:1000] ,column=p[1], method= 'ebisuzaki', numSurrogates = 33,alpha= df_concated_fixed_outlayers[[p[1]]][100:1000].std())#,smooth= 0.8,outputFile= None )
        
# =============================================================================
#         for col in df_sur_x1.columns:
#             df_sur_x1[col] = make_stationary(df_sur_x1[col])
#         
#         for col in df_sur_x2.columns:
#             df_sur_x2[col] = make_stationary(df_sur_x2[col])
# =============================================================================
        
        
        Dict_sur[(p[0], p[1])] = []   
        #measure ccm and save score in a dict
        sur_cols_x1 = list(df_sur_x1.columns)[1:]
        sur_cols_x2 = list(df_sur_x2.columns)[1:]
        
        df_suf = pd.DataFrame(index=df_concated_fixed_outlayers[:].index)
        df_suf = pd.concat([df_sur_x1[sur_cols_x1], df_sur_x2[sur_cols_x2]], axis=1)
        
        amplified_dfs = amplifyData(df_suf, subSetLength=100, jumpN=10)#, FromYear=2001, ToYear=2020)
        DictCols = build_colsDict(df_suf)
    
        for i, val in enumerate(amplified_dfs):
            val.columns = [DictCols[i] for i in val.columns]
            
            for col in val.columns:
                val[col] = make_stationary(val[col])
            
            amplified_dfs[i] = val
        
        #save amplified df as pickle to be read by the external process
        with open(BaseFolder+'surr_amplified_dfs.pickle', 'wb') as handle:
            pickle.dump(amplified_dfs, handle, protocol=pickle.HIGHEST_PROTOCOL)   
        
        with open(BaseFolder+'surr_DictCols.pickle', 'wb') as handle:
            pickle.dump(DictCols, handle, protocol=pickle.HIGHEST_PROTOCOL)  

        with open(BaseFolder+'surr_x1_x2_columns.pickle', 'wb') as handle:
            pickle.dump([sur_cols_x1, sur_cols_x2], handle, protocol=pickle.HIGHEST_PROTOCOL)      

        ##multiprocessing
        os.system('python '+BaseFolder+'ccm_multiproc_1.py '+BaseFolder + ' surr_')

        with open(BaseFolder + 'All_surr_results.pickle', 'rb') as handle:
           results_list_fixed = pickle.load(handle)   
                
        res = results_list_fixed
        tmp = []
        AllSurr=[]
        for j in res:
            try:
                s = j[1] 
                tmp.append([p[0], p[1], s])
            except:
                print('e')
               
        for j in tmp:
            try:
                s = j[2]['x1_mean'][-10:].mean()
                AllSurr.append([p[0], p[1], s])
            except:
                AllSurr.append([p[0], p[1], 0])
               
        Dict_sur[(p[0], p[1])].append(AllSurr)
    
        with open(BaseFolder+'surr_results.pickle', 'wb') as handle:
            pickle.dump(Dict_sur, handle, protocol=pickle.HIGHEST_PROTOCOL)
      
        
      
        
      
        
      
        
with open(BaseFolder+'surr_results.pickle', 'rb') as handle:
    Dict_sur = pickle.load(handle)

df_CausalFeatures2 = pd.read_csv(BaseFolder+'CCM_ECCM_curated.csv')


#Filter df_CausalFeatures2 by eccm
df_CausalFeatures2 = df_CausalFeatures2[df_CausalFeatures2['is_Valid'] == 2]
df_CausalFeatures2 = df_CausalFeatures2[df_CausalFeatures2['timeToEffect'] <= 20]

AllSurr = []
for i in Dict_sur.keys():
    if i in pairs:
        tmp = []
        for j in Dict_sur[i][0]:
            try:
                s = j[2] 
                tmp.append([i[0], i[1], s])
            except:
                print('e')
        for j in tmp:
            s = j[2] 
            AllSurr.append([i[0], i[1], s])
        

df_AllSurr = pd.DataFrame(data=AllSurr, columns=['x1', 'x2', 'Score'])
df_AllSurr['x1x2'] = df_AllSurr['x1']+"_"+df_AllSurr['x2']

df_truth = df_CausalFeatures2[['x1', 'x2', 'Score']]
df_truth = df_truth.reset_index()
df_truth['x1x2'] = df_truth['x1']+"_"+df_truth['x2']
df_truth = df_truth.groupby('x1x2', group_keys=False).apply(lambda x: x.loc[x.Score.idxmax()])
df_truth = df_truth.set_index("index").reset_index()



Dict_quantiles = {}
All_quantiles90 = []
All_quantiles95 = []
All_quantiles975 = []

#Calculate quantile and check if it above 
for i in df_AllSurr["x1x2"].unique():
    arr = df_AllSurr[df_AllSurr["x1x2"] == i]["Score"].values  
    
    q90 = np.quantile(arr, .90)    
    q95 = np.quantile(arr, .95)
    q975 = np.quantile(arr, .975)
    Dict_quantiles[i] = q95
    All_quantiles90.append([i, q90])
    All_quantiles95.append([i, q95])
    All_quantiles975.append([i, q975])

df_quantiles975 = pd.DataFrame(data=All_quantiles975, columns=["x1x2", "Score"])
df_quantiles975 = df_quantiles975.reset_index()

df_quantiles95 = pd.DataFrame(data=All_quantiles95, columns=["x1x2", "Score"])
df_quantiles95 = df_quantiles95.reset_index()

df_quantiles90 = pd.DataFrame(data=All_quantiles90, columns=["x1x2", "Score"])
df_quantiles90 = df_quantiles90.reset_index()



df_truth["x1x2"] = [fixPair(i) for i in df_truth["x1x2"].values.tolist()]
df_quantiles975["x1x2"] = [fixPair(i) for i in df_quantiles975["x1x2"].values.tolist()]
df_quantiles95["x1x2"] = [fixPair(i) for i in df_quantiles95["x1x2"].values.tolist()]
df_quantiles90["x1x2"] = [fixPair(i) for i in df_quantiles90["x1x2"].values.tolist()]
df_AllSurr["x1x2"] = [fixPair(i) for i in df_AllSurr["x1x2"].values.tolist()]


plt.figure(figsize=(25, 15))

plt.rcParams.update({'font.size': 24})

g = df_AllSurr.plot(kind="scatter", x="x1x2", y="Score" , color='gray', alpha=0.01, s=100, marker="o", figsize=(25, 15))
ax = df_truth.plot(kind="scatter", x="x1x2", y="Score" , color='red', s=20, marker="o", ax=g)
ax2 = df_quantiles975.plot(kind="scatter", x="x1x2", y="Score" , color='red', s=100, marker="_", ax=ax)
ax3 = df_quantiles95.plot(kind="scatter", x="x1x2", y="Score" , color='blue', s=100, marker="_", ax=ax2)
ax4 = df_quantiles90.plot(kind="scatter", x="x1x2", y="Score" , color='black', s=100, marker="_", ax=ax3)
ax4.set_xticklabels(ax.get_xticklabels(),rotation=90)
plt.savefig(BaseFolder+'Surr_plot.png',  bbox_inches='tight' , transparent=True )
plt.close()


#Filter df_CausalFeatures2 by eccm
df_CausalFeatures2 = pd.read_csv(BaseFolder+'CCM_ECCM_curated.csv')

df_CausalFeatures2 = df_CausalFeatures2[df_CausalFeatures2['is_Valid'] == 2]
df_CausalFeatures2 = df_CausalFeatures2[df_CausalFeatures2['timeToEffect'] <= 20]

#Filter df_CausalFeatures2 by quantile
df_CausalFeatures2['x1x2'] = df_CausalFeatures2['x1']+"_"+df_CausalFeatures2['x2']
try:
    del df_quantiles90["index"]
except:
    print()


df_CausalFeatures2["x1x2"] = [fixPair(i) for i in df_CausalFeatures2["x1x2"].values.tolist()]
# =============================================================================
df_quantiles90.columns = ["x1x2", "Score_quantile"]
df_quantiles90["Score_quantile"] = df_quantiles90["Score_quantile"].round(2)

# df_quantiles95 = df_quantiles95.reset_index()
# df_quantiles95 = df_quantiles95[["x1x2", "Score"]]
# df_quantiles95.columns = ["x1x2", "Score_quantile"]
# =============================================================================
df_CausalFeatures2 = pd.merge(df_CausalFeatures2, df_quantiles90, on="x1x2")
df_CausalFeatures2 = df_CausalFeatures2[df_CausalFeatures2["Score"] >= df_CausalFeatures2["Score_quantile"]]

try:
    del df_CausalFeatures2['Unnamed: 0']
except:
    print()

###################s



df_CausalFeatures2.to_csv(BaseFolder+"Surr_filtered.csv")


df_CausalFeatures2 = pd.read_csv(BaseFolder+'CCM_ECCM_curated.csv')

df_CausalFeatures2['x1_group'] = df_CausalFeatures2['x1'].apply(varsToGroups)
df_truth['x1_group'] = df_truth['x1'].apply(varsToGroups)


#FIGUREs - env and biology affect Microcystis
df_boxplot = df_CausalFeatures2[df_CausalFeatures2['x2'].isin(targetlist)]
df_boxplot = df_boxplot[df_boxplot['x1_group'].isin(['Environmental factors', 'Taxonomic groups'])]

df_boxplot_lags = df_CausalFeatures2[df_CausalFeatures2['x2'].isin(targetlist)]
df_boxplot_lags = df_boxplot_lags[df_boxplot_lags['x1_group'].isin(['Environmental factors', 'Taxonomic groups'])]


plt.figure(figsize=(15, 10))
plt.rcParams['xtick.labelsize'] = 25
plt.rcParams['ytick.labelsize'] = 25
g = sns.boxplot(data=df_boxplot, x="x1_group", y="Score", hue="x2").set(xlabel=None)
plt.legend([], [], frameon=False)

plt.xlabel("Group affects M. flos-aquae", fontsize=25)
plt.ylabel("CCM Score (p)", fontsize=25)

plt.savefig(BaseFolder+'CCM_microcystis_boxplot.png', bbox_inches='tight' , transparent=True )
plt.close()




plt.figure(figsize=(15, 10))
plt.rcParams['xtick.labelsize'] = 25
plt.rcParams['ytick.labelsize'] = 25

# Your boxplot code
g = sns.boxplot(data=df_boxplot_lags, x="x1_group", y="timeToEffect", hue="x2").set(xlabel=None)
plt.legend([], [], frameon=False)

plt.xlabel("Group affects M. flos-aquae", fontsize=25)
plt.ylabel("Delayed effect (days)", fontsize=25)

# Multiply y-axis values by 5
plt.gca().set_yticklabels([int(tick) * 5 for tick in plt.gca().get_yticks()])

plt.savefig(BaseFolder+'CCM_microcystis_delay_boxplot.png', bbox_inches='tight', transparent=True)
plt.close()







