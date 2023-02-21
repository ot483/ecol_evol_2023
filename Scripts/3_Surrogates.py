#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Oct  1 19:18:36 2022

@author: ofir
"""


import numpy as np
import pandas as pd
import pickle 
import matplotlib.pyplot as plt
import seaborn as sns
import pyEDM
from sklearn.preprocessing import MinMaxScaler
from scipy import stats
import os



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
PickleFolder = "./"


targetlist = ['2-Microcystis flos-aquae']

phytoCols_0_10_names = ['3-Cyclotella-kuetzinginiana',
 '2-Cylindrospermopsis',
 '4-Chlamydomonas',
 '5-Entzia acuta',
 '5-Ceratium hirundinella',
 '2-Microcystis flos-aquae',
 '5-Peridiniopsis polonicum',
 '4-Oocystis autospore',
 '4-Scenedesmus acuminatus',
 '2-Oscillatoria thin',
 '4-Elakatothrix gelatinosa',
 '3-Rhoicosphenia',
 '4-Closterium acerosum',
 '2-Cylindrospermopsis heterocyst',
 '2-Microcystis pulvera',
 '4-Pediastrum boryanum',
 '4-Tetraedron triangulare',
 '2-Aphanocapsa elachista',
 '4-Dictyosphaerium ehrenbergianum',
 '7-Prasynophyte',
 '2-Aphanocapsa delicatissima',
 '4-Lagerheimia genevensis',
 '5-Peridiniopsis penardiforme',
 '4-Quadrigula',
 '4-Choricystis',
 '4-Pediastrum simplex',
 '3-Synedra acus',
 '4-Staurastrum gracile',
 '4-Kirchneriella lunaris',
 '4-Botryococcus braunii',
 '3-Epithemia',
 '5-Peridiniopsis borgei',
 '4-Ankistrodesmus falcutus',
 '4-Euastrum denticulatus',
 '4-Kirchneriella elongata',
 '4-Nephrocytium',
 '4-Scenedesmus ecornis',
 '4-Chlorophyte, sphere',
 '2-Merismopedia minima',
 '3-Aulacoseira granulata',
 '5-Peridiniopsis elpatiewskyi',
 '2-anabaena bergii',
 '4-Franceia ovalis',
 '5-Glenodinium oculatum',
 '4-Pediastrum sturmii',
 '4-Scenedesmus armatus',
 '4-Cosmarium sphagnicolum',
 '5-Peridiniopsis Hula',
 '2-Aphanizomenon oval',
 '4-Tetraedron new autospore',
 '0-Malomonas',
 '4-Actinastrum hantzschii',
 '3-Synedra rumpens',
 '4-Oocystis submarina',
 '5-Gymnodinium spp',
 '6-Cryptomonas spp',
 '1-Tetrachloris',
 '4-Pandorina morum',
 '3-Anomoeoneis',
 '1-Planktomyces',
 '4-Crucigenia triangularis',
 '4-Coelastrum scabrum',
 '4-Scenedesmus bicellularis',
 '2-Coelosphaerium',
 '4-Chodatella citriformis autospore',
 '4-Gloeocystis',
 '4-Tetrastrum  apiculatum',
 '4-Chlorella',
 '4-Crucigenia tetrapedia',
 '9-Erkenia subaequiciliata',
 '2-Cyanophyte-sphere',
 '4-Monoraphidium thin',
 '2-Aphanocapsa pulchra',
 '4-Tetraedron minimum',
 '4-Scenedesmus quadricauda',
 '2-Anabaena nplanktonica',
 '4-Treubaria setigera',
 '5-Peridiniopsis Protoplast',
 '3-Discostella',
 '4-Mougeotia',
 '5-Peridinium inconspicuum',
 '3-Nitzschia',
 '2-unknown filamentous cyano',
 '4-Coelastrum microporum',
 '4-Coelastrum cambricum',
 '4-Scenedesmus obliquus',
 '4-Gloeococcus',
 '0-Ochromonas',
 '4-Selenastrum minutum',
 '4-Monoraphidium arcuatum',
 '4-Cosmarium S',
 '7-Carteria cordiformis',
 '3-Synedra L',
 '4-Collodyction',
 '3-Fragilaria',
 '4-Pediastrum tetras',
 '8-Trachelomonas spp',
 '5-Peridiniopsis oculatum',
 '5-Peridinium spp',
 '3-Cyclotella meneghiniana',
 '4-Pediastrum duplex',
 '3-Synedra affinis',
 '4-Chodatella quadrisetta',
 '2-Romeria-like',
 '8-Euglena',
 '3-Diatoma',
 '4-Scenedesmus acutiformis',
 '2-Oscillatoria thick',
 '2-Merismopedia',
 '3-Cymbella',
 '3-Navicula',
 '6-Rhodomonas spp',
 '1-Acronema',
 '4-Koliella',
 '4-Scenedesmus spinosus',
 '4-Staurastrum contortum',
 '4-Chodatella citriformis',
 '2-Anabaena spiroides',
 '4-Crucigeniella rectangularis',
 '4-Spondylosium moniliforme',
 '4-Oocystis spp',
 '2-Aphanizomenon heterocyst',
 '2-Anabaena sp',
 '4-Closterium acutum',
 '0-Uroglena',
 '4-Micractinium pusillum',
 '4-Monoraphidium contortum',
 '4-Tetraedron triangulare autospore',
 '4-Scenedesmus spp',
 '4-Tetraedron new',
 '4-Scenedesmus bijuga',
 '4-Ankistrodesmus nannoselene',
 '4-Golenkinia radiata',
 '2-Aphanizomenon flos aqua',
 '4-Tetraedron quadratum',
 '4-Tetraedron caudatum',
 '4-Oocystis lacustris',
 '2-Raphidiopsis medit',
 '2-Microcystis wesenbergii',
 '4-Tetraedron regulare',
 '2-Cyanodictyon imperfectum',
 '4-Staurastrum tetracerum',
 '2-Limnothrix',
 '4-Crucigenia fenestrata',
 '4-Kirchneriella microscopica',
 '5-Peridinium gatunense cyst',
 '2-Cylindrospermopsis akinete',
 '0-Monas',
 '2-Chroococus turgidus',
 '3-Synedra spp',
 '2-Phormidium',
 '5-Peridiniopsis berolinse',
 '4-Staurastrum spp',
 '2-Cylindrospermopsis spiral',
 '4-Closterium sp',
 '5-Ceratium hirundinella cyst',
 '3-Pleurosigma',
 '3-Synedra ulna',
 '4-Cosmarium L',
 '2-Lyngbya',
 '4-Chodatella longisetta',
 '5-Glenodinium, colourless',
 '3-Synedra M',
 '2-Chroococcus minutus',
 '4-Pediastrum clathratum',
 '4-Coelastrum proboscideum',
 '5-Peridiniopsis cunningtonii',
 '3-Gomphonema',
 '3-Cyclotella polymorpha',
 '4-Chlorophyte, unknown',
 '4-Franceia radians',
 '4-Dictyosphaerium pullchelum',
 '4-Tetraedron sp',
 '5-Dinoflagellate',
 '4-Cosmarium laeve',
 '4-Closterium aciculare',
 '4-Sphaerocystis',
 '2-Merismopedia glauca',
 '2-Radiocystis geminata',
 '4-Selenastrum bibrianum',
 '4-Chodatella ciliata',
 '4-Kirchneriella obesa',
 '4-Tetrastrum triangulare',
 '4-Scenedesmus denticulatus',
 '4-Coelastrum reticulatum',
 '2-Microcystis aeruginosa',
 '4-Eudorina elegans',
 '2-Cylindrospermopsis raciborskyi',
 '2-Aphanizomenon akinete',
 '5-Peridinium gatunense',
 '4-Cocomyxa',
 '4-Nephrochlamys',
 '4-Staurastrum manfeldti',
 '4-Oocystis novae-semliae',
 '2-Microcystis botrys',
 '2-Chroococcus limneticus']


ChemCols_0_10_names = ['Nitrit',
 'Nitrate',
 'NH4',
 'Oxygen',
 'Norg_par',
 'Norg',
 'Cl',
 'So4',
 'H2S',
 'TSS',
 'PTD',
 'Norg_dis',
 'Port',
 'Turbidity',
 'PH',
 'Ntot',
 'Ptot']


confounders = ['Temperature', 'Inflow']

Dict_groups = {
    '2' : 'Cyanobacteria',
    '3' : 'Diatomaceae',
    '4' : 'Chlorophyta',
    '5' : 'Dinoflagellate',
    '6' : 'Charyptophytes',
    '7' : 'Prasinophyte',
    '9' : 'Hepatophyta'}


taxa_groups = ['Prasinophyte',
 'Hepatophyta',
 'Charyptophytes',
 'Chlorophyta',
 'Diatomaceae',
 'Dinoflagellate',
 'Cyanobacteria',
 "haptophytes",
 "Cryptophytes"]



concated_ = pd.read_csv(BaseFolder+"dataset.csv")
concated_['Date'] = pd.to_datetime(concated_['Date'])
concated_ = concated_.set_index('Date')

phytoCols_0_10_names = [i for i in phytoCols_0_10_names if not i in targetlist]

Full_cols = ChemCols_0_10_names + phytoCols_0_10_names + targetlist + confounders + taxa_groups
Full_cols = [i for i in Full_cols if not i in ["haptophytes", "Cryptophytes"]]
concated_ = concated_[Full_cols]

#Normalize 0-1
df_upsampled_normalized = pd.DataFrame(index = concated_.index)
#df_upsampled_normalized = df_concated_smoothed.copy()
AllScalersDict = {}
for i in concated_.columns:
    scaler = MinMaxScaler([0,1])
    scaled_data = scaler.fit_transform(concated_[i].values.reshape(-1, 1))
    df_upsampled_normalized[i] = [j[0] for j in scaled_data]
    AllScalersDict[i] = scaler

df_concated_fixed_outlayers = df_upsampled_normalized.copy()

#fix outlayers
for i in df_concated_fixed_outlayers.columns:
    mask = (np.abs(stats.zscore(df_concated_fixed_outlayers[i])) > 3)
    df_concated_fixed_outlayers[i] = df_concated_fixed_outlayers[i].mask(mask).interpolate()
    
   
##Figure -2 microcystis 12 months, monthly average 
df_monthly_comparison_ = concated_.copy()
df_monthly_comparison_['Month'] = pd.DatetimeIndex(df_monthly_comparison_.reset_index()['Date']).month
df_monthly_comparison = df_monthly_comparison_.groupby('Month')[targetlist].agg('mean')
df_monthly_comparison_std = df_monthly_comparison_.groupby('Month')[targetlist].agg('std')/np.sqrt(20) #years

plt.figure()
df_monthly_comparison.plot(kind='bar', yerr=df_monthly_comparison_std, legend=False)
plt.savefig(BaseFolder+'Monthly_mean.png', figsize=(10,15 ),  bbox_inches='tight' , transparent=True )



df_concated_fixed_outlayers = df_concated_fixed_outlayers[df_concated_fixed_outlayers.index <= Today]
df_concated_fixed_outlayers = df_concated_fixed_outlayers[df_concated_fixed_outlayers.index >= '2000-01-01']

df_upsampled_normalized = df_upsampled_normalized[df_upsampled_normalized.index <= Today]
df_upsampled_normalized = df_upsampled_normalized[df_upsampled_normalized.index >= '2000-01-01']






with open(BaseFolder+'AllECCM_results.pickle', 'rb') as handle:
    allECCM = pickle.load(handle)

df_CausalFeatures2 = pd.read_csv(BaseFolder+'df_CausalFeatures2_shortCCM_curated.csv')
#del df_CausalFeatures2['Unnamed: 0']

df_CausalFeatures2 = df_CausalFeatures2[df_CausalFeatures2['is_Valid'] == 2]

df_CausalFeatures2 = df_CausalFeatures2[df_CausalFeatures2['timeToEffect'] <= 20]

Features2 = list(df_CausalFeatures2['x1'].unique()) + list(df_CausalFeatures2['x2'].unique())
Features2 = list(set(Features2))

Cols = Features2
df = df_upsampled_normalized[Cols]


df_upsampled_proc = df_upsampled_normalized.copy()
df_upsampled_proc = df_upsampled_proc.loc['2000-01-01':'2020-01-01']

df_upsampled_proc_diff = df_upsampled_proc.copy()
# first differencing
#deltaX xmap Y
for j in df_upsampled_proc_diff.columns:
    if j in phytoCols_0_10_names:
        if not j in targetlist:
            df_upsampled_proc_diff[j] = df_upsampled_proc_diff[j].diff()
            df_upsampled_proc_diff[j] = df_upsampled_proc_diff[j].dropna()






df_CausalFeatures2 = df_CausalFeatures2.drop_duplicates(['x1', 'x2'])
Features2 = list(df_CausalFeatures2['x1'].unique()) + list(df_CausalFeatures2['x2'].unique())
Features2 = list(set(Features2))




os.environ['MKL_NUM_THREADS'] = '1'

pairs = df_CausalFeatures2[['x1', 'x2']].values.tolist()
pairs = [(i[0], i[1]) for i in pairs]
pairs = list(set(pairs))


x = 0

for p in pairs:
    try:
        with open(BaseFolder+'surr_2000-2020_10k_diff.pickle', 'rb') as handle:
            Dict_sur = pickle.load(handle)
    except:
        Dict_sur = {}   
        with open(BaseFolder+'surr_2000-2020_10k_diff.pickle', 'wb') as handle:
            pickle.dump(Dict_sur, handle, protocol=pickle.HIGHEST_PROTOCOL)      
        
    allKeys = Dict_sur.keys()
    if not p in allKeys:
        x=x+1
        df_sur_x1 = pyEDM.SurrogateData(dataFrame=df_upsampled_proc_diff[[p[0]]][500:1000] ,column=p[0], method= 'ebisuzaki', numSurrogates = 100,alpha= None,smooth= 0.8,outputFile= None )
        df_sur_x2 = pyEDM.SurrogateData(dataFrame=df_upsampled_proc_diff[[p[1]]][500:1000] ,column=p[1], method= 'ebisuzaki', numSurrogates = 100,alpha= None,smooth= 0.8,outputFile= None )
        Dict_sur[(p[0], p[1])] = []   
        #measure ccm and save score in a dict
        sur_cols_x1 = list(df_sur_x1.columns)
        sur_cols_x2 = list(df_sur_x2.columns)
        
        df_suf = pd.DataFrame(index=df_upsampled_proc_diff[500:1000].index)
        df_suf = pd.concat([df_sur_x1[sur_cols_x1], df_sur_x2[sur_cols_x2]], axis=1)
        
        amplified_dfs = amplifyData(df_suf, subSetLength=100, jumpN=50)#, FromYear=2001, ToYear=2020)
        DictCols = build_colsDict(df_suf)
    
        for i, val in enumerate(amplified_dfs):
            val.columns = [DictCols[i] for i in val.columns]
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
            #break
        #s = []            
        for j in tmp:
            try:
                s = j[2]['x1_mean'][-10:].mean()
                AllSurr.append([p[0], p[1], s])
            except:
                AllSurr.append([p[0], p[1], 0])
               
        Dict_sur[(p[0], p[1])].append(AllSurr)
    
        with open(BaseFolder+'surr_2000-2020_10k_diff.pickle', 'wb') as handle:
            pickle.dump(Dict_sur, handle, protocol=pickle.HIGHEST_PROTOCOL)
      
with open(BaseFolder+'surr_2000-2020_10k_diff.pickle', 'rb') as handle:
    Dict_sur = pickle.load(handle)

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
df_truth['x1x2'] = df_truth['x1']+"_"+df_truth['x2']
df_truth = df_truth.groupby('x1x2', group_keys=False).apply(lambda x: x.loc[x.Score.idxmax()])

plt.figure(figsize=(30, 20))
sns.set(font_scale=1.25)
g = sns.boxplot(data=df_AllSurr, x="x1x2", y="Score", whis=[20, 80], showfliers = False, color='gray',boxprops=dict(alpha=.3))
ax = sns.scatterplot(data=df_truth, x="x1x2", y="Score", ax= g, color="red", size=10, legend=False)
ax.set_xticklabels(ax.get_xticklabels(),rotation=90)
plt.savefig(BaseFolder+'Surr_boxplot.png', figsize=(30,20),  bbox_inches='tight' , transparent=True )

Dict_quantiles = {}

#Calculate quantile and check if it above 
for i in df_AllSurr["x1x2"].unique():
    arr = df_AllSurr[df_AllSurr["x1x2"] == i]["Score"].values
    q = np.quantile(arr, .8)
    Dict_quantiles[i] = q
    

#Filter df_CausalFeatures2 by quantile
KeepEdges = ['PTD', 'Nitrate']

df_CausalFeatures2['x1x2'] = df_CausalFeatures2['x1']+"_"+df_CausalFeatures2['x2']

filtered = []

try:
    del df_CausalFeatures2['Unnamed: 0']
except:
    print()

#Keep edges under surr quantile, but have evidence in the literature - KeepEdges
for i in df_CausalFeatures2.values.tolist():
    try:
        if (i[0] in KeepEdges) and (i[1] in targetlist):
            filtered.append(i)
    except:
        print(i[-1])
        
#filter by quantile
for i in df_CausalFeatures2.values.tolist():
    try:
        if i[2] >= Dict_quantiles[i[-1]]:
            filtered.append(i)
    except:
        print(i[-1])

df_CausalFeatures2 = pd.DataFrame(data=filtered, columns=df_CausalFeatures2.columns)


def varsToGroups(x):
    grp = ''
    if (x in targetlist):
        grp = 'Microcystis'
    elif (x in phytoCols_0_10_names) and not (x in targetlist):
        grp = 'Phytoplankton'
    elif (x in ChemCols_0_10_names ):
        grp = 'Chemistry'
    elif (x == 'Temperature'):
        grp = 'Temperature'
    elif (x == 'Inflow'):
        grp = 'Inflow'
    elif (x in Dict_groups.values()):
        grp = 'Taxonomic group'
    return grp


df_CausalFeatures2['x1_group'] = df_CausalFeatures2['x1'].apply(varsToGroups)
df_violin = df_CausalFeatures2[df_CausalFeatures2['x2'].isin(targetlist)]
df_violin = df_violin[df_violin['x1_group'].isin(['Phytoplankton', 'Chemistry'])]

plt.figure(figsize=(15, 10))
sns.set(font_scale=2)
g = sns.boxplot(data=df_violin, x="x1_group", y="Score", hue="x2").set(xlabel=None)
plt.legend([], [], frameon=False)
plt.savefig(BaseFolder+'CCM_violin.png', figsize=(25,25),  bbox_inches='tight' , transparent=True )

plt.figure(figsize=(15, 10))
sns.set(font_scale=2)
g = sns.boxplot(data=df_violin, x="x1_group", y="timeToEffect", hue="x2", ).set(xlabel=None)
plt.legend([], [], frameon=False)
plt.savefig(BaseFolder+'CCM_delay_violin.png', figsize=(25,25),  bbox_inches='tight' , transparent=True )

phytNotcyano = [i for i in phytoCols_0_10_names if not i in targetlist]
keep_list = []
for i in Features2:
    if (not i in phytNotcyano) :
        keep_list.append(i)

df_CausalFeatures2 = df_CausalFeatures2[df_CausalFeatures2['x1'].isin(keep_list) &\
                                        df_CausalFeatures2['x2'].isin(keep_list) ]

df_CausalFeatures2_untouched = df_CausalFeatures2.copy()

##If multiple targets - make interactions symmetric among the targetlist items
df_tmp = df_CausalFeatures2[df_CausalFeatures2['x2'].isin(targetlist)]
all_x1 = df_tmp['x1'].unique()
symmetry_edges = []
for t in targetlist:
    for i in all_x1:
        if len(df_tmp[(df_tmp['x1'] == i) & (df_tmp['x2'] == t)]) == 0:
            if not i in targetlist:
                new_s = list(df_tmp[(df_tmp['x1'] == i)].Score)[0]
                new_timeToEffect =  list(df_tmp[(df_tmp['x1'] == i)].timeToEffect)[0]
                new_g = list(df_tmp[(df_tmp['x1'] == i)].x1_group)[0]
                symmetry_edges.append([i, t, new_s, 2, new_timeToEffect, new_g])
            
try:
    del df_CausalFeatures2['Unnamed: 0']
except:
    print()

try:
    del df_CausalFeatures2['x1x2']
except:
    print()
    
df_CausalFeatures2 = pd.concat([df_CausalFeatures2, pd.DataFrame(data=symmetry_edges, columns=df_CausalFeatures2.columns)], axis=0, ignore_index=False)

#Add domain expert edges
DomainExp = ['NH4', 'Nitrate']

df_tmp = df_CausalFeatures2[df_CausalFeatures2['x2'].isin(targetlist)]
all_x1 = df_tmp['x1'].unique()
DomEx_edges = []
for t in targetlist:
    for i in DomainExp:
            if not i in targetlist:
                new_s = 0.1
                new_timeToEffect =  6
                new_g = 'Chemistry'
                DomEx_edges.append([i, t, new_s, 2, new_timeToEffect, new_g])
            
try:
    del df_CausalFeatures2['Unnamed: 0']
except:
    print()

try:
    del df_CausalFeatures2['x1x2']
except:
    print()

df_CausalFeatures2 = pd.concat([df_CausalFeatures2, pd.DataFrame(data=DomEx_edges, columns=df_CausalFeatures2.columns)], axis=0, ignore_index=False)

df_CausalFeatures2.to_csv(BaseFolder+"Surr_filtered.csv")

