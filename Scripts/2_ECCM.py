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



Full_cols  = list(set(Full_cols + targetlist))

df_CausalFeatures2 = pd.read_csv(BaseFolder+'CCM_CausalFeatures2_results.csv')


Features2 = list(df_CausalFeatures2['x1'].unique()) + list(df_CausalFeatures2['x2'].unique())
Features2 = list(set(Features2))

Cols = Features2

df = df_upsampled_normalized[Cols]

with open(BaseFolder + 'All_ccm2_results.pickle', 'rb') as handle:
    All_causal_CCM_dfs = pickle.load(handle)


#ECCM ###############################################

df_CausalFeatures2 = pd.read_csv(BaseFolder+'df_CausalFeatures2_shortCCM.csv')

df_CausalFeatures2 = df_CausalFeatures2[df_CausalFeatures2["Score"] >= 0.01]


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


x1x2s = df_CausalFeatures2[['x1', 'x2']].values.tolist()
x1x2s = [(i[0], i[1]) for i in x1x2s]

with open(BaseFolder+'eccm_dataset.pickle', 'wb') as handle:
    pickle.dump(df_upsampled_proc_diff, handle, protocol=pickle.HIGHEST_PROTOCOL)      
   
with open(BaseFolder+'eccm_edges.pickle', 'wb') as handle:
    pickle.dump(x1x2s, handle, protocol=pickle.HIGHEST_PROTOCOL)   

os.system('python '+BaseFolder+'eccm_multiproc_1.py ' + BaseFolder  )




