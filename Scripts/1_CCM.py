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
import networkx as nx
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
df_monthly_comparison_std = df_monthly_comparison_.groupby('Month')[targetlist].agg('std')/np.sqrt(20) #n = 20 years

plt.figure()
df_monthly_comparison.plot(kind='bar', yerr=df_monthly_comparison_std, legend=False)
plt.savefig(BaseFolder+'Monthly_mean.png', figsize=(10,15 ),  bbox_inches='tight' , transparent=True )



df_concated_fixed_outlayers = df_concated_fixed_outlayers[df_concated_fixed_outlayers.index <= Today]
df_concated_fixed_outlayers = df_concated_fixed_outlayers[df_concated_fixed_outlayers.index >= '2000-01-01']

df_upsampled_normalized = df_upsampled_normalized[df_upsampled_normalized.index <= Today]
df_upsampled_normalized = df_upsampled_normalized[df_upsampled_normalized.index >= '2000-01-01']



Full_cols  = list(set(Full_cols + targetlist))

###############################################

df_upsampled_proc = df_upsampled_normalized.copy()
df_upsampled_proc = df_upsampled_proc.loc['2000-01-01':]

amplified_dfs = amplifyData(df_upsampled_proc, subSetLength=100, jumpN=10)

# first differencing
#deltaX xmap Y
for i, vali in enumerate(amplified_dfs):
    for j in vali.columns:
        if j in phytoCols_0_10_names:
            #amplified_dfs[i][j] = np.log(amplified_dfs[i][j])
            if not j in targetlist:
                amplified_dfs[i][j] = vali[j].diff()
                amplified_dfs[i][j] = amplified_dfs[i][j].dropna()
                
DictCols = build_colsDict(df_upsampled_proc)

for i, vali in enumerate(amplified_dfs):
    vali.columns = [DictCols[i] for i in vali.columns]
    amplified_dfs[i] = vali



#save amplified df as pickle to be read by the external process
with open(BaseFolder+'ccm1_amplified_dfs.pickle', 'wb') as handle:
    pickle.dump(amplified_dfs, handle, protocol=pickle.HIGHEST_PROTOCOL)   

with open(BaseFolder+'ccm1_DictCols.pickle', 'wb') as handle:
    pickle.dump(DictCols, handle, protocol=pickle.HIGHEST_PROTOCOL)  

with open(BaseFolder+'ccm1_x1_x2_columns.pickle', 'wb') as handle:
    pickle.dump([Full_cols, targetlist], handle, protocol=pickle.HIGHEST_PROTOCOL)     


os.system('python '+BaseFolder+'ccm_multiproc_1.py '+ BaseFolder + ' ccm1_' )

with open(BaseFolder + 'All_' + 'ccm1_' + 'results.pickle', 'rb') as handle:
    All_CCM_dfs = pickle.load(handle)



#check convergence
for counti, i in enumerate(All_CCM_dfs):
    All_CCM_dfs[counti] = list(All_CCM_dfs[counti])
    df_Scores = i[1]
    try:
        l=int(len(df_Scores)/2)
        if ((df_Scores["x1_mean"][-10:].std() <= 0.05) == True) and \
            (df_Scores["x1_mean"][l:].mean() >= df_Scores["x2_mean"][:l].mean() and \
             (df_Scores["x1_mean"][-10:].mean() >= 0.025)):
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
            if (i[1]["x1_mean"][-10:].mean() >= 0.01) and (i[-1] == True):
                
            #if (i[-2] == True) and (i[-1] == True):
                i[1]["x1_mean"].plot()
                print(i[2][0][2] + ' ' + i[2][0][3])
                CausalFeatures.append([i[2][0][2], i[2][0][3],  i[1]["x1_mean"][-10:].mean()])
        except:
                xx=1

df_CausalFeatures = pd.DataFrame(data=CausalFeatures, columns=['x1', 'x2', 'Score'])


Features = list(df_CausalFeatures['x1'].unique()) + list(df_CausalFeatures['x2'].unique())
Features = list(set(Features))
#all causal variables vs themselvs

Features = Features + targetlist
Features = list(set(Features))

amplified_dfs = amplifyData(df_upsampled_proc[Features], subSetLength=100, jumpN=10)

# first differencing
#deltaX xmap Y
for i, vali in enumerate(amplified_dfs):
    for j in vali.columns:
        if j in phytoCols_0_10_names:
            if not j in targetlist:
                amplified_dfs[i][j] = vali[j].diff()
                amplified_dfs[i][j] = amplified_dfs[i][j].dropna()
            
DictCols = {}
DictCols = build_colsDict(df_upsampled_proc[Features])

for i, vali in enumerate(amplified_dfs):
    vali.columns = [DictCols[i] for i in vali.columns]
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

with open(BaseFolder + 'All_' + 'ccm2_'+ 'results.pickle', 'rb') as handle:
    All_causal_CCM_dfs = pickle.load(handle)


x=0
#check convergence
for counti, i in enumerate(All_causal_CCM_dfs):
    All_causal_CCM_dfs[counti] = list(All_causal_CCM_dfs[counti])
    df_Scores = i[1]
    
    try:
        l=int(len(df_Scores)/2)
        if ((df_Scores["x1_mean"][-10:].std() <= 0.05) == True) and (df_Scores["x1_mean"][l:].mean() >= df_Scores["x2_mean"][:l].mean()): 
            All_causal_CCM_dfs[counti].append(True)
            print('true')  
            x = x+1
        else:
            All_causal_CCM_dfs[counti].append(False)       
    except:
        All_causal_CCM_dfs[counti].append(False)


#Check std
CCM_sig_l = []

for counti, i in enumerate(All_causal_CCM_dfs):
    for j in i[2]:
        try:
            x1 = j[2]
            x2 = j[3]
            s = j[1]            
            CCM_sig_l.append([x1, x2, s])   
        except:
            print('E')
        
df_CCM_sig = pd.DataFrame(data=CCM_sig_l, columns=['x1', 'x2', 'score'])
df_CCM_sig['x1x2'] = df_CCM_sig['x1']+'_'+df_CCM_sig['x2']
df_CCM_sig_std = df_CCM_sig.groupby(['x1x2'])['score'].std().to_frame()
df_CCM_sig_std = df_CCM_sig_std[df_CCM_sig_std['score'] <= 0.1]

CausalFeatures2  = []

for i in All_causal_CCM_dfs:
    if (len(i[2]) > 0):
        try:
            if (i[1]["x1_mean"][-10:].mean() >= 0.05) and (i[-1] == True):
                print(i[2][0][2]+' '+i[2][0][3])    
    
                CausalFeatures2.append([i[2][0][2], i[2][0][3],  i[1]["x1_mean"][-10:].mean()])
                i[1]["x1_mean"].plot(color='gray', alpha=0.25)

        except:
            xx=1

df_CausalFeatures2 = pd.DataFrame(data=CausalFeatures2, columns=['x1', 'x2', 'Score'])



#Temperature
df_CausalFeatures2 =  df_CausalFeatures2[(df_CausalFeatures2['x2'] != 'Temperature')]

df_CausalFeatures2_temperature  = df_CausalFeatures2[(df_CausalFeatures2['Score'] > 0.01) & \
                                                     (df_CausalFeatures2['x1'] == 'Temperature')]

df_CausalFeatures2_Inflow  = df_CausalFeatures2[(df_CausalFeatures2['Score'] > 0.01) & \
                                                 (df_CausalFeatures2['x1'] == 'Inflow')]
    
#Environmental
df_CausalFeatures2_chem =  df_CausalFeatures2[((df_CausalFeatures2['x1'].isin(ChemCols_0_10_names)) | \
                                         (df_CausalFeatures2['x2'].isin(ChemCols_0_10_names))) & \
                                         (df_CausalFeatures2['Score'] >= 0.01)    ]

#targetlist
df_CausalFeatures2_cyano =  df_CausalFeatures2[((df_CausalFeatures2['x1'].isin(targetlist)) | \
                                         (df_CausalFeatures2['x2'].isin(targetlist))) & \
                                         (df_CausalFeatures2['Score'] > 0.01)    ]
#Phytoplankton
df_CausalFeatures2_phto =  df_CausalFeatures2[((df_CausalFeatures2['x1'].isin(phytoCols_0_10_names)) | \
                                         (df_CausalFeatures2['x2'].isin(phytoCols_0_10_names))) & \
                                         (df_CausalFeatures2['Score'] > 0.01)    ]#0.25

df_CausalFeatures2_taxa =  df_CausalFeatures2[((df_CausalFeatures2['x1'].isin(list(Dict_groups.values()))) | \
                                         (df_CausalFeatures2['x2'].isin(list(Dict_groups.values())))) & \
                                         (df_CausalFeatures2['Score'] > 0.01)    ]

df_CausalFeatures2 = pd.concat([df_CausalFeatures2_temperature, df_CausalFeatures2_Inflow, df_CausalFeatures2_chem, df_CausalFeatures2_cyano], axis=0)
df_CausalFeatures2 = df_CausalFeatures2.drop_duplicates()




##All_causal_CCM_dfs
ll = [len(i[2]) for i in All_causal_CCM_dfs]
sc = []
for i, vali in enumerate(All_causal_CCM_dfs):
    try:
        x1x2 =  [vali[2][0][2], vali[2][0][3]] 
        sc.append([x1x2, ll[i]])
    except:
        x=1

sc  = [i for i in sc if i[1] >= 10]
df_tmp = pd.DataFrame()

for i in sc:
    df_tmp = pd.concat([df_tmp, df_CausalFeatures2[(df_CausalFeatures2['x1'] == i[0][0]) & (df_CausalFeatures2['x2'] == i[0][1])] ], axis=0)

df_CausalFeatures2 = df_tmp.copy()
df_CausalFeatures2 = df_CausalFeatures2.drop_duplicates()


Features2 = list(df_CausalFeatures2['x1'].unique()) + list(df_CausalFeatures2['x2'].unique())
Features2 = list(set(Features2))

#Causal Network
G = nx.DiGraph() 

for i in df_CausalFeatures2[["x1", "x2", "Score"]].values.tolist():
    G.add_edge(i[0], i[1], weight = abs(i[2])*10)
 
  


df_CausalFeatures2.to_csv(BaseFolder+'CCM_CausalFeatures2_results.csv')


