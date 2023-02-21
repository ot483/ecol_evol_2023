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
import os
from scipy.signal import argrelmax, argrelmin
import scipy
import skccm as ccm
from skccm.utilities import train_test_split


def calcCCM(df, x1, x2, prefix, d):

    x_1, x_2 = df[x1], df[x2]  
    try:
    
        #Find optimal E using simplex projection
        libstart = 1
        libend = int(len(df)*0.8)
        predstart = libend+1
        predend = len(df)
        try:
            df_EmbedDimepyEDM = pyEDM.EmbedDimension(dataFrame=df[[x1, x2]].reset_index(),
                          columns = x1,
                          maxE = 6,
                          target = x2,
                          lib = str(libstart)+" "+str(libend),
                          pred = str(predstart)+" "+str(predend),
                          showPlot=False,
                          numThreads=1) 
            
            optimalrho = df_EmbedDimepyEDM["rho"].max()
            embed = df_EmbedDimepyEDM[df_EmbedDimepyEDM["rho"] == optimalrho]["E"].values[0]
            if embed < 3:
                embed = 3
        except:
            embed=6
        ######

        #The second step of CCM is to use the S-map method to test the nonlinearity of the system. In this method, the nonlinear index (θ) is used to govern the weighting procedure, and the nonlinear dynamics system can be identified if the forecast skill improves as θ increases. In Fig. 2(d–f), the nonlinear models (θ > 0) gave better predictions than the linear model (θ = 0), which indicates statistical nonlinear behaviors in these three time series. Therefore, the CCM method can be applied to detect the causality between them.
        try:
            df_PNLpyEDM = pyEDM.PredictNonlinear(dataFrame = df[[x1,x2]].reset_index(),
                          E = int(embed),
                          columns = x1,
                          lib = str(libstart)+" "+str(libend),
                          pred = str(predstart)+" "+str(predend),
                          showPlot = False) 
            
            if (df_PNLpyEDM["rho"].max() != df_PNLpyEDM["rho"].values.tolist()[0]): \
                #and (df_PNLpyEDM["rho"].max() == df_PNLpyEDM["rho"].values.tolist()[-1]):
                    NonLinearity = True
            else:
                   NonLinearity =False
                   return [0, 0, 0, 0, False, 0, 0] 
        except:
            NonLinearity =False
        #####
        #NonLinearity = True###
        
        e1 = ccm.Embed(x_1)
        e2 = ccm.Embed(x_2)
        
        #Find optimal lag using mutual information.
        lagX1 = 2
        #if lag == 0:
        arr = e1.mutual_information(20)
        arr = pd.DataFrame(arr).ewm(span = 3).mean().values
        #pd.DataFrame(arr).plot()
        try:
            lagX1 = int(argrelmin(arr)[0][0])
        except:
            lagX1 = 2
        #print(arr[idx][0])
        
        if lagX1 < 2:
            lagX1 = 2        
               
        lagX2 = lagX1    
        
        lagX1 = int(lagX1)
        lagX2 = int(lagX2)
        embed = int(embed)        
        
        print("Selected lag "+str(lagX1))
        print("Selected embedding dim "+str(embed))


        X1 = e1.embed_vectors_1d(lagX1,embed)
        X2 = e2.embed_vectors_1d(lagX2,embed)
        
        
        #split the embedded time series
        x1tr, x1te, x2tr, x2te = train_test_split(X1,X2, percent=.75)
        
        CCM = ccm.CCM() #initiate the class
        
        #library lengths to test
        len_tr = len(x1tr)
        print("len_tr "+str(len_tr))
        #lib_lens = np.arange(10, len_tr, len_tr/2, dtype='int')
        lib_lens = list(range(10, len_tr-1, 1))
        
        #test causation
        CCM.fit(x1tr,x2tr)
        x1p, x2p = CCM.predict(x1te, x2te,lib_lengths=lib_lens)
        
        sc1,sc2 = CCM.score()
        
        df_Scores = pd.DataFrame()
        df_Scores["Library length"] = lib_lens
        df_Scores["x1"] = sc1
        df_Scores["x2"] = sc2
        
        df_Scores = df_Scores.set_index("Library length")
     
        df_Scores = df_Scores.fillna(0)
        print(df_Scores)
        Score_X1 = df_Scores["x1"].values[-3:].mean()

    #pd.DataFrame(convStd).plot()
    
    except Exception as e: 
        print(e)   
        lagX1 = 2
        embed = 5
        df_Scores, Score_X1, x1, x2, NonLinearity = 0, 0, 0, 0, False
    
    if (x1 in list(d.keys())) and (x2 in list(d.keys())):
        return [df_Scores, Score_X1, d[x1], d[x2], NonLinearity, lagX1, embed]
    else:
        return [0, 0, 0, 0, False, 0, 0]




def fullCCM(dfsList, col, targetCol, dic, prefix_, showFig = False):
    tmp_results = []
    for j in dfsList:

        j = j.fillna(0)
        tmp_results.append(calcCCM(j,
                                  x1=dic[col],
                                  x2=dic[targetCol],
                                  prefix=prefix_,
                                  d = dic.copy())) 
    Final_results = []
    calculated_dfs = []
    for k, valk in enumerate(tmp_results):
        if valk[-3] == True:
            calculated_dfs.append( valk[0].reset_index())
            Final_results.append(valk)
    
    if len(calculated_dfs) > 1:        
        c = pd.concat(calculated_dfs, axis=0, ignore_index=False)
        c_means = pd.DataFrame()
        c_means["x1_mean"] = c.groupby("Library length")["x1"].agg("mean") 
        c_means["x2_mean"] = c.groupby("Library length")["x2"].agg("mean") 
        c_means = c_means.reset_index()
        
        
        if showFig == True:         
                fig, ax = plt.subplots()
                ax.scatter(c_means["Library length"].values, c_means["x1_mean"].values, color="red", s=3)
                ax.scatter(c_means["Library length"].values, c_means["x2_mean"].values, color="gray", s=3)
                plt.savefig(BaseFolder+prefix_+"ccm_"+dic[targetCol]+"_"+dic[col]+".png")               
                plt.close()
                
        return c, c_means, Final_results   
    else:
        return 0, 0, Final_results


def calcECCM(x_1, x_2, L, E):
    e1 = ccm.Embed(x_1)
    e2 = ccm.Embed(x_2)

    X1 = e1.embed_vectors_1d(L,E)
    X2 = e2.embed_vectors_1d(L,E)
    #split the embedded time series
    x1tr, x1te, x2tr, x2te = train_test_split(X1,X2, percent=.75)
    
    CCM = ccm.CCM() #initiate the class
    
    #library lengths to test
    len_tr = len(x1tr)
    print("len_tr "+str(len_tr))
    #lib_lens = np.arange(10, len_tr, len_tr/2, dtype='int')
    lib_lens = list(range(10, len_tr, 1))
    
    #test causation
    CCM.fit(x1tr,x2tr)
    x1p, x2p = CCM.predict(x1te, x2te,lib_lengths=lib_lens)
    
    sc1,sc2 = CCM.score()
    
    df_Scores = pd.DataFrame()
    df_Scores["Library length"] = lib_lens
    df_Scores["x1"] = sc1
    df_Scores["x2"] = sc2
    
    df_Scores = df_Scores.set_index("Library length")
    
    df_Scores = df_Scores.fillna(0)
    print(df_Scores)
    Score_X1 = df_Scores["x1"].values[-10:].mean()
    Score_X2 = df_Scores["x2"].values[-10:].mean()
    
    return Score_X1, Score_X2




BaseFolder = "./"

y1p = 0.4
y2p = 0.4
y3p = 0.4
y4p = 0.4

y1 = [y1p]
y2 = [y2p]
y3 = [y3p]
y4 = [y4p]

cycles = 300

for i in range(0, cycles):
    y_1 = y1p * (3.9 - (3.9 * y1p))
    y_2 = y2p * (3.6 - (0.4 * y1p) - (3.6 * y2p))
    y_3 = y3p * (3.6 - (0.4 * y2p) - (3.6 * y3p))
    y_4 = y4p * (3.8 - (0.35 * y3p) - (3.8 * y4p))
   
    y1.append(y_1)
    y2.append(y_2)
    y3.append(y_3)
    y4.append(y_4)

    y1p = y_1
    y2p = y_2
    y3p = y_3
    y4p = y_4


concated_1 = pd.DataFrame()
 
concated_1['y1'] = y1
concated_1['y2'] = y2
concated_1['y3'] = y3
concated_1['y4'] = y4




y1p = 0.4
y2p = 0.4
y3p = 0.4
y4p = 0.4

y1 = [y1p]
y2 = [y2p]
y3 = [y3p]
y4 = [y4p]

cycles = 700
import random

for i in range(0, cycles):
    r = random.uniform(0.5, 1.5)
    y_1 = y1p * (3.9 - (3.9 * y1p))
    y_2 = y2p * (3.6 - (0.4 * y1p * r) - (3.6 * y2p))
    r = random.uniform(0.5, 1.5)    
    y_3 = y3p * (3.6 - (0.4 * y2p * r) - (3.6 * y3p))
    y_4 = y4p * (3.8 - (0.35 * y3p) - (3.8 * y4p))
   
    y1.append(y_1)
    y2.append(y_2)
    y3.append(y_3)
    y4.append(y_4)

    y1p = y_1
    y2p = y_2
    y3p = y_3
    y4p = y_4


concated_2 = pd.DataFrame()
 
concated_2['y1'] = y1
concated_2['y2'] = y2
concated_2['y3'] = y3
concated_2['y4'] = y4


concated_ = pd.concat([concated_1, concated_2], axis=0)

targetlist = ['y4'] 





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


amplified_dfs = amplifyData(df=concated_, subSetLength=999, jumpN=100)

            
DictCols = build_colsDict(concated_)

for i, vali in enumerate(amplified_dfs):
    vali.columns = [DictCols[i] for i in vali.columns]
    amplified_dfs[i] = vali


#calculate all possibilities within keepList entities
#********************************
#make multithread

from multiprocessing import Pool

Full_cols = ['y1', 'y2', 'y3', 'y4']

def manipulate(vali):
    All_CCM_dfs = []
    for j, valj in enumerate(Full_cols):
        #if j > i:
            All_CCM_dfs.append(fullCCM(dfsList=amplified_dfs,
                        col=valj,
                        targetCol=vali,
                        dic=DictCols.copy(),
                        prefix_="sim_",
                        #lag=0,
                        showFig = False))
            
            print(str(valj)+" : "+str(vali))    
    return All_CCM_dfs



import os


##multiprocessing
pool = Pool(1)

results_list_final = []

results_list_final = pool.map(manipulate, Full_cols)

results_list_fixed = []
 
for i in results_list_final:
    results_list_fixed = results_list_fixed + i
            
with open(BaseFolder+'sim_het_noW_CCM.pickle', 'wb') as handle:
    pickle.dump(results_list_fixed, handle, protocol=pickle.HIGHEST_PROTOCOL)


All_CCM_dfs = results_list_fixed



with open(BaseFolder+'sim_het_noW_CCM.pickle', 'rb') as handle:
    All_CCM_dfs = pickle.load(handle)




#check convergence

for counti, i in enumerate(All_CCM_dfs):
    All_CCM_dfs[counti] = list(All_CCM_dfs[counti])
    df_Scores = i[1]
    try:
        l=int(len(df_Scores)/2)
        if   (df_Scores["x1_mean"][l:].mean() >= df_Scores["x2_mean"][:l].mean() and \
             (df_Scores["x1_mean"][-100:].mean() >= 0.01)):
            print('dd')
            All_CCM_dfs[counti].append(True)
            
        else:
            All_CCM_dfs[counti].append(False)
    except:
        All_CCM_dfs[counti].append(False)


#check cross correlation

for counti, i in enumerate(All_CCM_dfs):
    df_Scores = i[1]
    try:
        sig1=df_Scores["x1_mean"].values.tolist()
        sig2=df_Scores["x2_mean"].values.tolist()
        corr = np.correlate(sig1,sig2,"full")
        corr = corr.max()
        
        if corr < 0.9: 
            All_CCM_dfs[counti].append(True)
            
        else:
            All_CCM_dfs[counti].append(False)
    except:
        All_CCM_dfs[counti].append(False)

CausalFeatures  = []

for i in All_CCM_dfs:
    if (len(i[2]) > 0):
        try:
            
            if (i[2][0][0]["x1"][:730][-10:].mean() >= 0.01):
                
            #if (i[-2] == True) and (i[-1] == True):
                i[2][0][0]["x1"][:730].plot()
                print(i[2][0][2] + ' ' + i[2][0][3])
                CausalFeatures.append([i[2][0][2], i[2][0][3],  i[2][0][0]["x1"][-10:].mean()])
        except:
            print('e')

df_CausalFeatures = pd.DataFrame(data=CausalFeatures, columns=['x1', 'x2', 'Score'])
df_CausalFeatures2 = df_CausalFeatures

Features = list(df_CausalFeatures['x1'].unique()) + list(df_CausalFeatures['x2'].unique())
Features = list(set(Features))

Features = Features + targetlist
Features = list(set(Features))

Features2 = Features


#Causal Network

import networkx as nx

G = nx.DiGraph() 

for i in df_CausalFeatures2[["x1", "x2", "Score"]].values.tolist():
    G.add_edge(i[0], i[1], weight = abs(i[2])*10)

largest = max(nx.kosaraju_strongly_connected_components(G), key=len)

pos = nx.spring_layout(G, dim=2, k=0.25, iterations=10)

edges = G.edges()
d = dict(G.degree)

#colors = [G[u][v]['color'] for u,v in edges]
weights = [G[u][v]['weight'] for u,v in edges]

fig = plt.figure(figsize=(50,50))

        
nx.draw(G, 
        pos, 
        with_labels=True, 
        connectionstyle='arc3, rad = 0.1', 
        node_size=[v * 1000 for v in d.values()], 
        font_size=35,
        arrowsize=20, 
        arrowstyle='fancy',
        #node_color=color_map
        )

plt.savefig(BaseFolder+"causalityNet_sim_het_noW.png")


    

os.environ['MKL_NUM_THREADS'] = '1'


import pydot
Cols = Features2




Cols = [i.replace('_', ' ') for i in Cols ]
df = concated_[Cols]



import bnlearn as bn


df_CausalFeatures2.to_csv(BaseFolder+'df_CausalFeatures2_sim_het_noW.csv')


df_CausalFeatures2 = df_CausalFeatures2[df_CausalFeatures2['Score'] >= 0.01 ]



#ECCM ###############################################


df_upsampled_proc = concated_.copy()

ll=1
ee=4
allECCM = []

from scipy.signal import hilbert

for j in edges:
    tmp_results = []
    for i in list(range(-20, 20, 1)): 
        #tmp_results = []
        #for k in amplified_dfs:
            x1 = df_upsampled_proc[j[0]].values.tolist() #k[j[0]].values.tolist()  #j[0]
            x2 = df_upsampled_proc[j[1]].values.tolist() #k[j[1]].values.tolist() #j[1]
            
            #x1 -> x2_shifted 
            df_tmp = pd.DataFrame()
            df_tmp["x1"] = x1[0:1000]
            df_tmp["x2"] = x2[0:1000]
            df_tmp["x2_shifted"] = df_tmp["x2"].to_frame().shift(periods=i)
            if i < 0:
                df_tmp = df_tmp[:i]
            if i > 0:
                df_tmp = df_tmp[i:]
                
            s1_x2Shifted, s2_x2Shifted = calcECCM(x_1=df_tmp["x1"].copy(),
                                                x_2=df_tmp["x2_shifted"].copy(),
                                                L=ll,
                                                E=ee)        
            
            #x2 -> x1_shifted
            df_tmp = pd.DataFrame()
            df_tmp["x1"] = x1[0:1000]
            df_tmp["x2"] = x2[0:1000]        
            df_tmp["x1_shifted"] = df_tmp["x1"].to_frame().shift(periods=i)
            if i < 0:
                df_tmp = df_tmp[:i]
            if i > 0:
                df_tmp = df_tmp[i:]
                
            s1_x1Shifted, s2_x1Shifted = calcECCM(x_1=df_tmp["x2"].copy(),
                                                    x_2=df_tmp["x1_shifted"].copy(),
                                                    L=ll,
                                                    E=ee)                
            
            tmp_results.append([i, s1_x2Shifted, s1_x1Shifted])
            
    df_ = pd.DataFrame(tmp_results, columns=["l", "x1", "x2"]).set_index("l")
    #save envelopes
 
    df_["x1"] = df_["x1"].rolling(5, min_periods=1, center=True).mean()
    df_["x2"] = df_["x2"].rolling(5, min_periods=1, center=True).mean()
 
    
 ###
    #cwt_peaks_x1 = scipy.signal.find_peaks_cwt(df_["x1"].values.tolist(), widths=np.arange(1, 5))
    #cwt_peaks_x2 = scipy.signal.find_peaks_cwt(df_["x2"].values.tolist(), widths=np.arange(1, 5))
    #
    #cwt_peaks_x1 = [p-20 for p in cwt_peaks_x1] 
    #cwt_peaks_x2 = [p-20 for p in cwt_peaks_x2]    
    #
    
    cwt_peaks_x1 = scipy.signal.find_peaks_cwt(df_["x1"].values.tolist(), widths=np.arange(2, 25))
    cwt_peaks_x2 = scipy.signal.find_peaks_cwt(df_["x2"].values.tolist(), widths=np.arange(2, 25))   
      
    cwt_peaks_x1 = [p-20 for p in cwt_peaks_x1] 
    cwt_peaks_x2 = [p-20 for p in cwt_peaks_x2] 


    cwt_peaks_x1 = [p for p in cwt_peaks_x1 if p <= 0] 
    cwt_peaks_x2 = [p for p in cwt_peaks_x2 if p <= 0] 
    
    #is x1 on x2 direct or indirect
    is_valid = 0
    if cwt_peaks_x1 == []:
        is_valid = 0
    elif not cwt_peaks_x2 == []:
        if (cwt_peaks_x1[-1] > cwt_peaks_x2[-1]):
            is_valid = 1 #direct
    elif (cwt_peaks_x2 == []):
            is_valid = 1 #direct
 
    allECCM.append([j[0], cwt_peaks_x1, j[1], cwt_peaks_x2, is_valid, df_])
    print("*************************************")
    print(str(i))
    print("*************************************")    
    df_.plot(legend=False).get_figure().savefig(BaseFolder+"eccm_sim_heterogenous_noW_"+str(j[0])+"_"+str(j[1]+".png"))       
 



with open(BaseFolder+'eccm_sim_heterogenous_noW_.pickle', 'wb') as handle:
    pickle.dump(allECCM, handle, protocol=pickle.HIGHEST_PROTOCOL)




df_CausalFeatures2 = pd.read_csv(BaseFolder+'df_CausalFeatures2_sim_het_noW_curated.csv')

df_CausalFeatures2 = df_CausalFeatures2[df_CausalFeatures2['Expert'] == 2]


df_violin = df_CausalFeatures2[df_CausalFeatures2['x2'].isin(targetlist)]

g = sns.boxplot(data=df_violin, x="x1", y="Score", hue="x2")
plt.show()


ll = [len(i[2]) for i in All_CCM_dfs]




Cols = list(df_CausalFeatures2['x1'].unique()) + list(df_CausalFeatures2['x2'].unique())
Cols = list(set(Cols))




df_CausalFeatures2_dag = df_CausalFeatures2.copy()


import networkx as nx

G_dag = nx.from_pandas_edgelist(df_CausalFeatures2_dag, 'x1', 'x2', create_using=nx.DiGraph())

#create DAG from existing network. 1 - find nodes without in-edges. 2 - iterate: get all out-edges, if any of those nodes go back to 
#the previous step nodes, delete the edge. 3 - get all the out-edges which go forward. 4- goto #2
G_dag_tmp = G_dag.copy()
trimmed = []
s=0
while s == 0:
    try:
        cycles = nx.find_cycle(G_dag_tmp)
        
        if (cycles[0][0] in ["Temperature"]):
            G_dag_tmp.remove_edge(cycles[-1][0], cycles[-1][1]) 
        elif (cycles[-1][0] in targetlist) and not (cycles[-1][1] in targetlist):
            G_dag_tmp.remove_edge(cycles[-1][0], cycles[-1][1])
        elif (cycles[-1][1] in targetlist) and not (cycles[0][1] in targetlist):
            G_dag_tmp.remove_edge(cycles[0][0], cycles[0][1])    
        elif (cycles[0][1] in targetlist) and not (cycles[-1][1] in targetlist):
           G_dag_tmp.remove_edge(cycles[-1][0], cycles[-1][1]) 
        elif (cycles[0][0] in targetlist):
           G_dag_tmp.remove_edge(cycles[-1][0], cycles[-1][1])          
        else:
            G_dag_tmp.remove_edge(cycles[0][0], cycles[0][1])
    except:
        print('E')
        s=1

#Trim edge phytoplankton (no out-edges)

s=0
while s == 0:
    try:
        remove = [node for node,degree in dict(G_dag_tmp.out_degree()).items() if (degree == 0) and not (node in targetlist)]

        G_dag_tmp.remove_nodes_from(remove)
        if len(remove) == 0:
            s=1
    except:
        print('E')
        s=1    

    
    
edges = G_dag_tmp.edges
DAG = bn.make_DAG(list(edges))

# Plot and make sure the arrows are correct.
bn.plot(DAG)


df_concated_fixed_outlayers = concated_

df = df_concated_fixed_outlayers[list(G_dag_tmp.nodes)]





df_cut = pd.DataFrame()
cols = []
cols_remove = []

df[df < 0] = 0


for i in df.columns:
            l = df[i].values.tolist()
            m = max(l)          
            #df_cut[i] = pd.cut(df[i], bins = [0, m, np.inf], labels=['1', '2'], include_lowest=True)
            df_cut[i] = pd.cut(df[i], bins = [0, m/20, 3*(m/5), np.inf], labels=['0', '1', '2'], include_lowest=True)
 

for i in df_cut[cols]:
    try:
        df_cut[i] = df_cut[i].astype(str)
    except:
        print('E')


edges = G_dag_tmp.edges

DAG = bn.make_DAG(list(edges))

DAG_global = bn.parameter_learning.fit(DAG, df_cut, methodtype='bayes')
DAG_global_learned = bn.structure_learning.fit(df_cut)



#compare networks - CCM and learned 
ccm_dags_djmat = DAG_global['adjmat']*1
ccm_dags_djmat = ccm_dags_djmat[['y1', 'y2', 'y3', 'y4']]
ccm_dags_djmat = ccm_dags_djmat.T[['y1', 'y2', 'y3', 'y4']].T


####FIGURE
sns.set(font_scale=3)
g = sns.clustermap(ccm_dags_djmat, col_cluster=False, row_cluster=False,  linewidths=0.1, cmap='Blues', xticklabels=True, yticklabels=True)
g.ax_row_dendrogram.set_visible(False)
g.ax_col_dendrogram.set_visible(False)
plt.show()
#######



ccm_eccm = df_CausalFeatures2.pivot(index='x1', columns='x2', values='Score')
IDX=['y1', 'y2', 'y3', 'y4']
missing = [i for i in IDX if not i in ccm_eccm.columns]
ccm_eccm[missing] = np.nan
missing = [i for i in IDX if not i in ccm_eccm.T.columns]
ccm_eccmT = ccm_eccm.T
ccm_eccmT[missing] = np.nan
ccm_eccm = ccm_eccmT[IDX].T
ccm_eccm = ccm_eccm[IDX]

sns.set(font_scale=3)
g = sns.clustermap(ccm_eccm.fillna(0), col_cluster=False, row_cluster=False,  linewidths=0.1, cmap='Blues', xticklabels=True, yticklabels=True)
g.ax_row_dendrogram.set_visible(False)
plt.show()
#######










