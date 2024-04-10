#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: ofir
"""

#Save reqs
#python -m  pipreqs.pipreqs .


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
from sklearn.metrics import roc_auc_score
from sklearn import metrics





def to_str(l):
    return str("_".join(l))


def discreteBounds(l):
    ranges =[]   
    for k,g in groupby(enumerate(l),lambda x:x[0]-x[1]):
        group = (map(itemgetter(1),g))
        group = list(map(int,group))
        ranges.append((group[0],group[-1]))
        return ranges
    

def setBounds(p, d):
    l = []
    for i in p:
        x = d[i][-1]
        l.append(x)
    return l



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



Today = '2021-01-01'
BaseFolder = "./"


targetlist = ['2-Microcystis flos-aquae']

phytoCols_0_10_names = []

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

#fix outlayers
for i in df_concated_fixed_outlayers.columns:
    mask = (np.abs(stats.zscore(df_concated_fixed_outlayers[i])) > 3)
    df_concated_fixed_outlayers[i] = df_concated_fixed_outlayers[i].mask(mask).interpolate()
    df_interpolatedNotNormalized[i] = df_interpolatedNotNormalized[i].mask(mask).interpolate()#(method='polynomial', order=2)
   
df_concated_fixed_outlayers = df_concated_fixed_outlayers[df_concated_fixed_outlayers.index <= Today]
df_concated_fixed_outlayers = df_concated_fixed_outlayers[df_concated_fixed_outlayers.index >= '2000-01-01']

df_upsampled_normalized = df_upsampled_normalized[df_upsampled_normalized.index <= Today]
df_upsampled_normalized = df_upsampled_normalized[df_upsampled_normalized.index >= '2000-01-01']


df_CausalFeatures2 = pd.read_csv(BaseFolder+"Surr_filtered.csv")

df_CausalFeatures2['x1_group'] = df_CausalFeatures2['x1'].apply(varsToGroups)

df_CausalFeatures2_untouched = df_CausalFeatures2.copy()


            
for i in df_CausalFeatures2.columns:
    if "Unnamed" in i:
        try:
            del df_CausalFeatures2[i]
        
        except:
            print()

try:
    del df_CausalFeatures2['x1x2']
except:
    print()
    
#Add domain expert edges
DomainExp =['NH4','Port']
#DomainExp = []

df_tmp = df_CausalFeatures2[df_CausalFeatures2['x2'].isin(targetlist)]
all_x1 = df_tmp['x1'].unique()
DomEx_edges = []
for t in targetlist:
    for i in DomainExp:
            if not i in targetlist:
                new_s = 0.1
                new_timeToEffect =  6
                new_g = 'Environmental factors'
                DomEx_edges.append([i, t, new_s, 2, new_timeToEffect, new_g, ])
            
      
for i in df_CausalFeatures2.columns:
    if "Unnamed" in i:
        try:
            del df_CausalFeatures2[i]
        
        except:
            print()

try:
    del df_CausalFeatures2['x1x2']
except:
    print()
    
try:
    del df_CausalFeatures2['Score_quantile']
except:
    print()
    
df_CausalFeatures2 = pd.concat([df_CausalFeatures2, pd.DataFrame(data=DomEx_edges, columns=df_CausalFeatures2.columns)], axis=0, ignore_index=False)






Cols = list(df_CausalFeatures2['x1'].unique()) + list(df_CausalFeatures2['x2'].unique())
Cols = list(set(Cols))
Cols = [i.replace('_', ' ') for i in Cols]

#make the network as DAG, not cyclic
df_CausalFeatures2_dag = df_CausalFeatures2.copy()

G_dag = nx.from_pandas_edgelist(df_CausalFeatures2_dag, 'x1', 'x2', create_using=nx.DiGraph())

#create DAG from existing network. 
G_dag_tmp = G_dag.copy()
trimmed = []
s=0
while s == 0:
    try:
        cycles = nx.find_cycle(G_dag_tmp)
        
        if (cycles[0][0] in ["Temperature", "Inflow"]):
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
            G_dag_tmp.remove_edge(cycles[-1][0], cycles[-1][1])
    except:
        print()
        s=1

#Trim edge phytoplankton (no out-edges)
s=0
while s == 0:
    try:
        remove = [node for node,degree in dict(G_dag_tmp.out_degree()).items() if (degree == 0) and not (node in targetlist)]
        remove += [node for node,degree in dict(G_dag_tmp.in_degree()).items() if (degree == 0) and not (node in targetlist) and (node in phytoCols_0_10_names)]

        G_dag_tmp.remove_nodes_from(remove)
        if len(remove) == 0:
            s=1
    except:
        print()
        s=1    

    
edges = G_dag_tmp.edges
DAG = bn.make_DAG(list(edges))

# Plot and make sure the arrows are correct.
bn.plot(DAG)
# Check the current CPDs in the DAG.
#CPDs = bn.print_CPD(DAG)

# Learn the parameters from data set.
# As input we have the DAG without CPDs.

df = df_interpolatedNotNormalized.dropna()[list(G_dag_tmp.nodes)].dropna().copy()
df = df.resample('5D').interpolate("linear")

#fix outlayers
for i in df.columns:
    mask = (np.abs(stats.zscore(df[i])) > 3)
    df[i] = df[i].mask(mask).interpolate(method='polynomial', order=2)
    

df = df.loc['2014-01-01':]
df[df < 0] = 0
df = df.dropna()


df_cut = pd.DataFrame()
cols = []
cols_remove = []

for i in Cols:
    try:
        if (i in taxa_groups):
            ql = np.quantile(df[i].values, 0.75)    
            #qh = np.quantile(df[i].values, 0.95)       
            
            df_cut[i] = pd.cut(df[i], bins = [0, ql, np.inf], labels=['0', '1'], include_lowest=True)
            if len(df_cut[i].unique()) >= 2:
                         cols.append(i)
        elif (i in targetlist):
            ql = np.quantile(df[i].values, 0.75)    
            #qh = np.quantile(df[i].values, 0.95)       

            df_cut[i] = pd.cut(df[i], bins = [0, ql, np.inf], labels=['0', '1'], include_lowest=True)
            if len(df_cut[i].unique()) >= 2:
                         cols.append(i)
                         
        elif (i == "Temperature"):
            ql = 18.5  
            qh = 21.5
            
            df_cut[i] = pd.cut(df[i], bins = [0,  ql, qh, np.inf], labels=['0', '1', '2'], include_lowest=True)
            if len(df_cut[i].unique()) >= 2:
                         cols.append(i)
                                  
        elif (i == "Oxygen"):
                       
            qh = np.quantile(df[i].values, 0.85)
            ql = np.quantile(df[i].values, 0.6)            
            df_cut[i] = pd.cut(df[i], bins = [0,  ql, qh, np.inf], labels=['0', '1', '2'], include_lowest=True)
            if len(df_cut[i].unique()) >= 2:
                         cols.append(i)
                         
        elif (i == "Nitrit"):
                     
            qh = np.quantile(df[i].values, 0.85)
            ql = np.quantile(df[i].values, 0.55)            
            df_cut[i] = pd.cut(df[i], bins = [0,  ql, qh, np.inf], labels=['0', '1', '2'], include_lowest=True)
            if len(df_cut[i].unique()) >= 2:
                         cols.append(i)
                         

        elif (i == "Nitrate"):
                       
            qh = np.quantile(df[i].values, 0.85)
            ql = np.quantile(df[i].values, 0.55)
            df_cut[i] = pd.cut(df[i], bins = [0,  ql, qh, np.inf], labels=['0', '1', '2'], include_lowest=True)
            if len(df_cut[i].unique()) >= 2:
                         cols.append(i)                         
                         

        elif (i == "Ammonium"):
            
            qh = np.quantile(df[i].values, 0.85)
            ql = np.quantile(df[i].values, 0.55)            
            df_cut[i] = pd.cut(df[i], bins = [0,  ql, qh, np.inf], labels=['0', '1', '2'], include_lowest=True)
            if len(df_cut[i].unique()) >= 2:
                         cols.append(i)    

        elif (i == "Port"):
            
            qh = np.quantile(df[i].values, 0.85)
            ql = np.quantile(df[i].values, 0.5)            
            df_cut[i] = pd.cut(df[i], bins = [0,  ql, qh, np.inf], labels=['0', '1', '2'], include_lowest=True)
            if len(df_cut[i].unique()) >= 2:
                         cols.append(i)    
                         
                         
        else:   
            qh = np.quantile(df[i].values, 0.75)
            ql = np.quantile(df[i].values, 0.3)
            
            df_cut[i] = pd.cut(df[i], bins = [0, ql, qh, np.inf], labels=['0', '1', '2'], include_lowest=True)
            if len(df_cut[i].unique()) >= 2:
                cols.append(i)
            else:
                
                cols_remove.append(i)            
    except Exception as e:
        print(e)
        cols_remove.append(i)
        
cols = list(set(cols))
df_cut = df_cut[cols]


counts = {}
for i in cols:
        df_cut[i] = df_cut[i].astype(str)
        counts[i] = df_cut[i].value_counts()

for i in df_cut.columns:
    df_cut[i].astype(float).plot(title=i)
    plt.show()


##- shift columns according to - timeToEffect
#timetoeffect - dict
G_dag = nx.DiGraph()

for i in df_CausalFeatures2.values.tolist():
    G_dag.add_edge(i[0], i[1], weight=i[4])

lengths = dict(nx.all_pairs_dijkstra_path_length(G_dag))

dict_allLags = {}

for j in targetlist:
    for i in cols:
        try:
            dict_allLags[( i, j)] = lengths[i][j]
        except:
            print("missing interaction")
            
#loop over columns and shift according to timetoeffect dict
dict_df_cuts = {}
##TODO 
for i in targetlist:
    df_tmp = df_cut.copy()
    for j in cols:
        try:
            s = dict_allLags[(j, i)] 
            df_tmp[j] = df_tmp[j].shift(int(s*7/5)) #fix shift according to interpolation. if weekly, use s as is
        except:
            print("missing interaction")
    df_tmp = df_tmp.dropna()
    dict_df_cuts[i] = df_tmp



dict_acc = {}

for t in targetlist:
    #Split test - train
    df_cut= dict_df_cuts[t].sample(frac=0.75,random_state=42)
    df_cut_test= dict_df_cuts[t].drop(df_cut.index)
    
    #make testset balanced
    column = t
    df_cut_test = df_cut_test.groupby(column).sample(n=df_cut_test[column].value_counts().min(), random_state=42)
    
    #df_cut_test = dict_df_cuts[t]
    df_cut_test.columns = [i.replace("_", " ") for i in df_cut.columns] 
    #df_cut = dict_df_cuts[t]
    df_cut.columns = [i.replace("_", " ") for i in df_cut.columns] 
        
    edges = list(G_dag_tmp.edges)
    edges_fixed = []
    
    for i in edges:
        if (not i[0].replace(" ", "_").replace("_", " ") in cols_remove) and (not i[1].replace(" ", "_").replace("_", " ") in cols_remove):
            edges_fixed.append(i)
    
    edges = edges_fixed
    DAG = bn.make_DAG(list(edges))
    
    nodes = list(DAG['adjmat'].columns)
    
    DAG_global = bn.parameter_learning.fit(DAG, df_cut[nodes], methodtype='bayes')
    dict_df_cuts[t+"_dag_global"] = DAG_global
    #For comparison - learn structure from data using the causal features
    DAG_global_learned = bn.structure_learning.fit(df_cut[nodes])
    dict_df_cuts[t+"_dag_global_learned"] = DAG_global_learned
       
    #validate 
    dict_test = {}
    l = [list(i) for i in DAG_global['model_edges']]
    model_nodes = [item for sublist in l for item in sublist]
    model_nodes = list(set(model_nodes))
    
    cases = df_cut_test[model_nodes].values.tolist()
    keys = model_nodes
    
    all_p = []
    for i, vali in enumerate(cases):
        dict_test = {}
        for j, valj in enumerate(keys):
            dict_test[valj] = str(vali[j])
        
        for j in targetlist:
            try:
                del dict_test[j]
            except:
                print()
                
        q1 = bn.inference.fit(DAG_global, variables=[t], evidence=dict_test)
        
        all_p.append(q1.df.p[1])
    
    df_test = pd.DataFrame()
    df_test['Observed'] = df_cut_test[t].values.tolist()
    df_test['Predicted'] = all_p
    df_test = df_test.astype(float)    
    
    
    plt.figure(figsize=(15, 10))
    plt.rcParams['xtick.labelsize'] = 25
    plt.rcParams['ytick.labelsize'] = 25

    ax = df_test.reset_index().plot(kind="scatter", s=30, x="index", y="Predicted", c="orange", figsize=(15,10))
    df_test.reset_index().plot(kind="scatter", x="index", y="Observed", secondary_y=False, ax=ax, title=t)
    plt.ylabel('Probability', fontsize=25)
    plt.xlabel('Test samples', fontsize=25)
    
    plt.savefig(BaseFolder+t+'_BN_model_validation.png', bbox_inches='tight' , transparent=True)
    plt.close()
    

    plt.figure(figsize=(15, 15))
    plt.rcParams['xtick.labelsize'] = 25
    plt.rcParams['ytick.labelsize'] = 25
    
    g = sns.boxplot(x=df_test["Observed"],
                    y=df_test["Predicted"],
                    boxprops={"facecolor": (.4, .6, .8, .5)} )

    plt.legend([], [], frameon=False)
    plt.savefig(BaseFolder+t+'_BN_model_results.png',  bbox_inches='tight' , transparent=True )
    plt.close()

    def roundProb(p):
        if p >= 0.5:
            return 1
        else:
            return 0
    
    #create confusion matrix
    df_test['p_binary'] =  df_test['Predicted'].apply(roundProb)
    
    acc = accuracy_score( df_test['Observed'].values, df_test['p_binary'].values)
    cm = confusion_matrix( df_test['Observed'].values, df_test['p_binary'].values)
    
    plt.figure(figsize=(20, 15))
    sns.set(font_scale=3)
    sg = sns.heatmap(cm, annot=True, cmap="Blues")
    
    plt.ylabel('Predicted', fontsize=25)
    plt.xlabel('Observed', fontsize=25)
    
    plt.savefig(BaseFolder+t+'_BN_model_confusionMatrix.png',  bbox_inches='tight' , transparent=True )    
    plt.close()

    print(t+" acc = " + str(acc))
    dict_acc[t] = acc
    
    AllNodes = [item for sublist in DAG_global['model_edges'] for item in sublist]
    AllNodes= list(set(AllNodes))
    AllEdges = edges
    
    g = pydot.Dot()
    
    for node in AllNodes:
        if node in environmentalCols: 
            g.add_node(pydot.Node( node, color='orange', style='filled'))      
        elif node in targetlist: 
            g.add_node(pydot.Node( node, color='cyan', style='filled' ))
        else:
            g.add_node(pydot.Node( node, color='blue', style='filled' ))      
     
    for i in AllEdges:
        g.add_edge(pydot.Edge(i[0],
                        i[1],
                        color='black',
                        #style='dashed'
                        ))
    g.write_png(BaseFolder+t+"_CausalDAG_NET.png")

    rocauc = roc_auc_score(df_test['p_binary'].values, df_test['Observed'].values)
    print("roc_auc  " +str(rocauc))
    #AUC_ROC
   

with open(BaseFolder+'f_max_results.pickle', 'wb') as handle:
    pickle.dump([], handle, protocol=pickle.HIGHEST_PROTOCOL)      


def f_max(v):
    #make integer
    v = [round(i) for i in v]
    print(str(v))
    dict_evidence = {}
    l = [list(i) for i in DAG_global['model_edges']]
    model_nodes = [item for sublist in l for item in sublist]    
    for j, valj in enumerate(path[:-1]):
        try:
            if (valj in model_nodes) and (valj in AllNodes):
                    dict_evidence[valj] = str(v[j])
        except:
            print()
            
    for j in targetlist:
        try:
            del dict_evidence[j]
        except:
            print()
    print(dict_evidence) 
    q1 = bn.inference.fit(DAG_global, variables=[path[-1]], evidence=dict_evidence)
    df_q = q1.df
    #minimize min probability
    df_q_reduced = df_q[df_q[path[-1]] == '0']
    if len(df_q_reduced) > 0:
        score = df_q_reduced.p[0]   
    else:
       score = 0
       
    with open(BaseFolder + 'f_max_results.pickle', 'rb') as handle:
       listofscores = pickle.load(handle)  
       
    listofscores.append([path[-1], dict_evidence, score])
   
    with open(BaseFolder+'f_max_results.pickle', 'wb') as handle:
       pickle.dump(listofscores, handle, protocol=pickle.HIGHEST_PROTOCOL) 
   
    return 1-score 



res_sub_max = []
#res_sub_min = []

for t in targetlist:
    DAG_global = dict_df_cuts[t+'_dag_global']
    
    AllNodes = [item for sublist in DAG_global['model_edges'] for item in sublist]
    AllNodes= list(set(AllNodes))
    
    dictBounds = {}
    dict_NodesUniqueValues = {}
        
    for j in dict_df_cuts[t].columns:
        unq = list(dict_df_cuts[t][j].unique())
        unq = [int(k) for k in unq]
    
        dictBounds[j] = discreteBounds(unq)    
        dict_NodesUniqueValues[j] = [str(u) for u in unq]      
      
    #res_sub = []
    path = AllNodes
    path = [i for i in path if i != t]
    path = path+[t]
    ###
    
    bounds = setBounds(path, dictBounds) 
    
    ### create 20K permutations.
    vec=[]
    listOfRandomVecs = []
    for j in range(0,20000):
        for k, valk in enumerate(bounds):
            r=4
            while (r >= bounds[k][0]) and (r <= bounds[k][1]): 
                r=randrange(3)
            vec.append(randrange(3))
        listOfRandomVecs.append(vec)
        vec = []
        
    #max
    for j in listOfRandomVecs:
        try:       
            result = f_max(j[:-1])
            res_sub_max.append([path[0], path[-1], path, result])
        except:   
                print()

#Read all PM results. arrange as DF
with open(BaseFolder + 'f_max_results.pickle', 'rb') as handle:
    max_listofscores = pickle.load(handle)  

#prep for Df
allmax = []

for i in max_listofscores:
    tmp = list(i[1].values())
    tmp.append(i[2])
    tmp.append(i[0])
    allmax.append(tmp)

df_de_max = pd.DataFrame(data=allmax, columns=list(max_listofscores[0][1].keys())+['Score']+['y'])
df_de_max = df_de_max.drop_duplicates()
df_de_max_vecs = df_de_max[df_de_max['Score'] > 0.5]
df_de_min_vecs = df_de_max[df_de_max['Score'] < 0.5]



#####FIGURE
cols = df_de_max_vecs.columns
cols = [i for i in cols if not i in ['Score', 'y']]
df_de_max_vecs = df_de_max_vecs[cols].astype(float)




def fix_labels(l):  
    fixed_labels = l.copy()
    for i, vali in enumerate(l):
        if len(vali.split('-')) > 1:
            fixed_labels[i] = vali[2:]
        if vali == "NH4":
            fixed_labels[i] = "Ammonium"
        if vali == "Port":
            fixed_labels[i] = "Phosphate"
        if vali == "PH":
            fixed_labels[i] = "pH"
        if vali == "NH4":
            fixed_labels[i] = "Ammonium"        
        if vali == "Charyptophytes":
            fixed_labels[i] = "Cryptophytes"            
        if vali == "Hepatophyta":
            fixed_labels[i] = "haptophytes"            
        if vali == "Ntot":
            fixed_labels[i] = "Nitrogen (total)" 
        if vali == "Nitrit":
            fixed_labels[i] = "Nitrite" 
        
    label_groups = []
    for i in l:                
        if i in environmentalCols:
            label_groups.append('Environmental factors')      
        elif i in taxa_groups:
            label_groups.append('Taxonomic groups')         
        elif i == 'Temperature':
            label_groups.append('Temperature')
        elif i == 'Inflow':
            label_groups.append('Inflow') 
        else:
            label_groups.append(i)
            
    return fixed_labels, label_groups

        


#####FIGUREs

#Mean max
df_de_max = pd.DataFrame(data=allmax, columns=list(max_listofscores[0][1].keys())+['Score']+['y'])
df_de_max = df_de_max.drop_duplicates()
df_de_max = df_de_max[df_de_max['Score'] > 0.5]
plt.close()

for t in targetlist:
    l = df_de_max[[i for i in AllNodes if not i in targetlist]].astype(int).mean().reset_index().values.tolist()  
    df_mean = df_de_max[[i for i in AllNodes if not i in targetlist]].astype(int).mean()
    ll, _ = fix_labels([i[0] for i in l])
    df_mean.index = ll
    l = df_mean.reset_index().values.tolist()
    #inverse for visualization
    d={}
    for i in l:
        try:
            d[i[0]] = 1/i[1]
        except:
            d[i[0]] = 0
            
    scaler = MinMaxScaler(feature_range=(0, 0.458))
    
    scaler.fit(np.array(list(d.values())).reshape(-1, 1))
    X = scaler.transform(np.array(list(d.values())).reshape(-1, 1))
    for k, valk in enumerate(d.keys()):
        d[valk] = X[k][0]        
    
    g = pydot.Dot()
    AllNodes_ = AllNodes
    AllNodes_, _ = fix_labels(AllNodes_)
    
    edgesL_ = [i[0] for i in edges]
    edgesR_ = [i[1] for i in edges]   
    edgesL_, _ = fix_labels(edgesL_)
    edgesR_, _ = fix_labels(edgesR_)  
    edges_ = [(edgesL_[i], edgesR_[i]) for i in range(0, len(edgesL_))]
    
    for node in AllNodes_:  
        if not node in t:                
            nd = pydot.Node(node,
                            style='filled',
                            fontsize="20pt",
                            fillcolor=str(d[node])+" 1 1" )
            g.add_node(nd)
            
    for c, i in enumerate(edges):
        lbl = df_CausalFeatures2[(df_CausalFeatures2['x1'] == i[0]) &  (df_CausalFeatures2['x2'] == i[1])]['timeToEffect'].values.tolist()[0]
        if (lbl >= 0) and (lbl <= 2):
            is_direct = 'black'
        else:
            is_direct = 'gray'
        
        if lbl == 0:
            lbl = '<5'
        else:
            if str(lbl) == 'nan':
                lbl = ''
            else:
                lbl = str(int(lbl)*5) #Multiply by 5 to transfer from timesteps to days
                  
        g.add_edge(pydot.Edge(edges_[c][0],
                        edges_[c][1],
                        color = is_direct,
                        style = "filled",
                        label = lbl,
                        fontsize="20pt"
                        ))
    g.write_png(BaseFolder+"CausalDAG_NET_MAX_"+ t.replace(" ", "_") +".png")


#Mean min
df_de_max = pd.DataFrame(data=allmax, columns=list(max_listofscores[0][1].keys())+['Score']+['y'])
df_de_max = df_de_max.drop_duplicates()
df_de_max = df_de_max[df_de_max['Score'] < 0.5]

for t in targetlist:
    l = df_de_max[[i for i in AllNodes if not i in targetlist]].astype(int).mean().reset_index().values.tolist()  
    df_mean = df_de_max[[i for i in AllNodes if not i in targetlist]].astype(int).mean()
    ll, _ = fix_labels([i[0] for i in l])
    df_mean.index = ll
    l = df_mean.reset_index().values.tolist()
    #inverse for visualization
    d={}
    for i in l:
        try:
            d[i[0]] = 1/i[1]
        except:
            d[i[0]] = 0
    
    scaler = MinMaxScaler(feature_range=(0, 0.458))
    
    scaler.fit(np.array(list(d.values())).reshape(-1, 1))
    X = scaler.transform(np.array(list(d.values())).reshape(-1, 1))
    for k, valk in enumerate(d.keys()):
        d[valk] = X[k][0]
    
    g = pydot.Dot()
    AllNodes_ = AllNodes
    AllNodes_, _ = fix_labels(AllNodes_)
    
    edgesL_ = [i[0] for i in edges]
    edgesR_ = [i[1] for i in edges]   
    edgesL_, _ = fix_labels(edgesL_)
    edgesR_, _ = fix_labels(edgesR_)  
    edges_ = [(edgesL_[i], edgesR_[i]) for i in range(0, len(edgesL_))]
    
    for node in AllNodes_:  
        if not node in t:                
            nd = pydot.Node(node,
                            style='filled',
                            fontsize="20pt",
                            fillcolor=str(d[node])+" 1 1" )
            g.add_node(nd)           
    
    for c, i in enumerate(edges):
        lbl = df_CausalFeatures2[(df_CausalFeatures2['x1'] == i[0]) &  (df_CausalFeatures2['x2'] == i[1])]['timeToEffect'].values.tolist()[0]
        if (lbl >= 0) and (lbl <= 2):
            is_direct = 'black'
        else:
            is_direct = 'gray'
    
        if lbl == 0:
            lbl = '<5'
        else:
            if str(lbl) == 'nan':
                lbl = ''
            else:
                lbl = str(int(lbl)*5) #Multiply by 5 to transfer from timesteps to days
            
        g.add_edge(pydot.Edge(edges_[c][0],
                        edges_[c][1],
                        color = is_direct,
                        style = "filled",
                        label = lbl,
                        fontsize="20pt"
                        ))
    g.write_png(BaseFolder+"CausalDAG_NET_MIN_"+ t.replace(" ", "_") +".png")

df_de_max = pd.DataFrame(data=allmax, columns=list(max_listofscores[0][1].keys())+['Score']+['y'])
df_de_max = df_de_max.drop_duplicates()
df_de_max = df_de_max[df_de_max['Score'] != 0.5]



#######Figures#########
for t in targetlist:    
    DAG_global_learned = dict_df_cuts[t+"_dag_global_learned"] 
    learned_dags_djmat = DAG_global_learned['adjmat']*1 
       
    IDX = confounders + environmentalCols + taxa_groups + targetlist
    IDX = [i for i in IDX if i in learned_dags_djmat.columns]
   
    learned_dags_djmat = learned_dags_djmat[IDX]
    learned_dags_djmat = learned_dags_djmat.reindex(IDX)

    IDX, label_groups = fix_labels(IDX)
    learned_dags_djmat.columns = IDX
    learned_dags_djmat.index = IDX
    
    plt.figure(figsize=(15, 15))
    plt.rcParams['xtick.labelsize'] = 25
    plt.rcParams['ytick.labelsize'] = 25
    #plt.xlabel("Source", fontsize=25)
    #plt.ylabel("Target", fontsize=25)
    g = sns.clustermap(learned_dags_djmat, cbar=False, col_cluster=False, row_cluster=False,  linewidths=0.1, cmap='Blues', xticklabels=True, yticklabels=True,\
                       )#row_colors=row_colors, col_colors=col_colors)
    g.ax_row_dendrogram.set_visible(False)
    g.ax_col_dendrogram.set_visible(False) 

    plt.savefig(BaseFolder+t+'_learned_fromCCMfeatures_dag.png',  bbox_inches='tight' , transparent=True )
    plt.close()

    #######

    #compare networks - CCM and learned 
    dict_df_cuts[t+"_dag_global"] 
    ccm_dags_djmat = DAG_global['adjmat']*1
    
    IDX = confounders + environmentalCols + taxa_groups + targetlist
    IDX = [i for i in IDX if i in ccm_dags_djmat.columns]
      
    ccm_dags_djmat = ccm_dags_djmat[IDX]
    ccm_dags_djmat = ccm_dags_djmat.reindex(IDX)

    IDX, label_groups = fix_labels(IDX)
    ccm_dags_djmat.columns = IDX
    ccm_dags_djmat.index = IDX
        
    plt.figure(figsize=(15, 15))
    plt.rcParams['xtick.labelsize'] = 25
    plt.rcParams['ytick.labelsize'] = 25
    #sns.set(font_scale=1.25)
    g = sns.clustermap(ccm_dags_djmat, cbar=False, col_cluster=False, row_cluster=False,  linewidths=0.1, cmap='Blues', xticklabels=True, yticklabels=True,\
                       )#row_colors=row_colors, col_colors=col_colors)
    g.ax_row_dendrogram.set_visible(False)
    g.ax_col_dendrogram.set_visible(False)
    #plt.xlabel("Source", fontsize=25)
    #plt.ylabel("Target", fontsize=25)
    plt.savefig(BaseFolder+t+'_ccm_dag.png',  bbox_inches='tight' , transparent=True )
    plt.close()

    #######
 
    
df_CausalFeatures2 = pd.read_csv(BaseFolder+"Surr_filtered.csv")

ccm_eccm = df_CausalFeatures2.pivot(index='x1', columns='x2', values='Score')
IDX = confounders + environmentalCols + taxa_groups + targetlist
IDX = [i for i in IDX if i in list(ccm_eccm.columns) + list(ccm_eccm.index)]

for i in IDX:
    if not i in list(ccm_eccm.columns) :
        ccm_eccm[i] = np.nan
        
ccm_eccm = ccm_eccm[IDX]
ccm_eccm = ccm_eccm.reindex(IDX)

IDX, label_groups = fix_labels(IDX)
ccm_eccm.columns = IDX
ccm_eccm.index = IDX

plt.figure(figsize=(15, 15))
plt.rcParams['xtick.labelsize'] = 25
plt.rcParams['ytick.labelsize'] = 25
g = sns.clustermap(ccm_eccm.fillna(0), cbar=True, col_cluster=False, row_cluster=False,  linewidths=0.1, cmap='Blues', xticklabels=True, yticklabels=True  )
g.ax_row_dendrogram.set_visible(False)
plt.savefig(BaseFolder+'ccm_eccm.png',  bbox_inches='tight' , transparent=True )
plt.close()

#######

#correlations
df_cor = concated_.copy()
IDX, label_groups = fix_labels(list(df_cor.columns))

df_cor.columns = IDX

#df_cor = df_cut.astype(int).corr()
df_cor = df_cor.fillna(0)
df_cor = df_cor.round(2)

#IDX = confounders + environmentalCols + taxa_groups + targetlist
#IDX = [i for i in IDX if i in df_cor.columns]

df_cor = df_cor[ccm_eccm.columns].corr()
df_cor = df_cor.reindex(ccm_eccm.columns)

#IDX, label_groups = fix_labels(IDX)
#df_cor.columns = IDX
#df_cor.index = IDX

plt.figure(figsize=(15, 15))
plt.rcParams['xtick.labelsize'] = 25
plt.rcParams['ytick.labelsize'] = 25
g = sns.clustermap(df_cor, cbar=True, vmin=-1, vmax=1, col_cluster=False, row_cluster=False,  linewidths=0.1, cmap='coolwarm', xticklabels=True, yticklabels=True,\
                   )
g.ax_row_dendrogram.set_visible(False)
g.ax_col_dendrogram.set_visible(False)

plt.xlabel("", fontsize=25)
plt.ylabel("", fontsize=25)
plt.savefig(BaseFolder+'corr.png',  bbox_inches='tight' , transparent=True )
plt.close()


###################################################33
#Sensitivity analysis 
#SHAP values
df_de_max = pd.DataFrame(data=allmax, columns=list(max_listofscores[0][1].keys())+['Score']+['y'])
df_de_max = df_de_max.drop_duplicates()
#df_de_max = df_de_max[(df_de_max['Score'] <= 0.4) |  (df_de_max['Score'] >= 0.6)]

# Define a function to calculate the differences
def calculate_diff(lst, mean_output):
    return [abs(x - mean_output) for x in lst]


def mean_contribution(inputs, output):
    inputs = np.asarray(inputs)
    new_inputs = []
    for c, i in enumerate(inputs):
        new_inputs.append([int(j) for j in i])
    
    # Calculate the mean output
    mean_output = np.mean(output)
    num_samples, num_vars = inputs.shape
    df_tmp = pd.DataFrame(data=new_inputs)
    df_tmp["y"] = output
    df_vars_contributions = pd.DataFrame()
    
    # Iterate over each input variable
    for i in range(num_vars):       
        df_lists = df_tmp.groupby(i)['y'].aggregate(list).reset_index()            
        # Apply the function to each row of the DataFrame
        df_lists['diff_to_mean_output'] = df_lists['y'].apply(calculate_diff, mean_output=mean_output)        
        df_vars_contributions[i] = df_lists['diff_to_mean_output'].apply(lambda x: sum(x) / len(x))
        
    return df_vars_contributions






inputs = df_de_max[[i for i in df_de_max.columns if not i in ["Score", "y"]]].values.tolist()
output = df_de_max["Score"].values.tolist()
mean_output = np.mean(output)

mean_contributions = mean_contribution(inputs, output).fillna(0)

xticks = [i for i in df_de_max.columns if not i in ["Score", "y"]]

# Calculate the sum of mean contributions
sum_mean_contributions = mean_contributions.mean()
sum_mean_contributions.index = xticks


# Create bar plot
plt.figure(figsize=(15, 11))
g = sum_mean_contributions.sort_values().plot(kind="bar", color='blue')  # Bars can be blue
plt.xlabel('Input Variable')
plt.ylabel('Mean Contribution')
plt.grid(False)  # Remove grid lines
plt.gca().set_facecolor('white')  # Set background color to white
plt.savefig(BaseFolder + "sensitivity_barplot.png", bbox_inches='tight', dpi=600)
plt.show()






# scenario validation
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats



# Sample data processing
df = df_interpolatedNotNormalized.dropna()[list(G_dag_tmp.nodes)].dropna().copy()
df = df.resample('5D').interpolate("linear")


df = df.reset_index()
df['Date'] = pd.to_datetime(df['Date'])
df = df.set_index('Date')

df = df[((df.index >= '2004-10-01') & (df.index < '2005-12-01')) | ((df.index >= '2014-10-01') & (df.index < '2015-12-01'))]

for i in df.columns:
    scaler = MinMaxScaler((0,1))
    scaled_data = scaler.fit_transform(df[i].values.reshape(-1, 1))
    df[i] = [j[0] for j in scaled_data]

# Plotting
plt.figure(figsize=(10, 6))

# Plot the values of the first target column
plt.plot(df.index, df[targetlist[0]], label=targetlist[0])

# Highlight background where values are above the 0.75 quantile of the first target column
quantile_75 = df[targetlist[0]].quantile(0.8)
plt.fill_between(df.index, 0, 1, where=(df[targetlist[0]] > quantile_75), color='red', alpha=0.3)

plt.xlabel('Date')
plt.ylabel('Values')
plt.title('Time Series Plot with Background Highlight for Targetlist[0]')
plt.legend()
plt.grid(True)
plt.show()




# Create lagged values DataFrame
lagged_values_dict = {}

# Iterate over each entry in dict_allLags
for col_pair, lag in dict_allLags.items():
    # Extract column names
    col_x1, col_x2 = col_pair
    
    # Shift the DataFrame by the lag
    lagged_values_df = df[[col_x1]].shift(periods=int(lag))
    
    # Store the lagged values in the dictionary
    lagged_values_dict[col_x1] = lagged_values_df[col_x1]

# Create a DataFrame containing the lagged values
lagged_values_df = pd.DataFrame(lagged_values_dict)

# Plot the heatmap
plt.figure(figsize=(20, 20))
sns.heatmap(lagged_values_df.T, cmap='RdBu_r', center=0, cbar=True, annot=False, fmt='.2f')

# Set the size of x and y tick labels
plt.tick_params(axis='x', labelsize=24)
plt.tick_params(axis='y', labelsize=24)

# Save the heatmap plot in BaseFolder with tight bounding box
plt.savefig(BaseFolder + 'scenario_combined_heatmap.png', bbox_inches='tight')

# Display the heatmap
plt.show()