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
from sklearn.preprocessing import MinMaxScaler
import bnlearn as bn
import networkx as nx
from itertools import groupby
from operator import itemgetter
import pydot
from sklearn.metrics import confusion_matrix, accuracy_score
from random import randrange
from sklearn.ensemble import RandomForestRegressor




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


df_CausalFeatures2 = pd.read_csv(BaseFolder+"Surr_filtered.csv")
df_CausalFeatures2_untouched = df_CausalFeatures2.copy()


Cols = list(df_CausalFeatures2['x1'].unique()) + list(df_CausalFeatures2['x2'].unique())
Cols = list(set(Cols))
Cols = [i.replace('_', ' ') for i in Cols]

#make the network as DAG, not cyclic
df_CausalFeatures2_dag = df_CausalFeatures2.copy()

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
            G_dag_tmp.remove_edge(cycles[-1][0], cycles[-1][1])
    except:
        print('E')
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
        print('E')
        s=1    

    
edges = G_dag_tmp.edges
DAG = bn.make_DAG(list(edges))

# Plot and make sure the arrows are correct.
bn.plot(DAG)


df =  concated_[concated_.index <= Today][list(G_dag_tmp.nodes)]


df = df.resample('1D').mean().interpolate("linear")

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
            ql = np.quantile(df[i].values, 0.8)    
            #qh = np.quantile(df[i].values, 0.95)       

            df_cut[i] = pd.cut(df[i], bins = [0, ql, np.inf], labels=['0', '1'], include_lowest=True)
            if len(df_cut[i].unique()) >= 2:
                         cols.append(i)
                         
        elif (i == "Temperature"):
            ql = 18.5  
            qh = 21
            
            df_cut[i] = pd.cut(df[i], bins = [0,  ql, qh, np.inf], labels=['0', '1', '2'], include_lowest=True)
            if len(df_cut[i].unique()) >= 2:
                         cols.append(i)

        elif (i == "PH"):
            ql = 8.25
            qh = 8.45
              
            df_cut[i] = pd.cut(df[i], bins = [0,  ql, qh, np.inf], labels=['0', '1', '2'], include_lowest=True)
            if len(df_cut[i].unique()) >= 2:
                         cols.append(i)
                                  
        elif (i == "Oxygen"):
            #ql = 5  
            #qh = 8.25
            
            qh = np.quantile(df[i].values, 0.9)
            ql = np.quantile(df[i].values, 0.6)            
            df_cut[i] = pd.cut(df[i], bins = [0,  ql, qh, np.inf], labels=['0', '1', '2'], include_lowest=True)
            if len(df_cut[i].unique()) >= 2:
                         cols.append(i)
                         
        elif (i == "Nitrate"):
            #ql = 0.175
            #qh = 0.25
            
            qh = np.quantile(df[i].values, 0.8)
            ql = np.quantile(df[i].values, 0.4)
            df_cut[i] = pd.cut(df[i], bins = [0,  ql, qh, np.inf], labels=['0', '1', '2'], include_lowest=True)
            if len(df_cut[i].unique()) >= 2:
                         cols.append(i)                         
                         

        elif (i == "Inflow"):
            #ql = 7 
            #qh = 16

            qh = np.quantile(df[i].values, 0.8)
            ql = np.quantile(df[i].values, 0.5)            
            df_cut[i] = pd.cut(df[i], bins = [0,  ql, qh, np.inf], labels=['0', '1', '2'], include_lowest=True)
            if len(df_cut[i].unique()) >= 2:
                         cols.append(i)    

        elif (i == "Ntot"):
            ql = 0.5 
            qh = 0.75
            
            df_cut[i] = pd.cut(df[i], bins = [0,  ql, qh, np.inf], labels=['0', '1', '2'], include_lowest=True)
            if len(df_cut[i].unique()) >= 2:
                         cols.append(i)    
                         
        else:   
            qh = np.quantile(df[i].values, 0.85)
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


#print all cols
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
            dict_allLags[( i, j)] = lengths[j][i]
        except:
            print("missing interaction")
            
#loop over columns and shift according to timetoeffect dict
dict_df_cuts = {}

for i in targetlist:
    df_tmp = df_cut.copy()
    for j in cols:
        try:
            s = dict_allLags[(j, i)] 
            df_tmp[j] = df_tmp[j].shift(int(s)*7) 
        except:
            print("missing interaction")
    df_tmp = df_tmp.dropna()
    dict_df_cuts[i] = df_tmp





dict_acc = {}

for t in targetlist:
    #Split test - train
    df_cut= dict_df_cuts[t].sample(frac=0.8,random_state=123)
    df_cut_test= dict_df_cuts[t].drop(df_cut.index)

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
    nodes = list(set( [item for sublist in edges_fixed for item in sublist]))
    nodes = [i.replace(" ", "_") for i in nodes]
    nodes = [i.replace("_", " ") for i in nodes]
    
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
        
        all_p.append(q1.df.p[1])#+q1.df.p[2])
    
    df_test = pd.DataFrame()
    df_test['Observed'] = df_cut_test[t].values.tolist()
    df_test['Predicted'] = all_p
    df_test = df_test.astype(float)    
    
    plt.figure(figsize=(10, 5))
    sns.set(font_scale=1.25)
    ax = df_test.Observed.plot()
    df_test.Predicted.plot(secondary_y=False, ax=ax, title=t)
    plt.savefig(BaseFolder+t+'_BN_model_validation.png', bbox_inches='tight' , transparent=True )

    plt.figure(figsize=(20, 15))
    g = sns.boxplot(x=df_test["Observed"],
                    y=df_test["Predicted"],
                    boxprops={"facecolor": (.4, .6, .8, .5)} )
    
    plt.legend([], [], frameon=False)
    plt.savefig(BaseFolder+t+'_BN_model_results.png',  bbox_inches='tight' , transparent=True )
    
    #create confusion matrix
    df_test['p_binary'] =  df_test['Predicted'].round()
    acc = accuracy_score( df_test['Observed'].values, df_test['p_binary'].values)
    cm = confusion_matrix( df_test['Observed'].values, df_test['p_binary'].values)
    
    plt.figure(figsize=(20, 15))
    sns.set(font_scale=3)
    sg = sns.heatmap(cm, annot=True, cmap="Blues")
    plt.savefig(BaseFolder+t+'_BN_model_confusionMatrix.png',  bbox_inches='tight' , transparent=True )    
    
    print(t+" acc = " + str(acc))
    dict_acc[t] = acc
    
    AllNodes = [item for sublist in DAG_global['model_edges'] for item in sublist]
    AllNodes= list(set(AllNodes))
    AllEdges = edges
    
    g = pydot.Dot()
    
    for node in AllNodes:
        if (node in phytoCols_0_10_names) and (node not in targetlist):
            g.add_node(pydot.Node( node, color='green', style='filled' ))
        elif node in ChemCols_0_10_names: 
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



with open(BaseFolder+'f_max_results.pickle', 'wb') as handle:
    pickle.dump([], handle, protocol=pickle.HIGHEST_PROTOCOL)      


def f_max(v):
    #make integer
    v = [round(i) for i in v]
    print(str(v))
    dict_evidence = {}
    #print(path)
    l = [list(i) for i in DAG_global['model_edges']]
    model_nodes = [item for sublist in l for item in sublist]    
    for j, valj in enumerate(path[:-1]):
        try:
            if (valj in model_nodes) and (valj in AllNodes):
                    dict_evidence[valj] = str(v[j])
        except:
            print('E')
            
    for j in targetlist:
        try:
            del dict_evidence[j]
        except:
            print()
    print(dict_evidence) 
    q1 = bn.inference.fit(DAG_global, variables=[path[-1]], evidence=dict_evidence)
    df_q = q1.df
    #minimize min probability
    df_q_reduced = df_q[df_q[path[-1]] == 0]
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


def to_str(l):
    return str("_".join(l))



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
    
    ### create 100K permutations.
    vec=[]
    listOfRandomVecs = []
    for j in range(0,100000):
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

plt.figure(figsize=(15, 5))
sns.set(font_scale=1.25)
g = sns.clustermap(df_de_max_vecs, col_cluster=False, row_cluster=False,  linewidths=0.1, cmap='RdBu', xticklabels=True, yticklabels=True)
g.ax_row_dendrogram.set_visible(False)
plt.savefig(BaseFolder+'BN_vectors_allmax.png',  bbox_inches='tight' , transparent=True )



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
        
    label_groups = []
    for i in l:
        print(i)
        if (i in phytoCols_0_10_names) or (i in targetlist):
            label_groups.append('Phytoplankton')
                
        elif i in ChemCols_0_10_names:
            label_groups.append('Chemistry')
                
        elif i in taxa_groups:
            label_groups.append('Taxa')    
                
        elif i == 'Temperature':
            label_groups.append('Temperature')
        elif i == 'Inflow':
            label_groups.append('Inflow') 
        else:
            label_groups.append(i)
            
    return fixed_labels, label_groups



#####FIGURE
cols = df_de_min_vecs.columns
cols = [i for i in cols if not i in ['Score', 'y']]
df_de_min_vecs = df_de_min_vecs[cols].astype(float)

plt.figure(figsize=(15, 5))
sns.set(font_scale=1.25)
g = sns.clustermap(df_de_min_vecs, col_cluster=False, row_cluster=False,  linewidths=0.1, cmap='RdBu', xticklabels=True, yticklabels=True)
g.ax_row_dendrogram.set_visible(False)
plt.savefig(BaseFolder+'BN_vectors_allmin.png',  bbox_inches='tight' , transparent=True )




#sensitivity analysis using RF

def RF_sensitivity(df_, title):    
    clf = RandomForestRegressor(max_depth=3, random_state=0)
    c = [i for i in df_.columns if i != 'Score']
    c = [i for i in c if i != 'y']
    clf.fit(df_[c], df_['Score'])
    importances = clf.feature_importances_
    #std = np.std([tree.feature_importances_ for tree in clf.estimators_], axis=0)
    forest_importances = pd.Series(importances, index=c)
    IDX = confounders + ChemCols_0_10_names + taxa_groups + targetlist
    IDX = [i for i in IDX if i in forest_importances.index]
    forest_importances = forest_importances.reindex(IDX)
    IDX, _ = fix_labels(IDX)
    forest_importances.index = IDX
    #forest_importances.columns = IDX
    plt.figure(figsize=(25, 20))
    f = forest_importances.plot.bar(title=title)
    plt.rc('xtick', labelsize=40) 
    plt.rc('ytick', labelsize=40) 
    plt.savefig(BaseFolder+title+'_sensitivity.png',  bbox_inches='tight' , transparent=True )
    return forest_importances



df_de_max = pd.DataFrame(data=allmax, columns=list(max_listofscores[0][1].keys())+['Score']+['y'])
df_de_max = df_de_max.drop_duplicates()
df_all_de = df_de_max.copy()



#####FIGURE
cols = df_all_de.columns
cols = [i for i in cols if not i in ['y']]
cols = [i for i in cols if i in taxa_groups] + [i for i in cols if i in ChemCols_0_10_names] + ['Score']
df_all_de = df_all_de[cols].astype(float)
#df_all_de = df_all_de[df_all_de['Score'] != 0.5]
df_all_de = df_all_de.sort_values(by='Score')
df_all_de = df_all_de.fillna(0)

plt.figure(figsize=(5, 15))
#sns.set(font_scale=1.25)
gg = sns.heatmap(df_all_de, xticklabels=True, yticklabels=False)
#gg.ax_row_dendrogram.set_visible(False)
plt.savefig(BaseFolder+'BN_vectors_all.png',  bbox_inches='tight' , transparent=True )


df_de_max = pd.DataFrame(data=allmax, columns=list(max_listofscores[0][1].keys())+['Score']+['y'])
df_de_max = df_de_max.drop_duplicates()
df_all_de = df_de_max.copy()
df_all_de = df_all_de[df_all_de['Score'] != 0.5] 


dict_FI = {}
for i in targetlist:
    dict_FI[i] = RF_sensitivity(df_all_de[df_all_de['y'] == i], title="Sensitivity-" + i)
    df = pd.DataFrame(dict_FI[i])
    #df = df[df[0] >= 0.01]
    dict_FI[i] = df  



#Figure  max scenario
#https://graphviz.org/Gallery/neato/color_wheel.svg

for t in targetlist:
    d = {i[0] : 1/i[1] for i in dict_FI[t].reset_index().values.tolist()}
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
            nd = pydot.Node( node, style='filled', fillcolor=str(d[node])+" 1 1" )
            g.add_node(nd)
              
    for c, i in enumerate(edges):
        lbl = df_CausalFeatures2[(df_CausalFeatures2['x1'] == i[0]) &  (df_CausalFeatures2['x2'] == i[1])]['timeToEffect'].values.tolist()[0]
        if (lbl >= 0) and (lbl <= 2):
            is_direct = 'filled'
        else:
            is_direct = 'dashed'
        
        if lbl == 0:
            lbl = '<1'
        else:
            if str(lbl) == 'nan':
                lbl = ''
            else:
                lbl = str(int(lbl))
            
        g.add_edge(pydot.Edge(edges_[c][0],
                        edges_[c][1],
                        color = 'black',
                        style = is_direct,
                        label = lbl
                        ))
    g.write_png(BaseFolder+"CausalDAG_NET_SENSITIVITY_"+ t.replace(" ", "_") +".png")


#Mean max
df_de_max = pd.DataFrame(data=allmax, columns=list(max_listofscores[0][1].keys())+['Score']+['y'])
df_de_max = df_de_max.drop_duplicates()
df_de_max = df_de_max[df_de_max['Score'] > 0.5]


for t in targetlist:
    l = df_de_max[[i for i in AllNodes if not i in targetlist]].astype(int).mean().reset_index().values.tolist()  
    df_mean = df_de_max[[i for i in AllNodes if not i in targetlist]].astype(int).mean()
    ll, _ = fix_labels([i[0] for i in l])
    df_mean.index = ll
    l = df_mean.reset_index().values.tolist()
    d = {i[0] : 1/i[1] for i in l}
    
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
            nd = pydot.Node( node, style='filled', fillcolor=str(d[node])+" 1 1" )
            g.add_node(nd)
            
    
    for c, i in enumerate(edges):
        lbl = df_CausalFeatures2[(df_CausalFeatures2['x1'] == i[0]) &  (df_CausalFeatures2['x2'] == i[1])]['timeToEffect'].values.tolist()[0]
        if (lbl >= 0) and (lbl <= 2):
            is_direct = 'filled'
        else:
            is_direct = 'dashed'
    
        
        if lbl == 0:
            lbl = '<1'
        else:
            if str(lbl) == 'nan':
                lbl = ''
            else:
                lbl = str(int(lbl))
        
            
        g.add_edge(pydot.Edge(edges_[c][0],
                        edges_[c][1],
                        color = 'black',
                        style = is_direct,
                        label = lbl
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
    d = {i[0] : 1/i[1] for i in l}
    
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
            nd = pydot.Node( node, style='filled', fillcolor=str(d[node])+" 1 1" )
            g.add_node(nd)           
    
    for c, i in enumerate(edges):
        lbl = df_CausalFeatures2[(df_CausalFeatures2['x1'] == i[0]) &  (df_CausalFeatures2['x2'] == i[1])]['timeToEffect'].values.tolist()[0]
        if (lbl >= 0) and (lbl <= 2):
            is_direct = 'filled'
        else:
            is_direct = 'dashed'
    
        
        if lbl == 0:
            lbl = '<1'
        else:
            if str(lbl) == 'nan':
                lbl = ''
            else:
                lbl = str(int(lbl))
        
            
        g.add_edge(pydot.Edge(edges_[c][0],
                        edges_[c][1],
                        color = 'black',
                        style = is_direct,
                        label = lbl
                        ))
    g.write_png(BaseFolder+"CausalDAG_NET_MIN_"+ t.replace(" ", "_") +".png")




df_de_max = pd.DataFrame(data=allmax, columns=list(max_listofscores[0][1].keys())+['Score']+['y'])
df_de_max = df_de_max.drop_duplicates()
df_de_max = df_de_max[df_de_max['Score'] != 0.5]

#co-occurance analysis
def coocc(df):
    dict_coocc = {}
    cols = [c for c in df.columns if not c in ['Score', 'y']]
    df = df[cols]
    for c in cols:
        df[c] = df[c].astype(str).replace('1', c).replace('2', c)
    
    dfT = df.T
    for c in dfT.columns:
        l = dfT[c].values.tolist()
        for ii in cols:
            for jj in cols:
                if ii != jj:
                    if (ii in l) and (jj in l):
                        if not ((ii, jj) in list(dict_coocc.keys())):
                            dict_coocc[(ii, jj)] = 0
                            dict_coocc[(jj, ii)] = 0
                        else:
                            dict_coocc[(ii, jj)] += 1
                            dict_coocc[(jj, ii)] += 1
 
    return dict_coocc

dict_all_cooc = {}

for i in targetlist:
    cols_sens =  list(dict_FI[i].index)
  
    #Max
    df_max = df_de_max[df_de_max['Score'] > 0.5]
    cols, _ = fix_labels(list(df_max.columns))
    df_max.columns = cols
    
    max_cooc = coocc(df_max[cols_sens])
    
    #Min
    df_min = df_de_max[df_de_max['Score'] < 0.5]
    cols, _ = fix_labels(list(df_min.columns))
    df_min.columns = cols
    
    min_cooc = coocc(df_min[cols_sens])

    df_minCooc = pd.DataFrame(columns=cols_sens, index=cols_sens)
    df_maxCooc = pd.DataFrame(columns=cols_sens, index=cols_sens)
    
    for j in max_cooc.keys():
            #try:
                df_maxCooc.at[j[0], j[1]] = max_cooc[j]
                df_maxCooc.at[j[1], j[0]] = max_cooc[j]
    for j in min_cooc.keys():
            #try:
                df_minCooc.at[j[0], j[1]] = min_cooc[j]
                df_minCooc.at[j[1], j[0]] = min_cooc[j]
             
    
    df_maxCooc = df_maxCooc.fillna(0)
    df_minCooc = df_minCooc.fillna(0)
   
    df_maxCooc = df_maxCooc.fillna(0)
    df_minCooc = df_minCooc.fillna(0)   
    
    dict_all_cooc[i+" max"] = df_maxCooc
    dict_all_cooc[i+" min"] = df_minCooc
       
    plt.figure(figsize=(15, 5))
    sns.set(font_scale=1.25)
    g = sns.clustermap(df_maxCooc, col_cluster=False, row_cluster=False,  linewidths=0.1, cmap='RdBu_r', xticklabels=True, yticklabels=True)
    g.ax_row_dendrogram.set_visible(False)
    plt.savefig(BaseFolder+i+'_coocurance_max.png',  bbox_inches='tight' , transparent=True )

    plt.figure(figsize=(15, 5))
    sns.set(font_scale=1.25)
    g = sns.clustermap(df_minCooc, col_cluster=False, row_cluster=False,  linewidths=0.1, cmap='RdBu_r', xticklabels=True, yticklabels=True)
    g.ax_row_dendrogram.set_visible(False)
    plt.savefig(BaseFolder+i+'_coocurance_min.png',  bbox_inches='tight' , transparent=True )

  
#######Figures#########

for t in targetlist:    
    DAG_global_learned = dict_df_cuts[t+"_dag_global_learned"] 
    learned_dags_djmat = DAG_global_learned['adjmat']*1 
       
    
    IDX = confounders + ChemCols_0_10_names + taxa_groups + targetlist
    IDX = [i for i in IDX if i in learned_dags_djmat.columns]
   
    learned_dags_djmat = learned_dags_djmat[IDX]
    learned_dags_djmat = learned_dags_djmat.reindex(IDX)

    IDX, label_groups = fix_labels(IDX)
    learned_dags_djmat.columns = IDX
    learned_dags_djmat.index = IDX
    
    plt.figure(figsize=(15, 15))
    sns.set(font_scale=1.25)
    g = sns.clustermap(learned_dags_djmat, col_cluster=False, row_cluster=False,  linewidths=0.1, cmap='Blues', xticklabels=True, yticklabels=True,\
                       )#row_colors=row_colors, col_colors=col_colors)
    g.ax_row_dendrogram.set_visible(False)
    g.ax_col_dendrogram.set_visible(False) 
    plt.savefig(BaseFolder+t+'_learned_fromCCMfeatures_dag.png',  bbox_inches='tight' , transparent=True )
    #######

    #compare networks - CCM and learned 
    dict_df_cuts[t+"_dag_global"] 
    ccm_dags_djmat = DAG_global['adjmat']*1
    
    IDX = confounders + ChemCols_0_10_names + taxa_groups + targetlist
    IDX = [i for i in IDX if i in ccm_dags_djmat.columns]
      
    ccm_dags_djmat = ccm_dags_djmat[IDX]
    ccm_dags_djmat = ccm_dags_djmat.reindex(IDX)

    IDX, label_groups = fix_labels(IDX)
    ccm_dags_djmat.columns = IDX
    ccm_dags_djmat.index = IDX
        
    plt.figure(figsize=(15, 15))
    sns.set(font_scale=1.25)
    g = sns.clustermap(ccm_dags_djmat, col_cluster=False, row_cluster=False,  linewidths=0.1, cmap='Blues', xticklabels=True, yticklabels=True,\
                       )#row_colors=row_colors, col_colors=col_colors)
    g.ax_row_dendrogram.set_visible(False)
    g.ax_col_dendrogram.set_visible(False)
    plt.savefig(BaseFolder+t+'_ccm_dag.png',  bbox_inches='tight' , transparent=True )
    #######
 
    
 
ccm_eccm = df_CausalFeatures2_untouched.pivot(index='x1', columns='x2', values='Score')
IDX = confounders + ChemCols_0_10_names + taxa_groups + targetlist
IDX = [i for i in IDX if i in ccm_eccm.index]

for i in IDX:
    if not i in ccm_eccm.columns:
        ccm_eccm[i] = np.nan
        
ccm_eccm = ccm_eccm[IDX]
ccm_eccm = ccm_eccm.reindex(IDX)

IDX, label_groups = fix_labels(IDX)
ccm_eccm.columns = IDX
ccm_eccm.index = IDX

plt.figure(figsize=(15, 15))
sns.set(font_scale=1.2)
g = sns.clustermap(ccm_eccm.fillna(0), col_cluster=False, row_cluster=False,  linewidths=0.1, cmap='Blues', xticklabels=True, yticklabels=True,\
                   )#row_colors=row_colors, col_colors=col_colors)
g.ax_row_dendrogram.set_visible(False)
plt.savefig(BaseFolder+'ccm_eccm.png',  bbox_inches='tight' , transparent=True )
#######


#Compare to correlations
df_cor = df_cut.astype(int).corr()
df_cor = df_cor.fillna(0)
df_cor = df_cor.round(2)

IDX = confounders + ChemCols_0_10_names + taxa_groups + targetlist
IDX = [i for i in IDX if i in df_cor.columns]

df_cor = df_cor[IDX]
df_cor = df_cor.reindex(IDX)

IDX, label_groups = fix_labels(IDX)
df_cor.columns = IDX
df_cor.index = IDX

plt.figure(figsize=(15, 15))
sns.set(font_scale=1.2)
g = sns.clustermap(df_cor, vmin=-1, vmax=1, col_cluster=False, row_cluster=False,  linewidths=0.1, cmap='coolwarm', xticklabels=True, yticklabels=True,\
                   )#row_colors=row_colors, col_colors=col_colors)
g.ax_row_dendrogram.set_visible(False)
g.ax_col_dendrogram.set_visible(False)
plt.savefig(BaseFolder+'corr.png',  bbox_inches='tight' , transparent=True )












