#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jan  1 11:06:04 2023

@author: ofir
"""




import gseapy

import sys
import numpy as np
#import torch
import pandas as pd
from datetime import date
import pickle 
import numpy as np
import matplotlib.pyplot as plt

import seaborn as sns
import pyEDM
from multiprocessing import Pool

from scipy.stats import gaussian_kde

from tqdm import tqdm # for showing progress bar in for loops

import pandas as pd
import datetime
from datetime import timedelta
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import MinMaxScaler
import os
import glob
from matplotlib import pyplot

import itertools as it

from numpy import concatenate
from math import sqrt
from sklearn.metrics import mean_squared_error

import seaborn as sns
import matplotlib.pyplot as plt

from scipy.signal import argrelmax, argrelmin
from pymongo import MongoClient
import scipy
from datetime import date
import skccm as ccm
from skccm.utilities import train_test_split

#explore CCM

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
        #fig, axs = plt.subplots(figsize=(10, 10))
        
        #df_Scores.plot().get_figure().savefig(BaseFolder+"ccm_"+prefix+str(x1)+"_"+str(x2)+".png")
        #fig.savefig(BaseFolder+"ccm_"+prefix+str(x1)+"_"+str(x2)+".png")
        #plt.close()
        
        df_Scores = df_Scores.fillna(0)
        Score_X1 = df_Scores["x1"].values[-5:].mean()

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
        #print(str(len(ChemCols))+"/"+str(count))
        #here collect and store all x1's
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
                # Calculate the point density
    # =============================================================================
    #             try:
    #                 xy = np.vstack([ c["Library length"].values, c["x1"].values])
    #                 z = gaussian_kde(xy)(xy)
    #                 
    #                 fig, ax = plt.subplots()
    #                 ax.scatter(c["Library length"].values,  c["x1"].values, c=z, s=1)
    #                 ax.scatter(c_means["Library length"].values, c_means["x1_mean"].values, color="red", s=7)
    #                 ax.scatter(c_means["Library length"].values, c_means["x2_mean"].values, color="gray", s=7)
    #         
    #                 plt.savefig(BaseFolder+prefix_+"ccm_"+targetCol+"_"+col+".png")
    #             except:
    # =============================================================================
                    fig, ax = plt.subplots()
                    ax.scatter(c_means["Library length"].values, c_means["x1_mean"].values, color="red", s=3)
                    ax.scatter(c_means["Library length"].values, c_means["x2_mean"].values, color="gray", s=3)
                    plt.savefig(BaseFolder+prefix_+"ccm_"+dic[targetCol]+"_"+dic[col]+".png")               
                    plt.close()
            print(c_means)        
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
    
    #df_Scores.plot().get_figure().savefig("/home/ofir/Dropbox/Projects/Peridinium/results/ccm_"+prefix+str(d[x1])+"_"+str(d[x2])+".png")
    df_Scores = df_Scores.fillna(0)
    print(df_Scores)
    Score_X1 = df_Scores["x1"].values[-5:].mean()
    Score_X2 = df_Scores["x2"].values[-5:].mean()
    
    return Score_X1, Score_X2


BaseFolder = sys.argv[1]
prefix = sys.argv[2]


with open(BaseFolder + prefix + 'amplified_dfs.pickle', 'rb') as handle:
    amplified_dfs = pickle.load(handle)

with open(BaseFolder + prefix + 'DictCols.pickle', 'rb') as handle:
    DictCols = pickle.load(handle)
    
with open(BaseFolder + prefix +'x1_x2_columns.pickle', 'rb') as handle:
    cols_x1, cols_x2 = pickle.load(handle)





def manipulate(v):
    All_causal_CCM_dfs = []
    for j, valj in enumerate(cols_x2):
            All_causal_CCM_dfs.append(fullCCM(dfsList=amplified_dfs,
                        col=v,
                        targetCol=valj,
                        dic=DictCols,
                        prefix_=prefix,
                        #lag=0,
                        showFig = False))
            
            print(str(valj)+" : "+str(v))    
    return All_causal_CCM_dfs    




pool = Pool(46)   

results_list_final = []
results_list_final = pool.map(manipulate,cols_x1)

results_list_fixed = []
for i in results_list_final:
    results_list_fixed = results_list_fixed + i

pool.close()
pool.join()
print('end')
    
with open(BaseFolder + 'All_' + prefix + 'results.pickle', 'wb') as handle:
    pickle.dump(results_list_fixed, handle, protocol=pickle.HIGHEST_PROTOCOL)      
    
    





























