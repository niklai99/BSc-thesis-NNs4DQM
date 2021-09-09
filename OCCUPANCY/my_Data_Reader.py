import numpy as np
import os
import h5py
import pandas as pd
import scipy

def BuildSample(FILE_NAME, n):
     
    df=pd.read_csv(FILE_NAME)
    df=df.sample(n=n)
    
    return df


def BuildDistribution(nbkg, nsig):
     
    bkg = scipy.stats.expon.rvs(loc=0,scale=0.125, size=nbkg)
    sig = scipy.stats.norm.rvs(loc=0.8,scale=0.02, size=nsig)
    data = np.concatenate((np.array(bkg), np.array(sig)))
    np.random.shuffle(data)
    df = pd.DataFrame({'feature0':data})

    return df







# def BuildSample_DY(N_Events, INPUT_PATH, seed, nfiles=20):
#     #random integer to select Zprime file between n files                                                                                                                                                             
#     u = np.arange(nfiles)#np.random.randint(100, size=100)                                                                                                                                                            
#     np.random.shuffle(u)

#     #BACKGROUND                                                                                                                                                                                                       
#     #extract N_Events from files                                                                                                                                                                                      
#     toy_label = INPUT_PATH.split("/")[-2]
#     print(toy_label)

#     HLF = np.array([])

#     for u_i in u:
#         f = h5py.File(INPUT_PATH+toy_label+str(u_i+1)+".h5",'r')
#         keys=list(f.keys())
#         #check whether the file is empty                                                                                                                                                                              
#         if len(keys)==0:
#             continue
#         cols=np.array([])
#         for i in range(len(keys)):
#             feature = np.array(f.get(keys[i]))
#             feature = np.expand_dims(feature, axis=1)
#             if i==0:
#                 cols = feature
#             else:
#                 cols = np.concatenate((cols, feature), axis=1)
#         print(cols.shape)
#         np.random.shuffle(cols) #don't want to select always the same event first                                                                                                                                     
        
#         if HLF.shape[0]==0:
#             HLF=cols
#             i=i+1
#         else:
#             HLF=np.concatenate((HLF, cols), axis=0)
#         f.close()
#         #print(HLF_REF.shape)                                                                                                                                                                                         
#         if HLF.shape[0]>=N_Events:
#             HLF=HLF[:N_Events, :]
#             break                                                                                                                                                                                               
#     print("HLF shape:", HLF.shape)
#     #print(HLF[:100,:])
#     # feature order: pt1, pt2, eta1, eta2, delta_phi, mass
#     #return HLF[:, [4, 5, 1, 2, 0, 3]]
#     return HLF[:,:]