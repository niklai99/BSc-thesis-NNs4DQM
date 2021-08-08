import numpy as np
import os
import h5py
import pandas as pd


def BuildSample_DY_Higgs(FILE_NAME, seed, nfiles=1):
    
    np.random.seed(seed)
    higgs_file = h5py.File(FILE_NAME,'r')
    dfH=pd.DataFrame({key:np.array(higgs_file.get(key)) for key in higgs_file.keys()})#,dtype=np.float32
    higgs_file.close()
    dfH=dfH.sample(frac=1)
    print('Building pandas!')
    mask_pt=(dfH['pt1']>20) & (dfH['pt2']>10) & (dfH['pt3']>10)
    return dfH[mask_pt]


def BuildSample_DY_val(N_Events, INPUT_PATH, seed, nfiles=20):
    #random integer to select Zprime file between n files         
    np.random.seed(seed)

    u = np.arange(nfiles)#np.random.randint(100, size=100)                                                                                                                                                            
    np.random.shuffle(u)

    #BACKGROUND                                                                                                                                                                                                       
    #extract N_Events from files                                                                                                                                                                                      
    toy_label = INPUT_PATH.split("/")[-2]
    print('File label: ', toy_label)

    HLF = np.array([])
    HLF_val = np.array([])
    split_happened=False
    
    for u_i in u:
        f = h5py.File(INPUT_PATH+toy_label+str(u_i+1)+".h5",'r')
        keys=list(f.keys())
        #check whether the file is empty                                                                                                                                                                              
        if len(keys)==0:
            continue
        cols=np.array([])
        for i in range(len(keys)):
            feature = np.array(f.get(keys[i]))
            feature = np.expand_dims(feature, axis=1)
            if i==0:
                cols = feature
            else:
                cols = np.concatenate((cols, feature), axis=1)
        print(f'Shape of file N. {u_i}',cols.shape)
        np.random.shuffle(cols) #don't want to select always the same event first                      
        
        if HLF.shape[0]==0:
            HLF=cols
            i=i+1
        else:
            HLF=np.concatenate((HLF, cols), axis=0)
        f.close()
        #print(HLF_REF.shape)                                                                                                                                                                                         
        if HLF.shape[0]>=2*N_Events:
            HLF_val=HLF[N_Events:2*N_Events, :]
            HLF=HLF[:N_Events, :]
            split_happened = True
            break  
            
    if split_happened ==False:
        raise ValueError('Not enough data to fill reference and validation! Exiting...')
    
    print("HLF shape:", HLF.shape)
    print("HLF_val shape:", HLF_val.shape)
    #print(HLF[:100,:])
    # feature order: pt1, pt2, eta1, eta2, delta_phi, mass
    if HLF.shape[1]==6:
        return HLF[:, [4, 5, 1, 2, 0, 3]], HLF_val[:, [4, 5, 1, 2, 0, 3]] 
    elif HLF.shape[1]==1:
        return HLF[:,:], HLF_val[:,:]
    
    
    
def BuildSample_DY(N_Events, INPUT_PATH, seed, nfiles=20):
    #random integer to select Zprime file between n files                                                                                          
    np.random.seed(seed)

    u = np.arange(nfiles)#np.random.randint(100, size=100)                                                                                                                                                            
    np.random.shuffle(u)

    #BACKGROUND                                                                                                                                                                                                       
    #extract N_Events from files                                                                                                                                                                                      
    toy_label = INPUT_PATH.split("/")[-2]
    print(toy_label)

    HLF = np.array([])
    enough_data = False
        
    for u_i in u:
        f = h5py.File(INPUT_PATH+toy_label+str(u_i+1)+".h5",'r')
        keys=list(f.keys())
        #check whether the file is empty                                                                                                                                                                              
        if len(keys)==0:
            continue
        cols=np.array([])
        for i in range(len(keys)):
            feature = np.array(f.get(keys[i]))
            feature = np.expand_dims(feature, axis=1)
            if i==0:
                cols = feature
            else:
                cols = np.concatenate((cols, feature), axis=1)
        print(f'Shape of file N. {u_i}',cols.shape)
        np.random.shuffle(cols) #don't want to select always the same event first                      

        if HLF.shape[0]==0:
            HLF=cols
            i=i+1
        else:
            HLF=np.concatenate((HLF, cols), axis=0)
        f.close()
        #print(HLF_REF.shape)                                                                                                                                                                                         
        if HLF.shape[0]>=N_Events:
            HLF=HLF[:N_Events, :]
            enough_data = True
            break   
    
    if enough_data == False:
        raise ValueError('Not enough data to fill reference and validation! Exiting...')
    print("HLF shape:", HLF.shape)
    #print(HLF[:100,:])
    # feature order: pt1, pt2, eta1, eta2, delta_phi, mass
    if HLF.shape[1]==6:
        return HLF[:, [4, 5, 1, 2, 0, 3]]
    elif HLF.shape[1]==1:
        return HLF[:,:]
    