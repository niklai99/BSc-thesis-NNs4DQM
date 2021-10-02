import pandas as pd
import numpy as np


class DataReader:
    '''gestisce la lettura dei dati'''
    
    def build_sample(self, file_name, n_data):
        '''legge i dati da un file e restituisce un dataframe dei dati'''
        
        path = '/lustre/cmswork/nlai/DATA/drift_distributions/'
        
        # legge i dati dal file
        df=pd.read_csv(path+file_name, sep=' ')
        
        # random sampling dei dati
        df=df.sample(n=n_data)
        
        return df['DRIFT_TIME'] # .to_numpy()


def BuildSample(file_name, n_data):
    '''legge i dati da un file e restituisce un dataframe dei dati'''
    
    path = '/lustre/cmswork/nlai/DATA/drift_distributions/'
    
    # legge i dati dal file
    df=pd.read_csv(path+file_name, sep=' ')
    
    # random sampling dei dati
    df=df.sample(n=n_data)
    
    return df['DRIFT_TIME']