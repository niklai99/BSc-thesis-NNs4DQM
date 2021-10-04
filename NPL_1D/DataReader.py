import pandas as pd


class DataReader:
    '''gestisce la lettura dei dati'''
    
    def build_sample(self, file_name, n_data):
        '''legge i dati da un file e restituisce un dataframe dei dati'''
        
        path = '/lustre/cmswork/nlai/NPL_1D/distributions_data/'
        
        # legge i dati dal file
        df=pd.read_csv(path+file_name)
        
        # random sampling dei dati
        df=df.sample(n=n_data)
        
        return df


def BuildSample(file_name, n_data):
    '''legge i dati da un file e restituisce un dataframe dei dati'''
    
    path = '/lustre/cmswork/nlai/NPL_1D/distributions_data/'
    
    # legge i dati dal file
    df=pd.read_csv(path+file_name)
    
    # random sampling dei dati
    df=df.sample(n=n_data)
    
    return df