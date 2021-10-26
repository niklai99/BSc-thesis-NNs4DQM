import numpy as np
import pandas as pd
import scipy 

    
class DataCreator():
    '''genera le distribuzioni di dati'''
    

    def make_distribution(self, n_background: int, n_signal: int, loc: float = 0.8, scale: float = 0.02):
        '''crea la distribuzione e ritorna il dataframe associato'''
        
        # imposto il numero di dati da generare
        n_sig = n_signal
        n_bkg = n_background
        
        # genero le distribuzioni
        bkg_distribution = scipy.stats.expon.rvs(loc=0, scale=0.125, size=n_bkg)
        sig_distribution = scipy.stats.norm.rvs(loc=loc, scale=scale, size=n_sig)
        
        data_distribution = np.concatenate((np.array(bkg_distribution), np.array(sig_distribution)))
        np.random.shuffle(data_distribution)
        
        self.data_df = pd.DataFrame({'feature0':data_distribution})
        
        return self.data_df
    

    def save_distribution(self, fname: str):
        '''salva la distribuzione in un file di testo'''
        
        file_path = '/lustre/cmswork/nlai/NPL_1D/distributions_data/' + str(fname)
        
        self.data_df.to_csv(file_path, index=False)
        
        
