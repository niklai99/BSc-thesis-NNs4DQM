import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


class Occupancy:
    '''gestisce dell occupanza dei canali'''
    
    
    def __init__(self, run_number: int, input_path: str, output_path: str, plot_path: str):
        '''numero identificativo della run, percorso del file di dati'''
        
        self.run_number = run_number
        self.input_path = input_path
        self.output_path = output_path
        self.plot_path = plot_path
        
        # genero il nome del file di input
        self.input_file = self.input_path + f'RUN00{self.run_number}_data.txt'
        
    
    def read_data(self):
        '''legge i dati dal file'''
        
        print(f'\nReading data from RUN00{self.run_number}_data.txt ... ')
        
        # read data from file
        self.stream = pd.read_csv(self.input_file, sep=' ')
        
        print('Data imported successfully')
 
        
        
    def compute_time(self):
        '''aggiunge una colonna con i tempi in nanosecondi'''
        
        print('\nConverting time features in nanoseconds...')
        self.stream['TIME'] = self.stream['ORBIT_CNT']*3564*25 + self.stream['BX_COUNTER']*25 + self.stream['TDC_MEAS']*25/30
        print('Conversion completed')
       
    
    
    def select_data(self):
        '''prende solo i canali fisici delle fpga e ritorna il dataframe'''
        
        print('\nSeparating data rows from trigger rows...')
        self.data = self.stream[(self.stream['TDC_CHANNEL']!=128)]
        self.data.reset_index(inplace=True, drop=True)
        
        return self.data

       
    
    def compute_run_time(self):
        '''calcola quanti secondi è durata la run'''
     
        self.run_time = (self.data['TIME'].max() - self.data['TIME'].min()) * 1e-9
        
        return self.run_time
    
    
    
    def split_fpga(self):
        '''divide i dati in base alla fpga che li ha acquisiti'''
    
        self.data_fpga0 = self.data[(self.data['FPGA']==0)]
        self.data_fpga1 = self.data[(self.data['FPGA']==1)]
        
        return self.data_fpga0, self.data_fpga1
    
    
    
    def make_histogram(self, df):
        '''crea l istogramma usando numpy in quanto ci può servire l altezza dei bin'''
       
        hist, _ = np.histogram(df['TDC_CHANNEL'], bins=np.arange(129))
        
        return hist
    
    
    
    def make_rate_histogram(self, hist, run_time):
        '''crea l istogramma della rate'''
        
        rate_hist = hist / run_time
        
        return rate_hist
    
    
    
    def save_hist(self, hist, distr_type, tag):
        '''salva istogramma su un file di testo'''
        
        if distr_type=='occ':
            folder='occupancy/'
        elif distr_type=='rate': 
            folder='rate/'
            
        self.ouput_file = self.output_path + folder + f'RUN00{self.run_number}_{tag}.txt'
        
        print(f'Saving histogram to RUN00{self.run_number}_{distr_type}_{tag}.txt ...')
        np.savetxt(self.ouput_file, hist)
        print('Saving completed')
        
        
          
    def make_distribution_plot(self, hist, distr_type, tag):
        '''crea e salva i grafici'''
        
        # creo figure&axes
        fig, ax = plt.subplots(figsize=(14,8))

        # istogramma del tempo di deriva
        ax = sns.histplot(x=np.arange(128),  bins=np.arange(129), weights=hist,
                         stat='count', element='bars', fill=True, color='#009cff', edgecolor='white')
        
        # gestione del plot range
        ax.set_xlim(0, 128)
        
        if distr_type=='occ':
            title='Occupancy'
            ylabel='Counts'
        elif distr_type=='rate': 
            ylabel='Rate [Hz]'
            title='Rate'
        
        # imposto titolo e label degli assi
        ax.set_title(f'Channel {title} - RUN00{self.run_number}', fontsize = 18)
        ax.set_xlabel('Channel', fontsize = 16)
        ax.set_ylabel(ylabel, fontsize = 16)
        
        # sistemo i ticks 
        ax.tick_params(axis = 'both', which = 'major', labelsize = 14, direction = 'out', length = 5)

        fig.tight_layout()
        
        if distr_type=='occ':
            folder='occupancy_distribution/'
        elif distr_type=='rate': 
            folder='rate_distribution/'
            
        file_name = f'RUN00{self.run_number}_{distr_type}_{tag}'
        fig.savefig(self.plot_path+folder+file_name, facecolor = 'white')
        
        
        
    def make_comparison_plot(self, hist_0, hist_1, distr_type):
        '''crea e salva i grafici'''
        
        def change_legend(ax, new_loc, fontsize, titlesize, **kws):
            '''funzione per modificare posizione e font size della legenda generata da seaborn'''

            old_legend = ax.legend_
            handles = old_legend.legendHandles
            labels = [t.get_text() for t in old_legend.get_texts()]
            title = old_legend.get_title().get_text()

            ax.legend(handles, labels, loc=new_loc, title=title, fontsize=fontsize, title_fontsize=titlesize, frameon = True, fancybox = False, framealpha = 0.5, **kws)

            return
        
        # creo figure&axes
        fig, ax = plt.subplots(figsize=(14,8))

        # istogramma del tempo di deriva
        ax = sns.histplot(x=np.arange(128),  bins=np.arange(129), weights=hist_0,
                         stat='count', element='bars', fill=True, color='#009cff', edgecolor='white',
                         label='fpga0')
        ax = sns.histplot(x=np.arange(128),  bins=np.arange(129), weights=hist_1,
                         stat='count', element='bars', fill=True, color='#FF6300', edgecolor='white',
                         label='fpga1')
        
        # gestione del plot range
        ax.set_xlim(0, 128)
        
       
        if distr_type=='occ':
            title='Occupancy'
            ylabel='Counts'
        elif distr_type=='rate': 
            ylabel='Rate [Hz]'
            title='Rate'
        
        # imposto titolo e label degli assi
        ax.set_title(f'Channel {title} - RUN00{self.run_number}', fontsize = 18)
        ax.set_xlabel('Channel', fontsize = 16)
        ax.set_ylabel(ylabel, fontsize = 16)
        
        # sistemo i ticks 
        ax.tick_params(axis = 'both', which = 'major', labelsize = 14, direction = 'out', length = 5)

        ax.legend()
        change_legend(ax=ax, new_loc="upper right", fontsize=14, titlesize=16)

        fig.tight_layout()
        
        if distr_type=='occ':
            folder='occupancy_distribution/'
        elif distr_type=='rate': 
            folder='rate_distribution/'
            
        file_name = f'RUN00{self.run_number}_{distr_type}'
        fig.savefig(self.plot_path+folder+file_name, facecolor = 'white')
        
        