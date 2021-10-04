import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
pd.options.mode.chained_assignment = None  # default='warn'


class Occupancy:
    '''gestisce dell occupanza dei canali'''
    
    
    def __init__(self, run_number: int, input_path: str, file_tag: str, output_path: str, plot_path: str):
        '''numero identificativo della run, percorso del file di dati'''
        
        self.run_number = run_number
        self.input_path = input_path
        self.file_tag = file_tag
        self.output_path = output_path
        self.plot_path = plot_path
        
        # genero il nome del file di input
        self.input_file = self.input_path + f'RUN00{self.run_number}_{self.file_tag}.txt' # _data or _cut_shifted_hstat_condor
        
    
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
    
    
    
    def select_trigger(self, n: int = -1):
        '''prende solo il canale non fisico associato al trigger e ritorna le prime n righe'''
        
        self.trig = self.stream[(self.stream['TDC_CHANNEL']==128)].iloc[0:n]
        self.trig.reset_index(inplace=True, drop=True)
        
        return self.trig
        
        
        
    def compute_dt_dataframe(self):
        '''calcola il tempo di deriva, restituisce il dataframe completo ed il tempo di deriva è una nuova colonna'''
        
        def is_unique(s):
            '''funzione usata per controllare che in un array ci sia solo un unico valore'''
            a = s.to_numpy()
            return (a[0] == a).all()
        
        # lista per contenere i tempi di deriva 
        self.drift_df = pd.DataFrame()
        
        
        print('\nComputing drift time...')
        # per ogni segnale misurato dallo scintillatore
        for i, trig_orb_cnt in enumerate(self.trig['ORBIT_CNT']):
            # chiamo 'event' il dataframe contentente tutti i dati con stesso ORBIT_CNT dello scintillatore i-esimo
            # escludendo i segnali dello scintillatore (TDC_CHANNEL 128)
            event = self.data[(self.data['ORBIT_CNT']==trig_orb_cnt)]

            # IMPORTANTE: controlla che l'evento non sia vuoto
            # se l'evento è vuoto vuol dire che non c'è alcun segnale che abbia ORBIT_CNT uguale al trigger
            if event.empty:
                continue

            # controllo che gli ORBIT_CNT nell'evento siano uguali e coincidano con quello dello scintillatore
            if (is_unique(event['ORBIT_CNT'])) and (event['ORBIT_CNT'].iloc[0]==trig_orb_cnt):   
                # calcolo la drift time come (tempo dati - tempo scintillatore i-esimo)
                event['DRIFT_TIME'] = event['TIME'] - self.trig.loc[i, 'TIME']
                # aggiungo alla lista le drift time dell'evento i-esimo 
                self.drift_df = self.drift_df.append(event, ignore_index=True)

        print('Drift time computed')
    
        return self.drift_df
    
    
    def cut_dt(self, df, left_bound: float, right_bound: float):
        '''elimina il rumore agli estremi della distribuzione'''
        
        # definisco due maschere per tagliare la distribuzione 
        left_mask = df['DRIFT_TIME'] > left_bound
        right_mask = df['DRIFT_TIME'] < right_bound
        
        # applico le maschere
#         self.drift_time_cut = df[(left_mask & right_mask)]
        df = df[(left_mask & right_mask)]
        
        return df # self.drift_time_cut
    
    
    
    def shift_dt(self, df, offset: float):
        '''shifta la distribuzione di un offset costante dovuto alla calibrazione del detector'''
        
        # shift della distribuzione di un offset costante
#         self.drift_time_shifted = df + offset
        df['DRIFT_TIME'] = df['DRIFT_TIME'] + offset
        
        return df # self.drift_time_shifted

       
    
    def compute_run_time(self, df):
        '''calcola quanti secondi è durata la run'''
     
        self.run_time = (df['TIME'].max() - df['TIME'].min()) * 1e-9
        
        return self.run_time
    
    
    
    def split_fpga(self, df):
        '''divide i dati in base alla fpga che li ha acquisiti'''
    
        self.data_fpga0 = df[(df['FPGA']==0)]
        self.data_fpga1 = df[(df['FPGA']==1)]
        
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
        
        