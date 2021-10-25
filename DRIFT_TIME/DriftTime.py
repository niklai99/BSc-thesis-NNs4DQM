import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
pd.options.mode.chained_assignment = None  # default='warn'


class DriftTime:
    '''gestisce il calcolo del tempo di deriva'''
    
    
    def __init__(self, run_number: int, input_path: str, output_path: str):
        '''numero identificativo della run, percorso del file di dati'''
        
        self.run_number = run_number
        self.input_path = input_path
        self.output_path = output_path
        
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
        

    
    def select_trigger(self, n: int = -1):
        '''prende solo il canale non fisico associato al trigger e ritorna le prime n righe'''
        
        self.trig = self.stream[(self.stream['TDC_CHANNEL']==128)].iloc[0:n]
        self.trig.reset_index(inplace=True, drop=True)
        
    
    
    def compute_dt(self):
        '''calcola il tempo di deriva, restituisce solamente il tempo di deriva'''
        
        def is_unique(s):
            '''funzione usata per controllare che in un array ci sia solo un unico valore'''
            a = s.to_numpy()
            return (a[0] == a).all()
        
        # lista per contenere i tempi di deriva 
        drift_time = []
        
        
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
#                 event['DRIFT_TIME'] = event['TIME'] - self.trig['TIME'].iloc[i]
                event['DRIFT_TIME'] = event['TIME'] - self.trig.loc[i, 'TIME']
                # aggiungo alla lista le drift time dell'evento i-esimo 
                drift_time.append(event['DRIFT_TIME'])
                
        self.drift_time = pd.concat(drift_time, ignore_index=True)
#         self.drift_time = np.concatenate(drift_time)
        print('Drift time computed')
    
        return self.drift_time
    
    
    
#     def compute_dt_dataframe(self):
#         '''calcola il tempo di deriva, restituisce il dataframe completo ed il tempo di deriva è una nuova colonna'''
        
#         def is_unique(s):
#             '''funzione usata per controllare che in un array ci sia solo un unico valore'''
#             a = s.to_numpy()
#             return (a[0] == a).all()
        
#         # lista per contenere i tempi di deriva 
#         self.drift_df = pd.DataFrame()
        
        
#         print('\nComputing drift time...')
#         # per ogni segnale misurato dallo scintillatore
#         for i, trig_orb_cnt in enumerate(self.trig['ORBIT_CNT']):
#             # chiamo 'event' il dataframe contentente tutti i dati con stesso ORBIT_CNT dello scintillatore i-esimo
#             # escludendo i segnali dello scintillatore (TDC_CHANNEL 128)
#             event = self.data[(self.data['ORBIT_CNT']==trig_orb_cnt)]

#             # IMPORTANTE: controlla che l'evento non sia vuoto
#             # se l'evento è vuoto vuol dire che non c'è alcun segnale che abbia ORBIT_CNT uguale al trigger
#             if event.empty:
#                 continue

#             # controllo che gli ORBIT_CNT nell'evento siano uguali e coincidano con quello dello scintillatore
#             if (is_unique(event['ORBIT_CNT'])) and (event['ORBIT_CNT'].iloc[0]==trig_orb_cnt):   
#                 # calcolo la drift time come (tempo dati - tempo scintillatore i-esimo)
#                 event['DRIFT_TIME'] = event['TIME'] - self.trig.loc[i, 'TIME']
#                 # aggiungo alla lista le drift time dell'evento i-esimo 
#                 self.drift_df = self.drift_df.append(event, ignore_index=True)

#         print('Drift time computed')
    
#         return self.drift_df
    
    
    
#     def select_ndata(self, ndata: int = -1):
#         '''selezione il numero di righe da analizzare'''
        
#         self.data = self.stream[:ndata]
        

    
#     def add_trigger_flag(self):
#         '''aggiunge un flag booleano (1 o 0) in base alla natura della riga'''
        
#         self.data = self.data.drop(self.data[(self.data['FPGA']==0) & (self.data['TDC_CHANNEL']==128)].index)
#         self.data['IS_TRIG'] = np.where((self.data['FPGA']==1) & (self.data['TDC_CHANNEL']==128), 1, 0)
        
        
    
#     def compute_dt_full(self):
#         '''calcola il tempo di deriva restituendo l intero dataframe raw di partenza'''
        
#         # dataframe finale
#         self.drift_df = pd.DataFrame()
        
#         group_data = self.data.groupby(by=['ORBIT_CNT'])
        
#         for orb, event in group_data:
#             # SE nel gruppo c'è un trigger value
#             if 1 in event['IS_TRIG'].values and event['IS_TRIG'].value_counts().loc[1]==1:
#                 # calcola la drift time 
#                 event['DRIFT_TIME'] = event['TIME'] - event[event['IS_TRIG']==1].iloc[0]['TIME']
#                 # rimpiazza lo 0 che viene fuori dal trigger con un NaN
#                 event['DRIFT_TIME'].replace(0, 1e10, inplace=True)
#             # SE nel gruppo non c'è il trigger value
#             else:
#                 # metti NaN come drift time
#                 event['DRIFT_TIME'] = 1e10
            
#             self.drift_df = pd.concat([self.drift_df, event], ignore_index=True)
# #             self.drift_df = self.drift_df.append(event, ignore_index=True)
            
#         return self.drift_df
    
    
    
    def cut_dt(self, df, left_bound: float, right_bound: float):
        '''elimina il rumore agli estremi della distribuzione'''
        
        # definisco due maschere per tagliare la distribuzione 
        left_mask = df > left_bound
        right_mask = df < right_bound
        
        # applico le maschere
        self.drift_time_cut = df[(left_mask & right_mask)]
#         df = df[(left_mask & right_mask)]
        
        return self.drift_time_cut # df
    
    
    
    def shift_dt(self, df, offset: float):
        '''shifta la distribuzione di un offset costante dovuto alla calibrazione del detector'''
        
#         shift della distribuzione di un offset costante
        self.drift_time_shifted = df + offset
#         df['DRIFT_TIME'] = df['DRIFT_TIME'] + offset
        
        return self.drift_time_shifted # df
        
       
    
    def save_dt(self, df, tag):
        '''salva i tempi di deriva su un file di testo'''
        
        self.ouput_file = self.output_path + f'RUN00{self.run_number}_{tag}.txt'
        
        print(f'Saving drift times to RUN00{self.run_number}_{tag}.txt ...')
        df.to_csv(self.ouput_file, sep=' ', index=False, header=True, na_rep='NaN')
        print('Saving completed')
    
    
    
    def make_distribution_plot(self, df, left_bound: float, right_bound: float, tag: str):
        '''crea e salva i grafici'''
        
        # creo figure&axes
        fig, ax = plt.subplots(figsize=(14,8))

        # istogramma del tempo di deriva
        ax = sns.histplot(x=df,  bins=100, 
                         stat='count', element='bars', fill=True, color='#009cff', edgecolor='white')
        
        # gestione del plot range
        ax.set_xlim(left_bound, right_bound)
        
        # imposto titolo e label degli assi
        ax.set_title(f'Drift Time Distribution - RUN00{self.run_number} - {tag}', fontsize = 18)
        ax.set_xlabel('Drift Time [ns]', fontsize = 16)
        ax.set_ylabel('Counts', fontsize = 16)
        
        # sistemo i ticks 
        ax.tick_params(axis = 'both', which = 'major', labelsize = 14, direction = 'out', length = 5)

        fig.tight_layout()
        plot_path = '/lustre/cmswork/nlai/PLOTS/DRIFT_TIME/drift_distributions/' 
        file_name = f'RUN00{self.run_number}_dt_distribution_{tag}'
        fig.savefig(plot_path+file_name, facecolor = 'white')
        
        
        
    def make_comparison_plot(self, df_raw, df_shifted, left_bound: float, right_bound: float, tag: str):
        '''crea e salva i grafici'''
        
        def change_legend(ax, new_loc, fontsize, titlesize, **kws):
            '''funzione per modificare posizione e font size della legenda generata da seaborn'''

            old_legend = ax.legend_
            handles = old_legend.legendHandles
            labels = [t.get_text() for t in old_legend.get_texts()]
            title = old_legend.get_title().get_text()

            ax.legend(handles, labels, loc=new_loc, title=title, fontsize=fontsize, title_fontsize=titlesize, frameon = True, fancybox = False, framealpha = 0.5, **kws)
            
        
        # creo figure&axes
        fig, ax = plt.subplots(figsize=(14,8))

        # istogramma del tempo di deriva
        ax = sns.histplot(x=df_raw,  bins=100, 
                         stat='count', element='bars', fill=True, color='#009cff', edgecolor='white',
                         label='raw drift times')
        ax = sns.histplot(x=df_shifted,  bins=100, 
                         stat='count', element='bars', fill=True, color='#FF6800', edgecolor='white',
                         label='shifted drift times')
        
        # gestione del plot range
        ax.set_xlim(left_bound, right_bound)
        
        # imposto titolo e label degli assi
        ax.set_title(f'Drift Time Distribution - RUN00{self.run_number} - {tag}', fontsize = 18)
        ax.set_xlabel('Drift Time [ns]', fontsize = 16)
        ax.set_ylabel('Counts', fontsize = 16)
        
        # sistemo i ticks 
        ax.tick_params(axis = 'both', which = 'major', labelsize = 14, direction = 'out', length = 5)

        ax.legend()
        change_legend(ax=ax, new_loc="upper right", fontsize=14, titlesize=16)

        fig.tight_layout()
        plot_path = '/lustre/cmswork/nlai/PLOTS/DRIFT_TIME/drift_distributions/' 
        file_name = f'RUN00{self.run_number}_dt_distribution_{tag}'
        fig.savefig(plot_path+file_name, facecolor = 'white')
        
        