from s3fs.core import S3FileSystem
import pandas as pd


class StreamReader:
    '''gestisce la lettura dello stream di dati su CloudVeneto'''
    
    
    def __init__(self, run_number: int, output_path: str, n_files: int):
        '''numero identificativo della run, percorso di output, numero di file da analizzare'''
        
        self.run_number = run_number
        self.output_path = output_path
        self.n_files =n_files
        
        # genero il nome del file di output
        self.output_file = self.output_path + f'RUN00{self.run_number}_data.txt'
        
        
    
    def readStream(self):
        '''legge i dati da cloud veneto'''

        # accesso a CVeneto
        print('\nConnecting to CloudVeneto...')
        try:
            s3 = S3FileSystem(
                                anon=False,
                                key="69a290784f914f67afa14a5b4cadce79",
                                secret="2357420fac4f47d5b41d7cdeb52eb184",
                                client_kwargs={'endpoint_url': 'https://cloud-areapd.pd.infn.it:5210', "verify":False}
                )
        except:
            print('\n\nERROR:')
            print('Unable to establish connection with CloudVeneto')
        else:
            print('Connection with CloudVeneto established correctly')


        # lettura dei file da CloudVeneto
        print('\nReading data files...')
        self.stream_df = pd.concat(
            [
                pd.read_csv(s3.open(f, mode='rb'))
                for f in s3.ls("/RUN00" + str(self.run_number) + "/")[
                    : self.n_files
                ]
                if f.endswith('.txt')
            ],
            ignore_index=True,
        )


        # feedback dei file di dati concatenati
        files_read = 'All' if self.n_files == -1 else str(self.n_files)
        print(f'{files_read} data files collected')


        
    def cleanData(self):
        '''pulisce i dati dalle informazioni che non mi servono per alleggerire il file'''

        # prendo solo i dati (label HEAD=2)
        mask_head = self.stream_df['HEAD']==2

        # ignoro i canali "non fisici" oltre il 128 (compreso, è il trigger)
        mask_tdc = self.stream_df['TDC_CHANNEL']<=128

        print('\nCleaning data...')
        self.stream_df = self.stream_df[mask_head & mask_tdc]
        print('Data cleaned successfully')

    
    
    def saveData(self):
        '''salva lo stream di dati su un file di testo'''

        print('\nSaving data to file...')
        self.stream_df.to_csv(self.output_file, sep=' ', index=False, header=True)
        print('Saving completed')

        return