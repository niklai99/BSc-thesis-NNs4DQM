from s3fs.core import S3FileSystem
import argparse
import pandas as pd


def argParser():
    '''gestisce gli argomenti passati da linea di comando'''
    
    parser = argparse.ArgumentParser() 
    parser.add_argument('-o','--output', type=str, default='/lustre/cmswork/nlai/DATA/', help="output directory")
    parser.add_argument('-run','--run', type=int, default=1220, help="run number")
    parser.add_argument('-n','--nfiles', type=int, default=-1, help="number of files")
    
    return parser.parse_args()



def makeOutputFile(output_path: str, run_number: int = 1220):
    '''costruisce il percorso ed il nome del file di output'''
    
    output_file = output_path + f'RUN00{run_number}_data.txt'
    
    return output_file



def readStream(run_number: int = 1220, n_files: int = -1):
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
    stream_df = pd.concat(
                            [
                                pd.read_csv(s3.open(f, mode='rb')) for f in s3.ls("/RUN00"+str(run_number)+"/")[:n_files]
                            ],
                            ignore_index=True
            )
    
    if n_files == -1:
        files_read = 'All'
    else:
        files_read = str(n_files)
    
    print(f'{files_read} data files collected')
    
    return stream_df



def cleanData(data: pd.DataFrame):
    '''pulisce i dati dalle informazioni che non mi servono per alleggerire il file'''
    
    # prendo solo i dati (label HEAD=2)
    mask_head = data['HEAD']==2
    
    # ignoro i canali "non fisici" oltre il 128 (compreso, è il trigger)
    mask_tdc = data['TDC_CHANNEL']<=128
    
    print('\nCleaning data...')
    data = data[mask_head & mask_tdc]
    print('Data cleaned successfully')
    
    return data



def saveData(data: pd.DataFrame, output_path: str, run_number: int = 1220):
    '''salva lo stream di dati su un file di testo'''
    
    print('\nSaving data to file...')
    data.to_csv(makeOutputFile(output_path, run_number), sep=' ', index=False, header=True)
    print('Saving completed')
    
    return



def main():
    
    args = argParser()
    
    RUNNUMBER = args.run
    OUT_PATH = args.output
    N_FILES = args.nfiles
    
    data = cleanData(readStream(RUNNUMBER, N_FILES))
    
    saveData(data, OUT_PATH, RUNNUMBER)
    
    print('\n\nExiting...\n\n')
    
    return


if __name__ == "__main__":
    main()