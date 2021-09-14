import argparse
from ReadStream import StreamReader


def argParser():
    '''gestisce gli argomenti passati da linea di comando'''
    
    parser = argparse.ArgumentParser() 
    parser.add_argument('-o','--output', type=str, default='/lustre/cmswork/nlai/DATA/raw_data/', help="output directory")
    parser.add_argument('-run','--run', type=int, default=1220, help="run number")
    parser.add_argument('-n','--nfiles', type=int, default=-1, help="number of files")
    
    return parser.parse_args()


def main(args):

    RUNNUMBER = args.run
    OUT_PATH = args.output
    N_FILES = args.nfiles
    
    reader = StreamReader(RUNNUMBER, OUT_PATH, N_FILES)
    
    reader.readStream()
    reader.cleanData()
    reader.saveData()
    
    print('\n\nExiting...\n\n')
    
    return


if __name__ == "__main__":
    args = argParser()
    main(args)