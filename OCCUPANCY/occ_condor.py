import os
import argparse
import numpy as np
import glob
import os.path
import time


def argParser():
    '''gestisce gli argomenti passati da linea di comando'''
    
    parser = argparse.ArgumentParser() 
    parser.add_argument('-i','--input', type=str, default='/lustre/cmswork/nlai/DATA/raw_data/', help="input directory")
    parser.add_argument('-o','--output', type=str, default='/lustre/cmswork/nlai/DATA/', help="output directory")
    parser.add_argument('-plot','--plot_path', type=str, default='/lustre/cmswork/nlai/PLOTS/OCCUPANCY/', help="output directory")
    parser.add_argument('-p','--pyscript', type=str, default = "occ_script.py",  help="name of python script to execute")

    
    return parser.parse_args()


def get_runs(path: str):
    '''legge i run number guardando i file di dati presenti nella cartella di input'''
    
    runs = sorted([int((name.split('RUN00')[1]).split('_')[0]) for name in os.listdir(path) if name.endswith('_data.txt')])

    return runs



def main(args):
    
    INPUT_PATH = args.input
    OUTPUT_PATH = args.output
    PLOT_PATH = args.plot_path
    
    runs = get_runs(INPUT_PATH)
    
#     os.system(f'mkdir {OUTPUT_PATH}')
    
    label = 'condor'+str(time.time())
    os.system(f'mkdir {label}')
    
    for run in runs:
        joblabel=str(run)
        if not os.path.isfile(f"{INPUT_PATH}/{joblabel}.txt"):
            # src file
            script_src = open(f"{label}/{joblabel}.src" , 'w')
            script_src.write("#!/bin/bash\n")
            script_src.write('eval "$(/lustre/cmswork/nlai/anaconda/bin/conda shell.bash hook)" \n')
            script_src.write(f"python {os.getcwd()}/{args.pyscript} -i {INPUT_PATH} -o {OUTPUT_PATH} -plot {PLOT_PATH} -run {run}") 
            script_src.close()
            os.system(f"chmod a+x {label}/{joblabel}.src") # THIS MAKES THE FILE EXECUTABLE
         
            # condor file
            script_condor = open(f"{label}/{joblabel}.condor", 'w')
            script_condor.write(f"executable = {label}/{joblabel}.src\n" )
            script_condor.write("universe = vanilla\n")
            script_condor.write(f"output = {label}/{joblabel}.out\n" )
            script_condor.write(f"error =  {label}/{joblabel}.err\n" )
            script_condor.write(f"log = {label}/{joblabel}.log\n")
            script_condor.write("+MaxRuntime = 500000\n")
            script_condor.write("queue\n")
            script_condor.close()
            # condor file submission
            os.system(f"condor_submit {label}/{joblabel}.condor") 
    
    return


if __name__ == "__main__":
    args = argParser()
    main(args)