import os
import argparse
import numpy as np
import glob
import os.path
import time


def argParser():
    '''gestisce gli argomenti passati da linea di comando'''
    
    parser = argparse.ArgumentParser() 
    parser.add_argument('-o','--output', type=str, default='/lustre/cmswork/nlai/DATA/raw_data/', help="output directory")
    parser.add_argument('-n','--nfiles', type=int, default=-1, help="number of files")
    parser.add_argument('-p','--pyscript', type=str, default = "download_data.py",  help="name of python script to execute")

    return parser.parse_args()


def main(args):
    
    OUTPUT_PATH = args.output

    runs = [1258, 1264, 1265, 1266]

    os.system(f'mkdir {OUTPUT_PATH}')

    label = 'condor'+str(time.time())
    os.system(f'mkdir {label}')

    for run in runs:
        joblabel=str(run)
        if not os.path.isfile(f"./{joblabel}.txt"):
            with open(f"{label}/{joblabel}.src" , 'w') as script_src:
                script_src.write("#!/bin/bash\n")
                script_src.write('eval "$(/lustre/cmswork/nlai/anaconda/bin/conda shell.bash hook)" \n')
                script_src.write(f"python {os.getcwd()}/{args.pyscript} -o {OUTPUT_PATH} -run {run} -n {args.nfiles} ")
            os.system(f"chmod a+x {label}/{joblabel}.src") # THIS MAKES THE FILE EXECUTABLE

            with open(f"{label}/{joblabel}.condor", 'w') as script_condor:
                script_condor.write(f"executable = {label}/{joblabel}.src\n" )
                script_condor.write("universe = vanilla\n")
                script_condor.write(f"output = {label}/{joblabel}.out\n" )
                script_condor.write(f"error =  {label}/{joblabel}.err\n" )
                script_condor.write(f"log = {label}/{joblabel}.log\n")
                script_condor.write("+MaxRuntime = 500000\n")
                script_condor.write("queue\n")
            # condor file submission
            os.system(f"condor_submit {label}/{joblabel}.condor") 

    return


if __name__ == "__main__":
    args = argParser()
    main(args)