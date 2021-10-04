import os
import argparse
import time

def argParser():
    '''gestisce gli argomenti passati da linea di comando'''
    
    parser = argparse.ArgumentParser()    #Python tool that makes it easy to create an user-friendly command-line interface
    parser.add_argument('-o','--output', type=str, help="output  EOS directory", required=True)
    # parser.add_argument('-i','--input', type=str, help="input  EOS directory", required=True)
    parser.add_argument('-p','--pyscript', type=str,default = "NPL_Train.py",  help="name of python script to execute")
    parser.add_argument('-t','--toys', type=str, default=150, help="number of repetitions", required=False)
    parser.add_argument('-sig','--signal', type=int, default=0, help="number of signal events", required=False)
    parser.add_argument('-bkg','--background', type=int, default=10000, help="number of background events", required=False)
    parser.add_argument('-ref','--reference', type=int, default=200000, help="number of reference events", required=False)
    parser.add_argument('-epochs','--epochs', type=int, default=200000, help="number of epochs", required=False)
    parser.add_argument('-latsize','--latsize', type=int, default=3, help="number of nodes in each hidden layer", required=False)
    parser.add_argument('-layers','--layers', type=int, default=1, help="number of layers", required=False)
    parser.add_argument('-wclip','--weight_clipping', type=float, default=7, help="weight clipping", required=False)
    parser.add_argument('-patience','--patience', type=int, default=1000, help="number of epochs between two consecutives saving points", required=False)
    
    return parser.parse_args()


def main(args):
    
    OUTPUT_PATH = str(args.output)
    
    N_TOYS = int(args.toys)

    os.system(f'mkdir {OUTPUT_PATH}')
    
    # folder to save the outputs of each condor job (file.out, file.log, file.err)
    label = 'condor_' + OUTPUT_PATH.split("/")[-1] + str(time.time())
    os.system(f'mkdir {label}')
    
    for i in range(N_TOYS):
        joblabel = str(i)                                                                                                                   
        if not os.path.isfile(f"{OUTPUT_PATH}/{joblabel}.txt"):
            # src file
            script_src = open(f"{label}/{joblabel}.src" , 'w')
            script_src.write("#!/bin/bash\n")
            script_src.write('eval "$(/lustre/cmswork/nlai/anaconda/bin/conda shell.bash hook)" \n')
            script_src.write(
                                f"python {os.getcwd()}/{args.pyscript} -o {OUTPUT_PATH}/{joblabel} \
                                -t {args.toys} -sig {args.signal} -bkg {args.background} -ref {args.reference} \
                                -epochs {args.epochs} -latsize {args.latsize} -layers {args.layers} \
                                -wclip {args.weight_clipping} -patience {args.patience}"
                            )
            

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