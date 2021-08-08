import os
import argparse
import numpy as np
import glob
import os.path
import time

if __name__ == '__main__':

    parser = argparse.ArgumentParser()    #Python tool that makes it easy to create an user-friendly command-line interface
    parser.add_argument('-o','--output', type=str, help="output  EOS directory", required=True)
#     parser.add_argument('-i','--input', type=str, help="input  EOS directory", required=True)
    parser.add_argument('-p','--pyscript', type=str,default = "NPL_train_1D.py",  help="name of python script to execute")
    # parser.add_argument('-q', '--queue', type=str, default = "1nw", help="LSFBATCH queue name")
#     parser.add_argument('-f', '--feature', type=str, default = "0", help="feature of the input datset to be analyzed")
    parser.add_argument('-t','--toys', type=str, default=500, help="number of repetitions", required=False)
    parser.add_argument('-sig','--signal', type=int, default=0, help="number of signal events", required=False)
    parser.add_argument('-bkg','--background', type=int, default=1000, help="number of background events", required=False)
    parser.add_argument('-ref','--reference', type=int, default=500000, help="number of reference events", required=False)
    parser.add_argument('-epochs','--epochs', type=int, default=200000, help="number of epochs", required=False)
    parser.add_argument('-latsize','--latsize', type=int, default=5, help="number of nodes in each hidden layer", required=False)
    parser.add_argument('-layers','--layers', type=int, default=3, help="number of layers", required=False)
    parser.add_argument('-wclip','--weight_clipping', type=float, default=2.15, help="weight clipping", required=False)
    args = parser.parse_args()
    
    #folder to save the outputs of the pyscript
    mydir = args.output#+"/"
    os.system(f"mkdir {mydir}")
    
    #folder to save the outputs of each condor job (file.out, file.log, file.err)
    label = 'condor_'+args.output.split("/")[-1]+str(time.time())
    os.system(f"mkdir {label}")

    for i in range(int(args.toys)):
        joblabel = str(i)#fileIN.split("/")[-1].replace(".h5","")                                                                                                                      
        if not os.path.isfile(f"{mydir}/{joblabel}.txt"):
            # src file
            script_src = open(f"{label}/{joblabel}.src" , 'w')
            script_src.write("#!/bin/bash\n")
            #script_src.write("source /cvmfs/sft.cern.ch/lcg/views/LCG_92//x86_64-slc6-gcc62-opt/setup.sh\n")
            script_src.write('eval "$(/lustre/cmswork/nlai/anaconda/bin/conda shell.bash hook)" \n')
            script_src.write(f"python {os.getcwd()}/{args.pyscript} -o {mydir}/{joblabel}") #/{joblabel}
#             script_src.write(f"python {os.getcwd()}/{args.pyscript} -o {mydir}/{joblabel} -i {args.input} -t {int(args.toys)} -sig {args.signal} -bkg {args.background} -ref {args.reference} -epochs {args.epochs} -latsize {args.latsize} -layers {args.layers} -wclip {args.weight_clipping}") #/{joblabel}
            script_src.close()
            os.system(f"chmod a+x {label}/{joblabel}.src") #THIS MAKES THE FILE EXECUTABLE
            #os.system("bsub -q %s -o %s/%s.log -J %s_%s < %s/%s.src" %(args.queue, label, joblabel, label, joblabel, label, joblabel))
            
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