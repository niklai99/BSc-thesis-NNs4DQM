import os
import argparse
import numpy as np
import glob
import os.path
import time

if __name__ == '__main__':

    parser = argparse.ArgumentParser()    #Python tool that makes it easy to create an user-friendly command-line interface
    parser.add_argument('-o','--output', type=str, help="output  EOS directory", required=True)
    parser.add_argument('-i','--input', type=str, help="input  EOS directory", required=True)
    parser.add_argument('-p','--pyscript', type=str,default = "TOY2.py",  help="name of python script to execute")
    parser.add_argument('-q', '--queue', type=str, default = "1nw", help="LSFBATCH queue name")
#    parser.add_argument('-f', '--feature', type=str, default = "0", help="feature of the input daatset to be analyzed")
    parser.add_argument('-t', '--toys', type=int, default = "100", help="number of toys to be processed")
    args = parser.parse_args()
    
    #folder to save the outputs of the pyscript
    mydir = args.output#+"/"
    os.system(f"mkdir {mydir}")
    
    #folder to save the outputs of each condor job (file.out, file.log, file.err)
    label = 'condor_'+args.output.split("/")[-1]+'_5D_'+str(time.time())
    os.system(f"mkdir {label}")

    for i in range(int(args.toys)):
        joblabel = str(i)#fileIN.split("/")[-1].replace(".h5","")                                                                                                                      
        if not os.path.isfile(f"{mydir}/{joblabel}.txt"):
            # src file
            script_src = open(f"{label}/{joblabel}.src" , 'w')
            script_src.write("#!/bin/bash\n")
            #script_src.write("source /cvmfs/sft.cern.ch/lcg/views/LCG_92//x86_64-slc6-gcc62-opt/setup.sh\n")
            script_src.write('eval "$(/lustre/cmswork/dalsanto/anaconda3/bin/conda shell.bash hook)" \n') # CAMBIARE - DOVE TROVARE ANACONDA
            script_src.write('conda activate tf_env \n') # CAMBIARE ENV
            script_src.write(f"python {os.getcwd()}/{args.pyscript} -o {mydir}/{joblabel} -i {args.input} -t {int(args.toys)}") #/{joblabel}
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