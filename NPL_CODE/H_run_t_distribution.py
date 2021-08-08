import os
import argparse
import numpy as np
import glob
import os.path
import time

if __name__ == '__main__':

    parser = argparse.ArgumentParser()    #Python tool that makes it easy to create an user-friendly command-line interface
    parser.add_argument('-o','--output', type=str, help="output  EOS directory", required=True)
    #parser.add_argument('-i','--input', type=str, help="input  EOS directory", required=True)
    parser.add_argument('-p','--pyscript', type=str,default = "TOY2.py",  help="name of python script to execute")
    #parser.add_argument('-q', '--queue', type=str, default = "1nw", help="LSFBATCH queue name")
#    parser.add_argument('-f', '--feature', type=str, default = "0", help="feature of the input daatset to be analyzed")
    parser.add_argument('-t', '--toys', type=int, default = "100", help="number of toys to be processed")
    parser.add_argument('-sig','--signal', type=str, help="if signal", required=True)
    parser.add_argument('-DY','--DY', type=str, help='if DY+jets bkg', required=True)
    parser.add_argument('-CMS','--CMS', type=str, help='if data or MC', required=True)
    #parser.add_argument('-frac_bkg','--background', type=float, help="frac of background events", required=True)
    parser.add_argument('-frac_ref','--reference', type=float, help="frac of reference events", required=True)
    parser.add_argument('-epochs','--epochs', type=int, help="number of epochs", required=True)
    parser.add_argument('-latsize','--latsize', type=int, help="number of nodes in each hidden layer", required=True)
    parser.add_argument('-layers','--layers', type=int, help="number of layers", required=True)
    parser.add_argument('-wclip','--weight_clipping', type=float, help="weight clipping", required=True)
    parser.add_argument('-act','--internal_activation',type=str, help='internal activation', required=True)
    args = parser.parse_args()
    
    #folder to save the outputs of the pyscript
    mydir = args.output#+"/"
    os.system(f"mkdir {mydir}")
    
    #folder to save the outputs of each condor job (file.out, file.log, file.err)
    label = 'condor_'+args.output.split("/")[-1]+'_Higgs_'+str(time.time())
    os.system(f"mkdir {label}") # creo cartella condor con la sigla 

    for i in range(int(args.toys)):
        joblabel = str(i)#fileIN.split("/")[-1].replace(".h5","")                                                                                                                      
        if not os.path.isfile(f"{mydir}/{joblabel}.txt"):
            # src file
            script_src = open(f"{label}/{joblabel}.src" , 'w')
            script_src.write("#!/bin/bash\n")
            #script_src.write("source /cvmfs/sft.cern.ch/lcg/views/LCG_92//x86_64-slc6-gcc62-opt/setup.sh\n")
            script_src.write('eval "$(/lustre/cmswork/dalsanto/anaconda3/bin/conda shell.bash hook)" \n')
            script_src.write('conda activate tf_env \n')
            script_src.write(f"python {os.getcwd()}/{args.pyscript} -o {mydir}/{joblabel}  -t {int(args.toys)}  -frac_ref {args.reference} -sig {args.signal} -DY {args.DY} -CMS {args.CMS} -epochs {args.epochs} -latsize {args.latsize} -layers {args.layers} -wclip {args.weight_clipping} -act {args.internal_activation}") #/{joblabel} -i {args.input} #-frac_sig {args.signal} -frac_bkg {args.background}
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