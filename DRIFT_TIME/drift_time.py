import argparse
from DriftTime import DriftTime


def argParser():
    '''gestisce gli argomenti passati da linea di comando'''
    
    parser = argparse.ArgumentParser() 
    parser.add_argument('-i','--input', type=str, default='/lustre/cmswork/nlai/DATA/raw_data/', help="input directory")
    parser.add_argument('-o','--output', type=str, default='/lustre/cmswork/nlai/DATA/drift_distributions/', help="output directory")
    parser.add_argument('-run','--run', type=int, default=1220, help="run number")
    parser.add_argument('-n','--n_trigger', type=int, default=-1, help="number of trigger signals")
    parser.add_argument('-left','--left_bound', type=float, default=-400, help="left bound for cutting distributions")
    parser.add_argument('-right','--right_bound', type=float, default=900, help="right bound for cutting distributions")
    
    return parser.parse_args()


def main(args):

    RUNNUMBER = args.run
    
    INPUT_PATH = args.input
    OUTPUT_PATH = args.output
    
    N_TRIGGER = args.n_trigger
    
    L_BOUND = args.left_bound
    R_BOUND = args.right_bound
    
    OFFSET_DETECTOR = 100
    
    drift_instance = DriftTime(RUNNUMBER, INPUT_PATH, OUTPUT_PATH)
    
    drift_instance.read_data()
    drift_instance.compute_time()
    drift_instance.select_data()
    drift_instance.select_trigger(N_TRIGGER)
    raw_dt = drift_instance.compute_dt()
#     raw_dt = drift_instance.compute_dt_dataframe()

#     drift_instance.select_ndata(N_TRIGGER)
#     drift_instance.add_trigger_flag()
#     raw_dt = drift_instance.compute_dt_full()
    shifted_dt = drift_instance.shift_dt(raw_dt, OFFSET_DETECTOR)
    cut_dt = drift_instance.cut_dt(shifted_dt, L_BOUND, R_BOUND)
    
    
    drift_instance.save_dt(raw_dt, 'raw_hstat_condor')
    drift_instance.save_dt(shifted_dt, 'shifted_hstat_condor')
    drift_instance.save_dt(cut_dt, 'cut_shifted_hstat_condor')
    
#     drift_instance.make_distribution_plot(cut_dt, L_BOUND, R_BOUND, 'raw_hstat_condor')
#     drift_instance.make_distribution_plot(shifted_dt, L_BOUND, R_BOUND, 'shifted_hstat_condor')
#     drift_instance.make_comparison_plot(cut_dt, shifted_dt, L_BOUND, R_BOUND, 'comparison_hstat_condor')
    
    print('\n\nExiting...\n\n')
    
    return


if __name__ == "__main__":
    args = argParser()
    main(args)