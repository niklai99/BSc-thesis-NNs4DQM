import argparse
from Occupancy import Occupancy


def argParser():
    '''gestisce gli argomenti passati da linea di comando'''
    
    parser = argparse.ArgumentParser() 
    parser.add_argument('-i','--input', type=str, default='/lustre/cmswork/nlai/DATA/raw_data/', help="input directory")
    parser.add_argument('-o','--output', type=str, default='/lustre/cmswork/nlai/DATA/', help="output directory")
    parser.add_argument('-plot','--plot_path', type=str, default='/lustre/cmswork/nlai/PLOTS/OCCUPANCY/', help="output directory")
    parser.add_argument('-run','--run', type=int, default=1220, help="run number")
    
    return parser.parse_args()


def main(args):

    RUNNUMBER = args.run
    
    INPUT_PATH = args.input
    OUTPUT_PATH = args.output
    PLOT_PATH = args.plot_path
    
    occupancy_instance = Occupancy(RUNNUMBER, INPUT_PATH, OUTPUT_PATH, PLOT_PATH)
    
    occupancy_instance.read_data()
    occupancy_instance.compute_time()
    data = occupancy_instance.select_data()
    run_time = occupancy_instance.compute_run_time()
    
    data_fpga0, data_fpga1 = occupancy_instance.split_fpga()
    
    hist_0 = occupancy_instance.make_histogram(data_fpga0)
    hist_1 = occupancy_instance.make_histogram(data_fpga1)
    
    rate_hist_0 = occupancy_instance.make_rate_histogram(hist_0, run_time)
    rate_hist_1 = occupancy_instance.make_rate_histogram(hist_1, run_time)
    
    occupancy_instance.save_hist(hist_0, 'occ', 'fpga0')
    occupancy_instance.save_hist(hist_1, 'occ', 'fpga1')
    
    occupancy_instance.save_hist(rate_hist_0, 'rate', 'fpga0')
    occupancy_instance.save_hist(rate_hist_1, 'rate', 'fpga1')
    
    occupancy_instance.make_distribution_plot(hist_0, 'occ', 'fpga0')
    occupancy_instance.make_distribution_plot(hist_1, 'occ', 'fpga1')
    occupancy_instance.make_comparison_plot(hist_0, hist_1, 'occ')
    
    occupancy_instance.make_distribution_plot(rate_hist_0, 'rate', 'fpga0')
    occupancy_instance.make_distribution_plot(rate_hist_1, 'rate', 'fpga1')
    occupancy_instance.make_comparison_plot(rate_hist_0, rate_hist_1, 'rate')
    
    
    print('\n\nExiting...\n\n')
    
    return


if __name__ == "__main__":
    args = argParser()
    main(args)