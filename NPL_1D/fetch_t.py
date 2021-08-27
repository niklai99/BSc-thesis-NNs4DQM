from pathlib import Path
import argparse
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import chi2


class FetchTpath:
    '''Classe per andare a prendere il file contenente il t dopo un esecuzione condor'''
    
    def __init__(self, output_dir : str):
        '''Inizializzo la dir di output come la cartella usata da NPL_Train.py ovvero quello che scrivo in -o quando lancio il codice'''
        self.output_dir = output_dir
        return
    
    def fetch_dir(self, toy : str, dir_name : str):
        '''Entro nella dir del toy tipo OUT/TOY/NOME'''
        self.output_path = self.output_dir + toy + '/' + dir_name + '/'
        return
    
    def fetch_t_file(self, fname : str):
        '''Prendo il file contenente il t'''
        self.t_file_path = self.output_path + fname
        return self.t_file_path
        
    def printer(self):
        '''Metodo per printare cose a piacere'''
        print(self.t_file_path)
        return
    
    
class FetchTvalue:
    '''Classe per leggere il file contenente il t e metterlo in una lista'''
    
    def __init__(self, t_file : str):
        '''Inizializzo il file come il path al file'''
        self.t_file = t_file
        return
    
    def read_file(self, toy : str):
        '''Lettura del file contenente il t'''
        # se il file esiste leggo il file
        if self.t_file.is_file():
            t = ( np.loadtxt(self.t_file) ).item()
            return t
        # se il file non esiste scrivo che manca il toy
        else:
#             print(f'missing toy {toy}')
            return
    
    def filter_list(self, t_list : list):
        '''Metodo per filtrare la lista che viene creata: quando il file non esiste viene messo in append un None che qui tolgo'''
        filtered_list = filter(None.__ne__, t_list)
        t_list = list(filtered_list)
        return t_list
    


def argParser():
    '''Funzione per gestire gli argomenti'''
    parser = argparse.ArgumentParser()
    parser.add_argument('-o', '--output', type=str, default='./E200kW100/', help="output directory", required=False)
    parser.add_argument('-t', '--toys', type=int, default=40, help="number of toys", required=False)
    parser.add_argument('-plot', '--plot', action='count', default=0, help="bool counter to choose whether to plot or not", required=False)
    parser.add_argument('-s', '--save', action='count', default=0, help="bool counter to choose whether to save plot or not", required=False)
    args = parser.parse_args()
    return args


def chi2Fitter(data):
    par = chi2.fit(data, floc=0, fscale=1)
    return par

def change_legend(ax, new_loc, fontsize, titlesize, **kws):
    '''funzione per modificare posizione e font size della legenda generata da seaborn'''
    
    ax.legend()
    handles, labels = ax.get_legend_handles_labels()
    ax.legend([handles[idx] for idx in [1,0]], [labels[idx] for idx in [1,0]])
    
    old_legend = ax.legend_
    handles = old_legend.legendHandles
    labels = [t.get_text() for t in old_legend.get_texts()]
    title = old_legend.get_title().get_text()
    
    ax.legend(handles, labels, loc=new_loc, title=title, fontsize=fontsize, title_fontsize=titlesize, frameon = True, fancybox = False, framealpha = 0.5, **kws)
    
    return


def tDistPlotter(data, args, wclip):
    '''Funzione per plottare la t distribution'''
    
    XMIN = 0
    XMAX = max(data) + np.abs(min(data))
    
    par = chi2Fitter(data)
    NDF = par[0]
    print('NDF = ' + format(NDF, '1.4f'))
    
    xgrid = np.linspace(XMIN, XMAX, 500)
    bins=7
    
    fig, ax = plt.subplots(figsize=(12,7))
    
    ax = sns.histplot(x=data, bins=bins, stat='density', element='bars', fill=True, color='#009cff', edgecolor='white', label='t distribution')
    ax.plot(xgrid, chi2.pdf(xgrid, *par), color='#FF0000', linestyle='solid', linewidth=5, alpha=0.6, label='fitted chi2')
    ax.text(0.7, 0.7, 'Expected NDF = ' + format(10, '1.0f') + '\n' \
                        + 'Fitted NDF = '+ format(NDF, '1.4f') + '\n\n' \
                        + 'W_clip = ' + format(wclip, '1.2f') \
#                         + 'Compatibility = ' + format(NDF/10, '1.4f') \
                        , fontsize = 16, transform=ax.transAxes)
    
    ax.set_title('t distribution', fontsize = 18)
    ax.set_xlabel('t', fontsize = 16)
    ax.set_ylabel('Density', fontsize = 16)
    ax.tick_params(axis = 'both', which = 'major', labelsize = 14, direction = 'out', length = 5)
    
    ax.set_xlim(XMIN, XMAX)
    
    change_legend(ax=ax, new_loc="upper right", fontsize=14, titlesize=16)
    
    fig.tight_layout()
    
    if args.save==1:
        fig.savefig('/lustre/cmswork/nlai/PLOTS/NPL_TEST/t_dist_E200k_w100_40kref_4kbkg.png', dpi = 300, facecolor = 'white')
        print('Figure saved')
    elif args.save!=1:
        pass
    
    plt.show()
    return 
    

def main():
    '''Main function per la lettura di tutti i valori di t trovati nel training della NN con condor'''
    
    args = argParser()
    
    OUTPUT_DIR = args.output
    TOYS = args.toys
    WCLIP = 7.0
    STD_DIR_NAME = f'1D_patience1000_ref40000_bkg4000_sig0_epochs100000_latent3_layers1_wclip{WCLIP}'
    STD_FILE_NAME = f'1D_patience1000_ref40000_bkg4000_sig0_toy{TOYS}_t.txt' 
    
    t_path = FetchTpath(OUTPUT_DIR)
    
    t_list = []
    
    for toy in range(TOYS):
        t_path.fetch_dir(str(toy), STD_DIR_NAME)
        t_file = Path( t_path.fetch_t_file(STD_FILE_NAME) )
        t = FetchTvalue(t_file)
        tval = t.read_file(toy)
        t_list.append(tval)
    
    t_list = t.filter_list(t_list)
#     print(t_list)
    missing_toys = TOYS - len(t_list)
    if missing_toys!=0:
        print('Missing Toys: ' + str(missing_toys))
    elif missing_toys==0:
        print('All toys processed!')
    
    if args.plot==1:
        tDistPlotter(data=t_list, args=args, wclip=WCLIP)
    elif args.plot!=1:
        pass
    
    return



if __name__ == "__main__":
    main()