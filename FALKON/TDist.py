import os
import h5py
import numpy as np
import pandas as pd
import scipy.stats
import matplotlib.pyplot as plt
import seaborn as sns

from RunParameters import RunParameters



class TDist:
    '''classe che gestisce la distribuzione dei t'''
    
    def __init__(self, toys, epochs, nref, nbkg, nsig, check_point_t, w_clip, out_dir, tfile, thistory):
        '''init: imposto parametri importanti della run e i file di interesse'''
        
        self.epochs = epochs
        self.nref = nref
        self.nbkg = nbkg
        self.nsig = nsig
        self.check_point_t = check_point_t
        self.toys = toys
        self.wclip = w_clip
        self.OUT_PATH = out_dir
        self.OUT_FILE_t = tfile
        self.OUT_FILE_t_history = thistory
        
        return
    
    def __call__(self):
        '''ciclo su tutti i file per prendere tutti i t e tutte le history'''
        
        # inizializzo le liste
        self.t_list=[]
        self.t_list_history=[]
        
        counter=0
        preview=0
        
        # ciclo su tutti i toy
        for i in range(self.toys-preview):
    
            # entro nella output directory, numero magico del toy i-esimo, e prendo il file t
            file_name = (
                self.OUT_PATH + 
                f'/{i}/' + 
                f'E{self.epochs}_latent3_layers1_wclip{self.wclip}_ntoy{self.toys}_ref{self.nref}_bkg{self.nbkg}_sig{self.nsig}_patience{self.check_point_t}' + 
                self.OUT_FILE_t
            )
            
            # controllo se è effettivamente un file esistente
            if os.path.isfile(file_name):
                # apro il file in read mode
                f = open(file_name, "r")
                # leggo cosa c'è scritto e lo inserisco nella lista
                self.t_list.append(float(f.readline()[:-1]))
                # chiudo il file
                f.close()
#                 print(i)
                # modo stranissimo per dire che se è andato tutto bene allora aumento il counter
                if np.logical_not(np.isnan(self.t_list[-1])):
                    counter += 1
            
            # entro nella output directory, numero magico del toy i-esimo, e prendo il file con la storia di t
            history_name = (
                self.OUT_PATH + 
                f'/{i}/' + 
                f'E{self.epochs}_latent3_layers1_wclip{self.wclip}_ntoy{self.toys}_ref{self.nref}_bkg{self.nbkg}_sig{self.nsig}_patience{self.check_point_t}' + 
                self.OUT_FILE_t_history
            )
            
            # controllo se è effettivamente un file esistente
            if os.path.isfile(history_name):
                # apro il file in read mode
                f = h5py.File(history_name, "r")
                # leggo cosa c'è scritto e lo inserisco nella lista
                try:
                    self.t_list_history.append(-2*np.array(f.get('loss')))
                except: 
                    print('Problem with toy ', i)
                # chiudo il file
                f.close()
        
        # converto in numpy array 
        self.t_list=np.array(self.t_list)
        self.t_list_history=np.array(self.t_list_history)
        
        # rimuovo nan values
        self.t_list = self.t_list[~np.isnan(self.t_list)]
        self.t_list_history = self.t_list_history[~np.isnan(self.t_list_history).any(axis=1), :]
        
#         self.t_list = self.t_list[self.t_list<40]
#         self.t_list_history = self.t_list_history[self.t_list_history[:, -1]<40]
        
        print(f"\nToys at disposal/Total toys: {counter}/{self.toys-preview}")
        
        return self.t_list, self.t_list_history
    
    def change_legend(self, ax, new_loc, fontsize, titlesize, **kws):
        '''funzione per modificare posizione e font size della legenda generata da seaborn'''

        old_legend = ax.legend_
        handles = old_legend.legendHandles
        labels = [t.get_text() for t in old_legend.get_texts()]
        title = old_legend.get_title().get_text()

        ax.legend(handles, labels, loc=new_loc, title=title, 
                  fontsize=fontsize, title_fontsize=titlesize, 
                  frameon = True, fancybox = False, framealpha = 0.5, **kws)
    
        return
    
    def plotterLayout(self, ax, title: str, titlefont: int, xlabel: str, ylabel: str, labelfont: int, xlimits=[], ylimits=[]):
        
        if xlimits:
            ax.set_xlim(xlimits[0], xlimits[1])
        if ylimits:
            ax.set_ylim(ylimits[0], ylimits[1])
            
        ax.set_title(title, fontsize=titlefont)
        
        ax.set_xlabel(xlabel, fontsize = labelfont)
        ax.set_ylabel(ylabel, fontsize = labelfont)
        
        # sistemo i ticks
        ax.tick_params(axis = 'both', which = 'major', labelsize = 14, direction = 'out', length = 5)
        
        return
    
    def plotOutPath(self, folder):
        '''costruisco il percorso di salvataggio delle figure'''
        
        path = f'/lustre/cmswork/nlai/PLOTS/DRIFT_TIME/{folder}/' 
        rPar = RunParameters(self.OUT_PATH, 0, self.toys)
        toys, w_clip, epochs, check_point_t, ref, bkg, sig, latent, layers = rPar.fetch_parameters()
        
        pngfile = f'E{epochs}_latent{latent}_layers{layers}_wclip{w_clip}_ntoy{toys}_ref{ref}_bkg{bkg}_sig{sig}_patience{check_point_t}'
        
        pngfile = pngfile.replace(' ', '')
        
        self.pngpath = path + pngfile
        
        return self.pngpath
    
    def plotTdist(self, t_list, t_list_ref, bins=7, ref_bins=7, dof=10, plot_folder = None):
        '''grafico della distribuzione dei t'''
        
        plt.rcParams["patch.force_edgecolor"] = True
            
#         XLIM = [xmin, xmax]
        
        # creo la griglia lungo x
        XGRID = np.linspace(0, 40, 1000)
        
        # fit della distribuzione con un chi2
        fit_par = scipy.stats.chi2.fit(t_list, floc=0, fscale=1)
        
        # creo figure&axes
        fig, ax = plt.subplots(figsize=(12,7))
        
        # istogramma della distrubuzione dei t
        sns.histplot(x=t_list, bins=bins, 
                          stat='density', element='bars', fill=True, color='#ff954c', edgecolor='#FF6800', 
                          label='t distribution', ax=ax)
        
        # parte di codice per aggiungere l'incertezza ai bin 
        hist, bin_edges = np.histogram(t_list, density=True, bins=bins)
        binswidth = bin_edges[1]-bin_edges[0]
        central_points = []
        for i in range(0, len(bin_edges)-1):
            half = (bin_edges[i] + bin_edges[i+1])/2
            central_points.append(half)
        # calcolo dell'incertezza dei bin
        err = np.sqrt(hist/(t_list.shape[0]*binswidth))
        # grafico delle incertezze sui bin 
        ax.errorbar(central_points, hist, yerr = err, color='#FF6800', marker='o', ls='')
        
        # istogramma della distribuzione reference
        sns.histplot(
            x=t_list_ref, bins=ref_bins, 
            stat='density', element='bars', fill=True, color='#aadeff', edgecolor='#009cff', 
            label='reference distribution', ax=ax, linewidth=1
        )
        
        # parte di codice per aggiungere l'incertezza ai bin 
        hist, bin_edges = np.histogram(t_list_ref, density=True, bins=ref_bins)
        binswidth = bin_edges[1]-bin_edges[0]
        central_points = []
        for i in range(0, len(bin_edges)-1):
            half = (bin_edges[i] + bin_edges[i+1])/2
            central_points.append(half)
        # calcolo dell'incertezza dei bin
        err = np.sqrt(hist/(t_list_ref.shape[0]*binswidth))
        # grafico delle incertezze sui bin 
        ax.errorbar(central_points, hist, yerr = err, color='#009cff', marker='o', ls='')
        
        # grafico della distribuzione teorica del chi2
        ax.plot(XGRID, scipy.stats.chi2.pdf(XGRID, df=dof), 
                color='#009cff', linestyle='solid', linewidth=5, alpha=0.6, 
                label=f'theoretical distribution, dof: {dof}')
        
        
        self.plotterLayout(ax=ax, title='t distribution', titlefont=18, xlabel='t', ylabel='density', labelfont=16)
        ax.set_xlim(left=0)
        # gestione della legenda
        ax.legend()
        handles, labels = ax.get_legend_handles_labels()
        ax.legend([handles[idx] for idx in [0, 2, 1]], [labels[idx] for idx in [0, 2, 1]])
        self.change_legend(ax=ax, new_loc="upper right", fontsize=14, titlesize=16)
        
        fig.tight_layout()
        if plot_folder:
            fig.savefig(self.plotOutPath(plot_folder)+'_Tdistribution.png', dpi = 300, facecolor='white')
        plt.show()
        
        return 
    
    
    def plotMedianHistory(self, t_history, dof=10, plot_folder = None):
        '''andamento della mediana'''
        
        median_history = np.median(t_history, axis=0)
        
        th_median = scipy.stats.chi2.median(df=dof)
        
        XMIN = 0
        XMAX = self.epochs
            
        XLIM = [XMIN, XMAX]
        
        fig, ax = plt.subplots(figsize=(12,7))
        
        x_tics = np.array(range(self.epochs))
        x_tics = x_tics[x_tics % self.check_point_t == 0]
        
        
        ax.plot(x_tics[:], median_history[:], color='#009cff', linestyle='solid', linewidth=3, alpha=1, 
                label=f'median final value: {median_history[-1]:.3f}')
        
        ax.hlines(y=th_median, xmin = XMIN, xmax = XMAX, 
                      color = '#FF0000', linestyle='dashed', linewidth = 3, alpha = 0.5, 
                    label = f'theoretical median: {th_median:.3f}')
        
        self.plotterLayout(ax=ax, xlimits=XLIM, title='median history', titlefont=18, xlabel='training epoch', ylabel='median', labelfont=16)
        ax.set_ylim(bottom=0)
        
        ax.legend()
        self.change_legend(ax=ax, new_loc="upper right", fontsize=14, titlesize=16)
    
        fig.tight_layout()
        if plot_folder:
            fig.savefig(self.plotOutPath(plot_folder)+'_median_history.png', dpi = 300, facecolor='white')
        plt.show()
        
        return median_history
    
    
    def plotMedianPval(self, median_history, dof=10, plot_folder = None):
        '''andamento del pvalue della mediana'''
        
        median_pval = scipy.stats.chi2.sf(median_history[:], df=dof)
        
        XMIN = 0
        XMAX = self.epochs
        
        XLIM = [XMIN, XMAX]
#         YLIM = [ymin, ymax]

        fig, ax = plt.subplots(figsize=(12,7))

        x_tics = np.array(range(self.epochs))
        x_tics = x_tics[x_tics % self.check_point_t == 0]
#         y_tics = np.array( np.arange(ymin, ymax, 0.5) )
        
        ax.plot(x_tics[10:], median_pval[10:], color='#009cff', linestyle='solid', linewidth=3, alpha=1, 
                label=f'median p-val final value: {median_pval[-1]:.3f}')
        
        self.plotterLayout(ax=ax, xlimits=XLIM, title='median p-value evolution', titlefont=18, xlabel='training epoch', ylabel='p-value', labelfont=16)
        
#         ax.set_yticks(y_tics)
        
        ax.legend()
        self.change_legend(ax=ax, new_loc="upper right", fontsize=14, titlesize=16)
        
        fig.tight_layout()
        if plot_folder:
            fig.savefig(self.plotOutPath(plot_folder)+'_median_pvalue.png', dpi = 300, facecolor='white')
        plt.show()
        return median_pval
    
    
    def plotMedianZ(self, median_pval, plot_folder = None):
        '''andamento della significanza della mediana'''
        
        median_Z = np.abs(scipy.stats.norm.ppf(1-median_pval[:]))
        
        XMIN = 0
        XMAX = self.epochs

        XLIM = [XMIN, XMAX]
    
        fig, ax = plt.subplots(figsize=(12,7))

        x_tics = np.array(range(self.epochs))
        x_tics = x_tics[x_tics % self.check_point_t == 0]
        
        ax.plot(x_tics[10:], median_Z[10:], color='#009cff', linestyle='solid', linewidth=3, alpha=1, 
                label=f'median Z final value: {median_Z[-1]:.3f}')
        
        self.plotterLayout(ax=ax, xlimits=XLIM, title='median significance evolution', titlefont=18, xlabel='training epoch', ylabel='Z', labelfont=16)
        
        ax.legend()
        self.change_legend(ax=ax, new_loc="upper right", fontsize=14, titlesize=16)
        
        fig.tight_layout()
        if plot_folder:
            fig.savefig(self.plotOutPath(plot_folder)+'_median_significance.png', dpi = 300, facecolor='white')
        plt.show()
        return median_Z