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
    
    def __init__(self, dof, bins, toys, epochs, check_point_t, w_clip, out_dir, tfile, thistory, save_flag):
        '''init: imposto parametri importanti della run e i file di interesse'''
        
        self.dof = dof
        self.bins = bins
        self.epochs = epochs
        self.check_point_t = check_point_t
        self.toys = toys
        self.wclip = w_clip
        self.OUT_PATH = out_dir
        self.OUT_FILE_t = tfile
        self.OUT_FILE_t_history = thistory
        self.save_flag = save_flag
        
        return
    
    
    def plotOutPath(self):
        '''costruisco il percorso di salvataggio delle figure'''
        
        path = '/lustre/cmswork/nlai/PLOTS/test/' # CAMBIARE
        rPar = RunParameters(self.OUT_PATH, 0, self.toys)
        toys, w_clip, epochs, check_point_t, ref, bkg, sig, latent, layers = rPar.fetch_parameters()
        
        pngfile = f'E{epochs}_latent{latent}_layers{layers}_wclip{w_clip}_ntoy{toys}_ref{ref}_bkg{bkg}_sig{sig}_patience{check_point_t}'
        
        pngfile = pngfile.replace(' ', '')
        
        self.pngpath = path + pngfile
        
        return self.pngpath
    
    
    def getT(self):
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
                f'E{self.epochs}_latent3_layers1_wclip{self.wclip}_ntoy{self.toys}_ref200000_bkg10000_sig0_patience1000' + 
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
                f'E{self.epochs}_latent3_layers1_wclip{self.wclip}_ntoy{self.toys}_ref200000_bkg10000_sig0_patience1000' + 
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
        
        self.t_list = self.t_list[self.t_list<40]
        self.t_list_history = self.t_list_history[self.t_list_history[:, -1]<40]
        
        print(f"\nToys at disposal/Total toys: {counter}/{self.toys-preview}")
        
        return 
    
    
    def computeMedian(self):
        '''calcolo la mediana per un rapido controllo di compatibilità'''
        
        # calcolo la mediana della lista
        median_t = np.median(self.t_list)
        print("\nMedian t distribution: ", median_t)
        
        # calcolo il p-value della lista 
        p_list = np.sum([1/(len(self.t_list)) for x in self.t_list if x>median_t])
        print(
            f"Median p-value: {p_list :.4f}\
            Median significance: {scipy.stats.norm.ppf(1-p_list):.4f}\
            from t list"
        )
        
        print(
            f"Median p-value: {scipy.stats.chi2.sf(median_t, df=self.dof):.4f}\
            Median significance: {scipy.stats.norm.ppf(1-scipy.stats.chi2.sf(median_t, df=self.dof)):.4f}\
            from chi2 distribution" 
        ) 
        
        return
    
    
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
    
    
    def plotTdist(self):
        '''grafico della distribuzione dei t'''
        
        # gestione del plot range
        XMIN = 0
        if max(self.t_list) >= 3*self.dof:
            XMAX = max(self.t_list) + min(self.t_list) 
        elif max(self.t_list) < 3*self.dof:
            XMAX = 3*self.dof
            
        XLIM = [XMIN, XMAX]
        
        # creo la griglia lungo x
        XGRID = np.linspace(XMIN, XMAX, 500)
        
        # numero di bin da utilizzare
        BINS = self.bins
        
        # fit della distribuzione con un chi2
        fit_par = scipy.stats.chi2.fit(self.t_list, floc=0, fscale=1)
        
        # creo figure&axes
        fig, ax = plt.subplots(figsize=(12,7))
        
        # istogramma della distrubuzione dei t
        ax = sns.histplot(x=self.t_list, bins=BINS, 
                          stat='density', element='bars', fill=True, color='#aadeff', edgecolor='#009cff', 
                          label='t distribution')
        
        # parte di codice per aggiungere l'incertezza ai bin 
        hist, bin_edges = np.histogram(self.t_list, density=True, bins=BINS)
        binswidth = bin_edges[1]-bin_edges[0]
        central_points = []
        for i in range(0, len(bin_edges)-1):
            half = (bin_edges[i] + bin_edges[i+1])/2
            central_points.append(half)
        # calcolo dell'incertezza dei bin
        err = np.sqrt(hist/(self.t_list.shape[0]*binswidth))
        # grafico delle incertezze sui bin 
        ax.errorbar(central_points, hist, yerr = err, color='#009cff', marker='o', ls='')
        
        # grafico della distribuzione teorica del chi2
        ax.plot(XGRID,scipy.stats.chi2.pdf(XGRID, df=self.dof), 
                color='#00FF00', linestyle='solid', linewidth=5, alpha=0.6, 
                label=f'theoretical distribution, dof: {self.dof}')
        
        # grafico del fit della distribuzione 
        ax.plot(XGRID, scipy.stats.chi2.pdf(XGRID, *fit_par), 
                color='#FF0000', linestyle='solid', linewidth=5, alpha=0.6, 
                label=f'fitted chi2, dof: '+format(fit_par[0],'1.4f'))
        
        
        self.plotterLayout(ax=ax, xlimits=XLIM, title='t distribution', titlefont=18, xlabel='t', ylabel='density', labelfont=16)
        
        # gestione della legenda
        ax.legend()
        handles, labels = ax.get_legend_handles_labels()
        ax.legend([handles[idx] for idx in [2, 0, 1]], [labels[idx] for idx in [2, 0, 1]])
        self.change_legend(ax=ax, new_loc="upper right", fontsize=14, titlesize=16)
        
        fig.tight_layout()
        if self.save_flag:
            fig.savefig(self.plotOutPath()+'_distribution.png', dpi = 300, facecolor='white')
        plt.show()
        return 
    
    
    def plotThistory(self):
        '''grafico della storia dei t'''
        
        
        # gestione del plot range
        XMIN = 0
        XMAX = self.epochs
        
        YMIN = 0
        if max(self.t_list) >= 3*self.dof:
            YMAX = max(self.t_list) + min(self.t_list) 
        elif max(self.t_list) < 3*self.dof:
            YMAX = 3*self.dof
            
        XLIM = [XMIN, XMAX]
        YLIM = [YMIN, YMAX]
        
        # creo figura e axes
        fig, ax = plt.subplots(figsize=(12,7))
        
        # ticks dell'asse x
        x_tics = np.array(range(self.epochs))
        x_tics = x_tics[x_tics % self.check_point_t == 0]
        
        # ciclo per plottare la storia di tutti i toys
        for i in range(len(self.t_list_history)):
            ax.plot(x_tics[1:], self.t_list_history[i][1:])
        
        # plotter layout method
        self.plotterLayout(ax=ax, xlimits=XLIM, ylimits=YLIM, title='t history', titlefont=18, xlabel='training epoch', ylabel='t', labelfont=16)
        
        fig.tight_layout()
        if self.save_flag:
            fig.savefig(self.plotOutPath()+'_history.png', dpi = 300, facecolor='white')
        plt.show()
        return
    
    
    def plotQuantilesEvo(self):
        '''grafico dell'evoluzione dei quantili della distribuzione dei t'''
        
        # gestione del plot range
        XMIN = 0
        XMAX = self.epochs
        
        YMIN = 0
        if max(self.t_list) >= 3*self.dof:
            YMAX = max(self.t_list) + min(self.t_list) 
        elif max(self.t_list) < 3*self.dof:
            YMAX = 3*self.dof
            
            
        XLIM = [XMIN, XMAX]
        YLIM = [YMIN, YMAX]
        
        # creo figura e axes
        fig, ax = plt.subplots(figsize=(12,7))
        
        # ticks
        x_tics = np.arange(0, self.epochs, self.check_point_t)
#         print(x_tics)
        
        # quantili
        quantile_list = [0.05,0.25,0.50,0.75,0.95]
        color_list = ['#ff0000', '#00ff00', '#0080ff', '#ff8000', '#ff00ff']
        
        # quantili teorici
        th_quantile_position = [scipy.stats.chi2.ppf(i, df=self.dof) for i in quantile_list]
        
        # quantili distribuzione
        t_quantile = np.quantile(self.t_list_history[:], quantile_list, axis=0)
        
        # ciclo per plottare i 5 quantili 
        for i in range(len(quantile_list)):
            ax.plot(x_tics[:], t_quantile[i][:], 
                    color = color_list[i], linestyle='solid', linewidth = 3, 
                    label = format(quantile_list[i], '1.2f'))
            ax.plot(x_tics[-1], th_quantile_position[i], marker='X', markersize = 15,
                    color = color_list[i])
            ax.hlines(y=th_quantile_position[i], xmin = XMIN, xmax = XMAX, 
                      color = color_list[i], linestyle='dashed', linewidth = 3, alpha = 0.5, 
                    label = 'theoretical ' + format(quantile_list[i], '1.2f'))
        
        # plot layout method
        self.plotterLayout(ax=ax, xlimits=XLIM, ylimits=YLIM, title='t quantiles evolution', titlefont=18, xlabel='training epoch', ylabel='t', labelfont=16)
        
        # plot legend
        ax.legend(title = 'Quantiles', ncol = 2)
        self.change_legend(ax=ax, new_loc="upper right", fontsize=14, titlesize=16)
        ax.legend(title = 'Quantiles', ncol = 2)
        
        fig.tight_layout()
        if self.save_flag:
            fig.savefig(self.plotOutPath()+'_quantiles.png', dpi = 300, facecolor='white')
        plt.show()
        return
    
    
    def plotChi2History(self):
        '''grafico andamento del chi2 per epoche'''
        
        self.t_chi2_history = []
        bin_number = self.bins 
        
        for counter in range(10, self.t_list_history.shape[1]): 
            # binning della distribuzione dei t per ogni checkpoint
            t_hist, binedges = np.histogram(self.t_list_history[:, counter], bins=bin_number, density=False) 
            # cerco il centro di ciascun bin
            bincenters = (binedges[:-1] + binedges[1:]) / 2
            # calcolo la larghezza dei bin
            bin_width = binedges[1]-binedges[0]
            # calcolo l'area dell'istogramma
            area_hist = bin_width*self.toys
            # calcolo l'altezza teorica di ciascun bin
            th_bins=[]
            for i in range(len(bincenters)):
                area = scipy.integrate.quad(lambda x: (area_hist*scipy.stats.chi2.pdf(x, df=self.dof)), binedges[i], binedges[i+1])[0]
                th_bins.append(area/(bin_width))
            # calcolo il chi2
            self.t_chi2_history.append( np.sum( (t_hist-th_bins)**2/th_bins, axis=0 ) ) 
            
        XMIN = 0
        XMAX = self.epochs
        
        XLIM = [XMIN, XMAX]

        fig, ax = plt.subplots(figsize=(12,7))

        x_tics = np.array(range(self.epochs))
        x_tics = x_tics[x_tics % self.check_point_t == 0][10:]

        
        ax.plot(x_tics[:], self.t_chi2_history[:], color='#009cff', linestyle='solid', linewidth=3, alpha=1, 
                label=fr'$\chi^2$ final value: {self.t_chi2_history[-1]:.3f}')
        
        self.plotterLayout(ax=ax, xlimits=XLIM, title=r'$\chi^2$ evolution', titlefont=18, xlabel='training epoch', ylabel=r'$\chi^2$', labelfont=16)
        ax.set_yscale('log')
        
        ax.legend()
        self.change_legend(ax=ax, new_loc="upper right", fontsize=14, titlesize=16)
        
        fig.tight_layout()
        if self.save_flag:
            fig.savefig(self.plotOutPath()+'_chi2.png', dpi = 300, facecolor='white')
        plt.show()

        return
    
    
    def plotChi2Compatibility(self):
        '''andamento compatibilità del chi2 con i gradi di libertà per epoche'''
        
        # calcolo la compatibilità come chi_nu/nu
        self.t_chi2_compatibility = np.array(self.t_chi2_history[:])/(self.bins-1)
    
        XMIN = 0
        XMAX = self.epochs
        
        XLIM = [XMIN, XMAX]

        fig, ax = plt.subplots(figsize=(12,7))

        x_tics = np.array(range(self.epochs))
        x_tics = x_tics[x_tics % self.check_point_t == 0][10:]
        
        ax.plot(x_tics[:], self.t_chi2_compatibility[:], color='#009cff', linestyle='solid', linewidth=3, alpha=1, 
                label=fr'$\chi^2 / \nu$ final value: {self.t_chi2_compatibility[-1]:.3f}'
               )
        
        self.plotterLayout(ax=ax, xlimits=XLIM, title='compatibility evolution', titlefont=18, xlabel='training epoch', ylabel=r'$\chi^2 / \nu$', labelfont=16)
        ax.set_yscale('log')
        
        ax.legend()
        self.change_legend(ax=ax, new_loc="upper right", fontsize=14, titlesize=16)
        
        fig.tight_layout()
        if self.save_flag:
            fig.savefig(self.plotOutPath()+'_compatibility.png', dpi = 300, facecolor='white')
        plt.show()
        
        return
    
    def plotChi2Compatibility2(self):
        '''andamento compatibilità del chi2 con i gradi di libertà per epoche -- metodo 2'''
        
        # calcolo la compatibilità come chi_nu/nu
        self.t_chi2_compatibility =  np.abs( np.array(self.t_chi2_history[:]) - (self.bins-1) )  / np.sqrt(2*(self.bins-1))
    
        XMIN = 0
        XMAX = self.epochs
        
        XLIM = [XMIN, XMAX]

        fig, ax = plt.subplots(figsize=(12,7))

        x_tics = np.array(range(self.epochs))
        x_tics = x_tics[x_tics % self.check_point_t == 0][10:]
        
        ax.plot(x_tics[:], self.t_chi2_compatibility[:], color='#009cff', linestyle='solid', linewidth=3, alpha=1, 
                label=fr'$\lambda$ final value: {self.t_chi2_compatibility[-1]:.3f}'
               )
        
        self.plotterLayout(ax=ax, xlimits=XLIM, title='compatibility evolution - version 2', titlefont=18, xlabel='training epoch', ylabel=r'$\lambda$', labelfont=16)
        ax.set_yscale('log')
        
        ax.legend()
        self.change_legend(ax=ax, new_loc="upper right", fontsize=14, titlesize=16)
        
        fig.tight_layout()
        if self.save_flag:
            fig.savefig(self.plotOutPath()+'_compatibility2.png', dpi = 300, facecolor='white')
        plt.show()
        
        return
    
    
    def plotPValHistory(self):
        '''andamento del pvalue del chi2 per epoche'''
        
        self.pvalue_history = scipy.stats.chi2.sf(self.t_chi2_history[:], df=self.bins-1) 

        XMIN = 0
        XMAX = self.epochs
        YMIN = 0
        YMAX = 1.2
        
        XLIM = [XMIN, XMAX]
#         YLIM = [YMIN, YMAX]

        fig, ax = plt.subplots(figsize=(12,7))

        x_tics = np.array(range(self.epochs))
        x_tics = x_tics[x_tics % self.check_point_t == 0][10:]
#         y_tics = np.array( np.arange(0, 1.1, 0.1) )
        
        ax.plot(x_tics[:], self.pvalue_history[:], color='#009cff', linestyle='solid', linewidth=3, alpha=1, 
                label=f'p-val final value: {self.pvalue_history[-1]:.3f}')
        
        self.plotterLayout(ax=ax, xlimits=XLIM, title='p-value evolution', titlefont=18, xlabel='training epoch', ylabel='p-value', labelfont=16)
#         ax.set_yticks(y_tics)
        
        ax.legend()
        self.change_legend(ax=ax, new_loc="upper right", fontsize=14, titlesize=16)
        
        fig.tight_layout()
        if self.save_flag:
            fig.savefig(self.plotOutPath()+'_pvalue.png', dpi = 300, facecolor='white')
        plt.show()
        
        return
    
    
    def plotSignificanceHistory(self):
        '''andamento della significanza del chi2 per epoche'''
        
        self.significance_history=np.abs(scipy.stats.norm.ppf(1-self.pvalue_history[:]))

        XMIN = 0
        XMAX = self.epochs
#         YMIN = 0
#         YMAX = max(self.significance_history)+min(self.significance_history)

        XLIM = [XMIN, XMAX]

        fig, ax = plt.subplots(figsize=(12,7))

        x_tics = np.array(range(self.epochs))
        x_tics = x_tics[x_tics % self.check_point_t == 0][10:]
#         y_tics = np.array( np.arange(0, 1.1, 0.1) )
        
        ax.plot(x_tics[:], self.significance_history[:], color='#009cff', linestyle='solid', linewidth=3, alpha=1, 
                label=f'Z final value: {self.significance_history[-1]:.3f}')
        
        self.plotterLayout(ax=ax, xlimits=XLIM, title='significance evolution', titlefont=18, xlabel='training epoch', ylabel='Z', labelfont=16)
        
        ax.legend()
        self.change_legend(ax=ax, new_loc="upper right", fontsize=14, titlesize=16)
        
        fig.tight_layout()
        if self.save_flag:
            fig.savefig(self.plotOutPath()+'_significance.png', dpi = 300, facecolor='white')
        plt.show()
        
        return
    
    
    def plotMedianHistory(self):
        '''andamento della mediana'''
        
        self.median_history = np.median(self.t_list_history, axis=0)
        
        th_median = scipy.stats.chi2.median(df=self.dof)
        
        XMIN = 0
        XMAX = self.epochs
        
        YMIN = 0
        if max(self.median_history) >= 3*self.dof:
            YMAX = max(self.median_history) + min(self.median_history) 
        elif max(self.median_history) < 3*self.dof:
            YMAX = 3*self.dof
            
        XLIM = [XMIN, XMAX]
        YLIM = [YMIN, YMAX]
        
        fig, ax = plt.subplots(figsize=(12,7))
        
        x_tics = np.array(range(self.epochs))
        x_tics = x_tics[x_tics % self.check_point_t == 0]
        
        
        ax.plot(x_tics[:],self.median_history[:], color='#009cff', linestyle='solid', linewidth=3, alpha=1, 
                label=f'median final value: {self.median_history[-1]:.3f}')
        
        ax.hlines(y=th_median, xmin = XMIN, xmax = XMAX, 
                      color = '#FF0000', linestyle='dashed', linewidth = 3, alpha = 0.5, 
                    label = f'theoretical median: {th_median:.3f}')
        
        self.plotterLayout(ax=ax, xlimits=XLIM, ylimits=YLIM, title='median history', titlefont=18, xlabel='training epoch', ylabel='median', labelfont=16)
        
        ax.legend()
        self.change_legend(ax=ax, new_loc="upper right", fontsize=14, titlesize=16)
    
        fig.tight_layout()
        if self.save_flag:
            fig.savefig(self.plotOutPath()+'_median_history.png', dpi = 300, facecolor='white')
        plt.show()
        return
    
    
    def plotMedianPval(self):
        '''andamento del pvalue della mediana'''
        
        self.median_pval = scipy.stats.chi2.sf(self.median_history[:], df=self.dof)
        
        XMIN = 0
        XMAX = self.epochs
        YMIN = 0
        YMAX = 1.2
        
        XLIM = [XMIN, XMAX]
#         YLIM = [YMIN, YMAX]

        fig, ax = plt.subplots(figsize=(12,7))

        x_tics = np.array(range(self.epochs))
        x_tics = x_tics[x_tics % self.check_point_t == 0]
#         y_tics = np.array( np.arange(0, 1.1, 0.1) )
        
        ax.plot(x_tics[10:], self.median_pval[10:], color='#009cff', linestyle='solid', linewidth=3, alpha=1, 
                label=f'median p-val final value: {self.median_pval[-1]:.3f}')
        
        self.plotterLayout(ax=ax, xlimits=XLIM, title='median p-value evolution', titlefont=18, xlabel='training epoch', ylabel='p-value', labelfont=16)
#         ax.set_yticks(y_tics)
        
        ax.legend()
        self.change_legend(ax=ax, new_loc="upper right", fontsize=14, titlesize=16)
        
        fig.tight_layout()
        if self.save_flag:
            fig.savefig(self.plotOutPath()+'_median_pvalue.png', dpi = 300, facecolor='white')
        plt.show()
        return
    
    
    def plotMedianZ(self):
        '''andamento della significanza della mediana'''
        
        self.median_Z = np.abs(scipy.stats.norm.ppf(1-self.median_pval[:]))
        
        XMIN = 0
        XMAX = self.epochs
        YMIN = 0
        YMAX = 1.2
        
        XLIM = [XMIN, XMAX]
#         YLIM = [YMIN, YMAX]

        fig, ax = plt.subplots(figsize=(12,7))

        x_tics = np.array(range(self.epochs))
        x_tics = x_tics[x_tics % self.check_point_t == 0]
#         y_tics = np.array( np.arange(0, 1.1, 0.1) )
        
        ax.plot(x_tics[10:], self.median_Z[10:], color='#009cff', linestyle='solid', linewidth=3, alpha=1, 
                label=f'median Z final value: {self.median_Z[-1]:.3f}')
        
        self.plotterLayout(ax=ax, xlimits=XLIM, title='median significance evolution', titlefont=18, xlabel='training epoch', ylabel='Z', labelfont=16)
#         ax.set_yticks(y_tics)
        
        ax.legend()
        self.change_legend(ax=ax, new_loc="upper right", fontsize=14, titlesize=16)
        
        fig.tight_layout()
        if self.save_flag:
            fig.savefig(self.plotOutPath()+'_median_significance.png', dpi = 300, facecolor='white')
        plt.show()
        return
    
    
    def ks_test_evo(self):
        
        self.D_history = []
        self.D_pval_history = []
        for i in range(self.t_list_history.shape[1]):
            self.D, self.Dpval = scipy.stats.kstest(
                rvs=self.t_list_history[:,i],
                cdf="chi2",
                args=(10, 0, 1)
            )
            self.D_history.append(self.D)
            self.D_pval_history.append(self.Dpval)
        
        self.D_history = np.array(self.D_history)
        self.D_pval_history = np.array(self.D_pval_history)
        
        XMIN = 0
        XMAX = self.epochs
        YMIN = 0
        YMAX = 1.2
        
        XLIM = [XMIN, XMAX]
        
        fig, ax = plt.subplots(figsize=(12,7))

        x_tics = np.array(range(self.epochs))
        x_tics = x_tics[x_tics % self.check_point_t == 0]
        
        ax.plot(x_tics[10:], self.D_history[10:], color='#009cff', linestyle='solid', linewidth=3, alpha=1, 
                label=f'D statistic final value: {self.D:.3f}')
        
        self.plotterLayout(ax=ax, xlimits=XLIM, title='D statistic evolution', titlefont=18, xlabel='training epoch', ylabel='D', labelfont=16)
        
        ax.legend()
        self.change_legend(ax=ax, new_loc="upper right", fontsize=14, titlesize=16)
        
        fig.tight_layout()
        if self.save_flag:
            fig.savefig(self.plotOutPath()+'_D_evolution.png', dpi = 300, facecolor='white')
        plt.show()
        
        fig, ax = plt.subplots(figsize=(12,7))

       
        
        ax.plot(x_tics[10:], self.D_pval_history[10:], color='#009cff', linestyle='solid', linewidth=3, alpha=1, 
                label=f'KS final pvalue: {self.Dpval:.3f}')
        
        self.plotterLayout(ax=ax, xlimits=XLIM, title='KS p-value evolution', titlefont=18, xlabel='training epoch', ylabel='KS pval', labelfont=16)
        
        ax.legend()
        self.change_legend(ax=ax, new_loc="upper right", fontsize=14, titlesize=16)
        
        fig.tight_layout()
        if self.save_flag:
            fig.savefig(self.plotOutPath()+'_KSp_evolution.png', dpi = 300, facecolor='white')
        plt.show()
        
        return
