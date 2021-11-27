import os
import h5py
import numpy as np
import pandas as pd
import scipy.stats
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.ticker import ScalarFormatter
import matplotlib.gridspec as gridspec
from scipy.interpolate import make_interp_spline, BSpline

import RunParameters
import TDist

def getTlist(outpath, ntoys=1):
    rPar = RunParameters.RunParameters(out_dir=outpath, ntoy=ntoys)
    toys, w_clip, epochs, check_point_t, ref, bkg, sig, latent, layers = rPar.fetch_parameters()
#     rPar.print_parameters()

    OUT_FILE_t = rPar.fetch_file()
    OUT_FILE_t_history = rPar.fetch_history()

    tDist = TDist.TDist(toys, epochs, ref, bkg, sig, check_point_t, w_clip, outpath, OUT_FILE_t, OUT_FILE_t_history)
    t_list, t_history = tDist()
    
    return t_list, t_history




def change_legend(ax, new_loc, fontsize, titlesize, **kws):
        '''funzione per modificare posizione e font size della legenda generata da seaborn'''

        old_legend = ax.legend_
        handles = old_legend.legendHandles
        labels = [t.get_text() for t in old_legend.get_texts()]
        title = old_legend.get_title().get_text()
        ax.legend(handles, 
                  labels, 
                  loc=new_loc, 
                  title=title, 
                  fontsize=fontsize, 
                  title_fontsize=titlesize, 
                  frameon = True, 
                  fancybox = False, 
                  framealpha = 1, 
    #                  bbox_to_anchor=(1.3, 1.0),
                  **kws)
        return


def plotTdist(t_lists, t_list_ref, bins=7, ref_bins=8, dof=10, plot_name=None, xmax=None):
    '''grafico della distribuzione dei t'''
    
    plt.rcParams["patch.force_edgecolor"] = True
    
    RUNS = [1253, 1265, 1242]

    
    # creo la griglia lungo x
    XMIN = 0
    XMAX = max(t_list_ref) + min(t_lists[-1])
    if  xmax:
        XMAX = xmax
        
    XGRID = np.linspace(XMIN, XMAX, 500)
    
    # creo figure&axes
    fig, ax = plt.subplots(figsize=(18,9))
    
    ########################################
    
    hist, bin_edges = np.histogram(t_list_ref, density=True, bins=ref_bins)

    binswidth = bin_edges[1]-bin_edges[0]
    central_points = [
        (bin_edges[i] + bin_edges[i + 1]) / 2 for i in range(len(bin_edges) - 1)
    ]
    err = np.sqrt(hist/(t_list_ref.shape[0]*binswidth))
    
    th_median = scipy.stats.chi2.median(df=dof)

    ax.plot(
        XGRID, 
        scipy.stats.chi2.pdf(XGRID, df=dof), 
        color='#005e99', 
        linestyle='solid', 
        linewidth=7, 
        alpha=0.6, 
        label=r'Target $\chi^{2}_{10}$'+f"\nmedian = {th_median:.2f}\n"
    )

    sns.histplot(
        x=bin_edges[:-1], weights=hist, bins=bin_edges,
        stat='density', element='bars', linewidth=2,
        fill=True, color='#aadeff', edgecolor='#009cff', 
        ax=ax
    )

    ax.errorbar(central_points, hist, yerr=err, color='#009cff', linewidth=2, marker='o', ls='')
    
    ########################################
    
    colors = ["#DC3522", "#00b32a", "#005e99"]
    for i in range(len(t_lists)):
        ax.axvline(
            x=t_lists[i][-1],
            color = colors[i], 
            linestyle='dashed', 
            linewidth = 5, 
            alpha = 1, 
            label = 't_{}'.format(RUNS[i])+'={:.2f}'.format(t_lists[i][-1])
        )
        if i == 0:
            ax.text(
                    t_lists[i][-1]+10, 0.08, 
                    f"RUN{RUNS[i]}\nt = {t_lists[i][-1]:.2f}", 
                    horizontalalignment='left', verticalalignment='center', 
                    color=colors[i],
                    fontsize=22,
                    fontweight="bold",
                    transform=ax.transData)
        else:
            ax.text(
                    t_lists[i][-1]-10, 0.08, 
                    f"RUN{RUNS[i]}\nt = {t_lists[i][-1]:.2f}", 
                    horizontalalignment='right', verticalalignment='center', 
                    color=colors[i],
                    fontsize=22,
                    fontweight="bold",
                    transform=ax.transData)

    ax.set_title(f'Observed test statistics', fontsize = 32)
    ax.set_xlabel('t', fontsize = 28)
    ax.set_ylabel('p(t)', fontsize = 28)

    ax.set_xlim(XMIN, XMAX)
    ax.set_ylim(0, 0.11)

    ax.tick_params(axis = 'both', which = 'major', labelsize = 24, direction = 'out', length = 5)
    ax.yaxis.get_offset_text().set_fontsize(24)
    ax.yaxis.set_major_formatter(ScalarFormatter(useMathText=True))
    ax.ticklabel_format(axis = 'y', style = 'sci', scilimits = (0,0))
    
    # gestione della legenda
#     ax.legend()
#     change_legend(ax=ax, new_loc="upper right", fontsize=22, titlesize=0)
    
    fig.tight_layout()
    if plot_name:
        fig.savefig(
            f'/lustre/cmswork/nlai/PLOTS/DRIFT_TIME/thesis/{plot_name}.pdf', 
            facecolor = 'white'
        )
        
    plt.show()
    
    return 




#######################################################################################
def plotTdist2(t_lists, t_list_ref, ref_bins=8, dof=10, plot_name=None, ymax=None):
    
    plt.rcParams["patch.force_edgecolor"] = True
    
    RUNS = [1253, 1265, 1242]

    # FIGURE CONFIGURATION
    ########################################
    fig = plt.figure(figsize=(15,8))
    fig.suptitle(
        t=f'Observed test statistics', 
#         x=0.5,
#         y=0.93,
        fontsize=28
    )
    gs = gridspec.GridSpec(1, 4)
    
    gs.update(wspace=0, hspace=0)
    
    ax = plt.subplot(gs[:, :-1])
    ax_hist = plt.subplot(gs[:, -1], sharey=ax)
    ax_hist.set_axis_off()
    
    XMIN = 0
    XMAX = 20 * 1e3
    x_tics = np.arange(0, XMAX, 1000)
    x_fine = np.linspace(0, XMAX, 1000)
    
    YMAX = 210
    if ymax:
        YMAX = ymax
    
    ax.set_xlim(XMIN, XMAX - 1e3)
    ax.set_ylim(0, YMAX)
    
    
    ax.set_xlabel('training epochs', fontsize = 24)
    ax.set_ylabel('t', fontsize = 24)

    ax.tick_params(axis = 'both', which = 'major', labelsize = 20, direction = 'out', length = 5)
#     ax.yaxis.get_offset_text().set_fontsize(24)
#     ax.yaxis.set_major_formatter(ScalarFormatter(useMathText=True))
#     ax.ticklabel_format(axis = 'y', style = 'sci', scilimits = (0,0))
#     start, end = ax.get_xlim()
#     ax.xaxis.set_ticks(np.arange(start, 110000, 5000))
    plt.setp(ax.get_xticklabels()[1::2], visible=False)
    
    XGRID = np.linspace(0, 50, 500)
    ax_hist.set_xlim(0, 0.11)
    
    ########################################
    
    # THEORETICAL MEDIAN OF CHI2
    ########################################
    th_median = scipy.stats.chi2.median(df=dof)
    ax.axhline(
        y=th_median,
        color = "#009cff", 
        linestyle='dashed', 
        linewidth = 5, 
        alpha = 1, 
        label=r"Target $\chi^{2}_{10}$" + "\n" + r"$\tilde{t}$" + f" = {th_median:.2f}\n"
    )
    ########################################
    
    # TOBS PLOT
    ########################################
    colors = ["#00b32a", "#FF8400", "#e52a16"]
    for i, run in enumerate(RUNS):
        spl = make_interp_spline(
            x_tics[::1], 
            t_lists[i][1:21:1], 
            k=2
        )
        smooth_t = spl(x_fine)
        ax.plot(
#             x_fine,
            x_tics,
#             smooth_t,
            t_lists[i][1:21],
            color = colors[i], 
            linestyle='solid', 
            linewidth = 5, 
            label = f"RUN {run}\nt = {t_lists[i][-1]:.2f}\n"
        )
    ########################################
    
    # SIDE HISTOGRAM
    ########################################
    hist, bin_edges = np.histogram(t_list_ref, density=True, bins=ref_bins)

    binswidth = bin_edges[1]-bin_edges[0]
    central_points = [
        (bin_edges[i] + bin_edges[i + 1]) / 2 for i in range(len(bin_edges) - 1)
    ]
    err = np.sqrt(hist/(t_list_ref.shape[0]*binswidth))
    ########################################
    
    # CHI2 PLOT
    ########################################
    ax_hist.plot(
        scipy.stats.chi2.pdf(XGRID, df=dof), 
        XGRID,
        color='#005e99', 
        linestyle='solid', 
        linewidth=4, 
        alpha=0.5
    )
    ########################################
    
    # HISTOGRAM PLOT
    ########################################
    sns.histplot(
        y=bin_edges[:-1], weights=hist, bins=bin_edges,
        stat='density', element='bars', linewidth=2,
        fill=True, color='#aadeff', edgecolor='#009cff', 
        ax=ax_hist
    )
    ax_hist.errorbar(
        central_points, 
        hist, 
        yerr=err, 
        color='#009cff', 
        linewidth=2, 
        marker='o', 
        ls=''
    )
    ########################################
    
    # LEGEND
    ########################################
    ax_lines, ax_labels = ax.get_legend_handles_labels()
    ax_hist.legend(ax_lines[::-1], ax_labels[::-1])
    change_legend(
        ax=ax_hist, 
        new_loc="upper center", 
        fontsize=18, 
        titlesize=0
    )
#     ax.legend()
#     change_legend(ax=ax, new_loc="upper right", fontsize=22, titlesize=0)
    
    fig.tight_layout()
    ########################################
    
    # SAVE FIGURE
    ########################################
    if plot_name:
        fig.savefig(
            f'/lustre/cmswork/nlai/PLOTS/DRIFT_TIME/thesis/{plot_name}.pdf', 
            facecolor = 'white'
        )
    ########################################
        
    plt.show()
    return 
    

