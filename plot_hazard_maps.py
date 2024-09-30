import sys
import os
import numpy as np
#import sklearn.metrics
import csv
import cartopy
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
from matplotlib import ticker
from matplotlib.colors import LightSource
from mpl_toolkits.axes_grid1 import make_axes_locatable
import pickle
# from scipy.io import netcdf
# import h5py
import xarray as xr
import pyart

import os
import sys
import argparse

#==========================================================================================================================
def local_parser():
    parser = argparse.ArgumentParser(description=sys.argv[0])
    parser.add_argument('--location', default=None, required=True, help='Where local S-PTHA is evaluated (e.g. CT, SR,..)')
    parser.add_argument('--percentage', default=None, required=True, help='Number of samples')
    #load arguments
    args = parser.parse_args()
    #if any
    if not sys.argv[1:]:
        print("Use -h or --help option for Help")
        sys.exit(0)
    return args
#-----------------------------------------
def from_local_parser():
    local_opts = local_parser()
    inpdir = local_opts.location
    percentage = local_opts.percentage
    return str(inpdir), int(percentage)
#===========================================================================================================================
inpdir, percentage = from_local_parser()



def find_difference(a, b):
    diff = []
    for i in range(len(a)):
      if b[i]!=0:
        diff.append(((a[i]-b[i])/b[i])*100)
      else:
        diff.append(np.nan)
    diff = np.array(diff)
    return diff

def plot_hazard_map(**kwargs):

    matplotlib.rcParams['xtick.direction'] = 'out'
    matplotlib.rcParams['ytick.direction'] = 'out'
    matplotlib.rcParams['axes.labelsize'] = '34'
    matplotlib.rcParams['xtick.labelsize'] = '34'
    matplotlib.rcParams['ytick.labelsize'] = '34'
    matplotlib.rcParams['figure.titlesize'] = '34'
    matplotlib.rcParams['axes.titlesize'] = '36'
    matplotlib.rcParams['legend.fontsize'] = '25'
    matplotlib.rcParams['font.family'] = 'serif'
    matplotlib.rcParams['font.sans-serif'] = 'Times'

    figdpi = 300
    outdir = os.path.join(os.getcwd(),'VISUALIZATION')
    if not os.path.exists(outdir):
        os.mkdir(outdir)

    return_time = np.array([475, 2475, 9975])
    forecast_time = 50
    poe_thresholds = np.array([0.1, 0.02, 0.005])
    # loading thresholds
    mih = np.array([1.000e-02, 5.000e-02, 1.000e-01, 2.000e-01,
       3.000e-01, 5.000e-01, 8.000e-01, 1.000e+00,
       1.500e+00, 2.000e+00, 2.500e+00, 3.000e+00,
       3.600e+00, 4.320e+00, 5.180e+00, 6.220e+00,
       7.460e+00, 8.950e+00, 1.074e+01, 1.289e+01,
       1.547e+01, 1.856e+01, 2.227e+01])
    mih_th = mih[0]

    stats = ['p16', 'mean', 'p84'] 
    hc_dir_sis = "SIS_MAPS/{}/{}".format(inpdir, percentage)
    hc_dir_disagg = "DISAGG_MAPS/{}".format(inpdir)

    haz_files_sis = [os.path.join(hc_dir_sis,'hc_' + stat + '.npy') for stat in stats]
    haz_files_disagg = [os.path.join(hc_dir_disagg,'hc_' + stat + '.npy') for stat in stats]  
    #fileBathy
    grdfile = os.path.join(hc_dir_disagg,'grid_bathy.grd')
    fgrid = xr.open_dataset(grdfile)
    xgrid = fgrid['x'].values
    ygrid = fgrid['y'].values
    zgrid = fgrid['z'].values
    znew = zgrid.flatten()
    xmin = np.amin(xgrid)
    xmax = np.amax(xgrid)
    ymin = np.amin(ygrid)
    ymax = np.amax(ygrid)
    zmin = np.amin(zgrid)
    zmax = np.amax(zgrid)
    z = 0.5*(zmax-zmin)

    ratio=(ygrid.shape[0]*2)/(xgrid.shape[0]*3)
    xsize=22
    ysize=xsize*ratio
    
    ls = LightSource(azdeg=315, altdeg=45)
    cmap = plt.cm.gray
    rgb = ls.shade(zgrid, cmap=cmap, vert_exag=10, blend_mode='overlay', vmin=-z, vmax=z)
    
    
    colors = ["midnightblue"]
    maplevels=np.concatenate((np.array([0.01,0.05,0.1,0.2]),np.linspace(0.5, 6.0, 12)),axis=0)
    perclevels = np.arange(-100, 120, 20)
    for arp, poe_th in enumerate(poe_thresholds):
      rows = ['Disaggregation', 'SIS', 'Rel. perc. diff.']
 
      fig1 = plt.figure(figsize=(xsize,ysize))
    
      if (inpdir=="SR"):
        fig2, axes = plt.subplots(nrows=3, ncols=3, sharex=True, sharey=True, figsize=(20.5,15), constrained_layout=True)
      
      else:
        fig2, axes = plt.subplots(nrows=3, ncols=3, sharex=True, sharey=True, figsize=(xsize,ysize), constrained_layout=True)
      for ic, stat in enumerate(stats):

        # load hazard curves
        hc_disagg = np.load(haz_files_disagg[ic])
        if inpdir=="LK":
          hc_disagg = 1-np.exp(-hc_disagg*50)
       
        hc_sis = np.load(haz_files_sis[ic])
        
        hmap_disagg = get_intensity(hc_disagg, mih, poe_th)
        hmap_disagg = hmap_disagg*(znew>=0)
        
        hmap_sis = get_intensity(hc_sis, mih, poe_th)
        hmap_sis = hmap_sis*(znew>=0)
        
        rpd = find_difference(hmap_disagg, hmap_sis)
        
        print("ARP {}y, stat. {}, N = {} ---> rpd {}".format(return_time[arp], stat, percentage, np.nanmean(rpd)))
        hmap_disagg = np.reshape(hmap_disagg, (ygrid.shape[0], xgrid.shape[0]))
        hmap_sis = np.reshape(hmap_sis, (ygrid.shape[0], xgrid.shape[0]))
        rpd = np.reshape(rpd, (ygrid.shape[0], xgrid.shape[0]))
        
        #plots
        ax1 = fig1.add_subplot(2, 3, ic+1)
        ax1.imshow(rgb, origin="lower", aspect="equal", alpha=0.75,
                   extent=(xmin, xmax, ymin, ymax))
        ax1.title.set_text(stat) #stat
        ax1.contourf(xgrid, ygrid, hmap_disagg, np.array([mih_th, mih[-1]]),
                        colors=colors[0], alpha=0.75, extend="neither",
                        extent=(xmin, xmax, ymin, ymax))
        ax1.contour(xgrid, ygrid, hmap_disagg, np.array([mih_th]),
                       colors=colors[0], alpha=1, extend="neither",
                       extent=(xmin, xmax, ymin, ymax),
                       linewidths=1.0, linestyles="solid")
        if (inpdir=="CT") or (inpdir=="SR"):
           textstr = 'Dis. (53550 scen.)'
        else:
           textstr = 'Dis. (23165 scen.)'
        # these are matplotlib.patch.Patch properties
        props = dict(boxstyle='round', facecolor='white', alpha=1)
        # place a text box in upper left in axes coords
        ax1.text(0.05, 0.95, textstr, transform=ax1.transAxes, fontsize=25,
        verticalalignment='top', bbox=props)
        ax1.set_xticks([])
        if (ic==0) and (ic+1==1):
           ax1.set_ylabel(r'Latitude ($^\circ$)', fontsize=34)
        #ax1.set_aspect('auto')

        ax1 = fig1.add_subplot(2, 3, ic+4)
        ax1.imshow(rgb, origin="lower", aspect="equal", alpha=0.75,
                   extent=(xmin, xmax, ymin, ymax))
        #ax1.title.set_text(stat) #stat
        ax1.contourf(xgrid, ygrid, hmap_sis, np.array([mih_th, mih[-1]]),
                   colors=colors[0], alpha=0.75, extend="neither",
                   extent=(xmin, xmax, ymin, ymax))
        ax1.contour(xgrid, ygrid, hmap_sis, np.array([mih_th]),
                   colors=colors[0], alpha=1, extend="neither",
                   extent=(xmin, xmax, ymin, ymax),
                   linewidths=1.0, linestyles="solid")
        #n_sis=6000 
        textstr = 'SIS ({} scen.)'.format(n_sis)
        
        # these are matplotlib.patch.Patch properties
        props = dict(boxstyle='round', facecolor='white', alpha=1)
        # place a text box in upper left in axes coords
        ax1.text(0.05, 0.95, textstr, transform=ax1.transAxes, fontsize=25,
        verticalalignment='top', bbox=props)
        if (ic==0):
           ax1.set_ylabel(r'Latitude ($^\circ$)', fontsize=34)
           ax1.set_xlabel(r'Longitude ($^\circ$)', fontsize=34)
        elif (ic>0) and (ic+4>4):
           ax1.set_xlabel(r'Longitude ($^\circ$)', fontsize=34)
        #ax1.set_aspect('auto')
        #------------------------------       
        ax2 = axes.flat[ic]
        ax3 = axes.flat[ic+3]
        ax4 = axes.flat[ic+6]

        ax2.imshow(rgb, origin="lower", aspect="equal", alpha=0.75,
                   extent=(xmin, xmax, ymin, ymax))
        cc = ax2.contourf(xgrid, ygrid, hmap_disagg, maplevels,
                   cmap="pyart_SpectralExtended", alpha=1, extend="max",  #plt.cm.viridis
                   extent=(xmin, xmax, ymin, ymax))
        ax2.contour(xgrid, ygrid, hmap_disagg, maplevels,
                   cmap="pyart_SpectralExtended", alpha=1, extend="max",   #plt.cm.viridis
                   extent=(xmin, xmax, ymin, ymax),
                   linewidths=1.0, linestyles="solid")
        
        if (inpdir=="SR") and (percentage==3000) and (arp==1): 
           ax2.set_xlim([15.25, xmax])
 
        ax2.title.set_text(stat)
        if (inpdir=="CT") or (inpdir=="SR"):
           textstr = 'Dis. (53550 scen.)'
        else:
           textstr = 'Dis. (23165 scen.)'
        # these are matplotlib.patch.Patch properties
        props = dict(boxstyle='round', facecolor='white', alpha=1)
        # place a text box in upper left in axes coords
        ax2.text(0.05, 0.95, textstr, transform=ax2.transAxes, fontsize=25,
        verticalalignment='top', bbox=props)
        #====================================================================
        ax3.imshow(rgb, origin="lower", aspect="equal", alpha=0.75,
                   extent=(xmin, xmax, ymin, ymax))
        cc = ax3.contourf(xgrid, ygrid, hmap_sis, maplevels,
                   cmap="pyart_SpectralExtended", alpha=1, extend="max",  #plt.cm.tab20b
                   extent=(xmin, xmax, ymin, ymax))
        ax3.contour(xgrid, ygrid, hmap_sis, maplevels,
                   cmap="pyart_SpectralExtended", alpha=1, extend="max",   #plt.cm.viridis
                   extent=(xmin, xmax, ymin, ymax),
                   linewidths=1.0, linestyles="solid")

        if (inpdir=="SR") and (percentage==3000) and (arp==1):
           ax3.set_xlim([15.25, xmax])      
        textstr = 'SIS ({} scen.)'.format(n_sis)

        # these are matplotlib.patch.Patch properties
        props = dict(boxstyle='round', facecolor='white', alpha=1)
        # place a text box in upper left in axes coords
        ax3.text(0.05, 0.95, textstr, transform=ax3.transAxes, fontsize=25,
        verticalalignment='top', bbox=props)
        #====================================================================
        ax4.imshow(rgb, origin="lower", aspect="equal", alpha=0.75,
                   extent=(xmin, xmax, ymin, ymax))
        rpd = ax4.contourf(xgrid, ygrid, rpd, perclevels,
                   cmap="pyart_balance", alpha=1, extend="max",  #plt.cm.plasma
                   extent=(xmin, xmax, ymin, ymax))
        CS = ax4.contour(xgrid, ygrid, hmap_disagg, [0.2],
                   colors="black", alpha=1, extend="max",   #plt.cm.viridis
                   extent=(xmin, xmax, ymin, ymax),
                   linewidths=0.3, linestyles="solid")

        if (inpdir=="SR"):
           ax4.set_xlim([15.25, xmax]) 
        textstr = 'Rel. perc. diff.'
        # these are matplotlib.patch.Patch properties
        props = dict(boxstyle='round', facecolor='white', alpha=1)
        # place a text box in upper left in axes coords
        ax4.text(0.05, 0.95, textstr, transform=ax4.transAxes, fontsize=25,
        verticalalignment='top', bbox=props)
        
        cols = axes.flat
        cols[6].set_xlabel(r'Longitude ($^\circ$)', fontsize=34)
        cols[7].set_xlabel(r'Longitude ($^\circ$)', fontsize=34)
        cols[8].set_xlabel(r'Longitude ($^\circ$)', fontsize=34)
        cols[0].set_ylabel(r'Latitude ($^\circ$)', fontsize=34)
        cols[3].set_ylabel(r'Latitude ($^\circ$)', fontsize=34)
        cols[6].set_ylabel(r'Latitude ($^\circ$)', fontsize=34)
 
      cbar2 = fig2.colorbar(cc, ax=axes[:2, -1], shrink=0.8, aspect=34)
      cbar2.ax.tick_params(labelsize=30)
      cbar2.set_label(label="FD [m]", size=34)
      cbar2 = fig2.colorbar(rpd, ax=axes[-1, -1])
      cbar2.ax.tick_params(labelsize=30) 
      cbar2.set_label(label="RPD [%]", size=34)
      fig1.tight_layout()
      
      if inpdir=="CT":
         city = "Catania"
      elif inpdir=="SR":
         city = "Siracusa"
      elif inpdir=="LK":
         city = "Larnaka"
      
      fig1.suptitle('{} Inundated area; ARP: {} years; N = {}'.format(city, return_time[arp], percentage), fontsize=36, y=1.035)
      
      fig2.suptitle('{} Flow depth [m]; ARP: {} years; N = {}'.format(city, return_time[arp], percentage), fontsize=36, y=1.035)
      fig1.savefig(os.path.join(outdir, "hazard_maps_inundation_ARP{}y_{}_{}.png".format(return_time[arp], inpdir, percentage)),  
                                format='png',  bbox_inches="tight")#dpi=300, bbox_inches="tight")
      fig2.savefig(os.path.join(outdir, "hazard_maps_flowdepth_ARP{}y_{}_{}.png".format(return_time[arp], inpdir, percentage)),
                   format='png',  bbox_inches="tight")#, dpi=300, bbox_inches="tight")
#    plt.show()

def get_intensity(hc, mih, p_th):
    """
    """
    n_points, n_mih = np.shape(hc)
    z = np.zeros(n_points)
    for i in range(n_points):
        curve = hc[i,:]
        # finding intensity value corresponding to probability threshold (pth)
        if (p_th < curve[0] and p_th > curve[-1] ):
            for j in range(len(curve)):
                if (curve[j] < p_th):
                    interp = (mih[j]-mih[j-1]) * (p_th-curve[j-1]) / (curve[j]-curve[j-1])
                    z[i] = mih[j-1] + interp
                    break
        elif (p_th >= curve[0]):
            z[i] = 0
        elif (p_th <= curve[-1]):
            z[i] = 0#mih[-1]
        else:
            pass

    return z

#inpdir = "SR"#"CT"#"SR"
if percentage==1500:
     if inpdir=="CT":
        n_sis = 925 
     elif inpdir=="SR":
        n_sis = 1221
     elif inpdir=="LK":
        n_sis = 711 
elif percentage==3000:
     if inpdir=="CT":
        n_sis = 1917 
     elif inpdir=="SR":
        n_sis = 2346
     elif inpdir=="LK":
        n_sis = 1019
elif percentage==6000:
     if inpdir=="CT":
        n_sis = 3198
     elif inpdir=="SR":
        n_sis = 3964
     elif inpdir=="LK":
        n_sis = 1364


plot_hazard_map()
