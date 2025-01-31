from Library_plots import get_city
import sys
import os
import numpy as np
#import sklearn.metrics
import argparse
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
#import pyart

def plot_hazard_map(**kwargs):

    matplotlib.rcParams['xtick.direction'] = 'out'
    matplotlib.rcParams['ytick.direction'] = 'out'
    matplotlib.rcParams['axes.labelsize'] = '30'
    matplotlib.rcParams['xtick.labelsize'] = '34'
    matplotlib.rcParams['ytick.labelsize'] = '34'
    matplotlib.rcParams['figure.titlesize'] = '34'
    matplotlib.rcParams['axes.titlesize'] = '18'
    matplotlib.rcParams['legend.fontsize'] = '12'
    matplotlib.rcParams['font.family'] = 'serif'
    matplotlib.rcParams['font.sans-serif'] = 'Times'
    #matplotlib.rcParams['figure.figsize'] = (12, 10)
    #plt.style.use('ggplot')
    figdpi = 150

    #wd = kwargs.get('wf_dict',None)

    #workdir = wd['workdir']
    #hc_dir = os.path.join(workdir, 'HAZARD/HazardInundationOutput')
    outdir = os.path.join(os.getcwd(),'VISUALIZATION')
    if not os.path.exists(outdir):
        os.mkdir(outdir)

    if return_time == 475:
      poe_th = 0.1
    elif return_time == 2475:
      poe_th = 0.02
    elif return_time == 9975:
      poe_th = 0.005

    #poe_thresholds = np.array([1-np.exp(-forecast_time*(arp**-1)) for arp in return_time])
 
    # loading thresholds
    mih = np.array([1.000e-02, 5.000e-02, 1.000e-01, 2.000e-01,
       3.000e-01, 5.000e-01, 8.000e-01, 1.000e+00,
       1.500e+00, 2.000e+00, 2.500e+00, 3.000e+00,
       3.600e+00, 4.320e+00, 5.180e+00, 6.220e+00,
       7.460e+00, 8.950e+00, 1.074e+01, 1.289e+01,
       1.547e+01, 1.856e+01, 2.227e+01])
    mih_th = mih[0]

    stats = 'mean'
    pathDir = "/nas/catstore2/cat/abbalice/DISAGG_MAPS/{}".format(inpdir)
 
    ci_path_old_up = os.path.join(pathDir,'ci_old_up_{}.npy'.format(percentage))
    ci_path_new_up = os.path.join(pathDir,'ci_new_up_{}.npy'.format(percentage))
    ci_path_old_dw = os.path.join(pathDir,'ci_old_dw_{}.npy'.format(percentage))
    ci_path_new_dw = os.path.join(pathDir,'ci_new_dw_{}.npy'.format(percentage))
    haz_file_disagg = os.path.join("/nas/catstore2/cat/abbalice/DISAGG_MAPS/{}".format(inpdir),'hc_mean.npy')
    haz_file_sis = os.path.join("/nas/catstore2/cat/abbalice/SIS_MAPS/{}/{}".format(inpdir, percentage),'hc_mean.npy')
    #fileBathy
    grdfile = os.path.join(pathDir,'grid_bathy.grd')
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

    fig, ax = plt.subplots(figsize=(18,15))
    
    # load hazard curves
    hc_disagg = np.load(haz_file_disagg)
    hc_sis = np.load(haz_file_sis)
    ci_new_sis_up = np.load(ci_path_new_up)
    ci_old_sis_up = np.load(ci_path_old_up)
    ci_new_sis_dw = np.load(ci_path_new_dw)
    ci_old_sis_dw = np.load(ci_path_old_dw)
  
    hmap_disagg = get_intensity(hc_disagg, mih, poe_th)
    hmap_disagg = hmap_disagg*(znew>=0) 
    hmap_sis = get_intensity(hc_sis, mih, poe_th)
    hmap_sis = hmap_sis*(znew>=0)
    
    hmap_ci_new_sis_up = get_intensity(ci_new_sis_up, mih, poe_th)
    hmap_ci_new_sis_up = hmap_ci_new_sis_up*(znew>=0)
    hmap_ci_old_sis_up = get_intensity(ci_old_sis_up, mih, poe_th)
    hmap_ci_old_sis_up = hmap_ci_old_sis_up*(znew>=0)
    
    hmap_ci_new_sis_dw = get_intensity(ci_new_sis_dw, mih, poe_th)
    hmap_ci_new_sis_dw = hmap_ci_new_sis_dw*(znew>=0)
    hmap_ci_old_sis_dw = get_intensity(ci_old_sis_dw, mih, poe_th)
    hmap_ci_old_sis_dw = hmap_ci_old_sis_dw*(znew>=0)
   
    print(hmap_ci_old_sis_up.max(), hmap_ci_new_sis_up.max())
    sorted_inds = np.argsort(hmap_disagg)

    if (inpdir=="SR") and (return_time==475):

       plt.fill_between(
           hmap_disagg[sorted_inds],
           hmap_ci_old_sis_up[sorted_inds],
           hmap_ci_new_sis_up[sorted_inds],
           where=(hmap_ci_old_sis_up[sorted_inds]<=hmap_ci_new_sis_up[sorted_inds]),
           label='95% CI',
           interpolate=True,
           color="purple",
           alpha=0.3
           )

       plt.fill_between(
           hmap_disagg[sorted_inds],
           hmap_ci_old_sis_up[sorted_inds],
           hmap_ci_old_sis_dw[sorted_inds],
           label='95% CI',
           color="gray",
           alpha=0.5
           )

       plt.fill_between(
           hmap_disagg[sorted_inds],
           hmap_ci_old_sis_dw[sorted_inds],
           hmap_ci_new_sis_dw[sorted_inds],
           where=(hmap_ci_old_sis_dw[sorted_inds]>=hmap_ci_new_sis_dw[sorted_inds]),
           interpolate=True,
           label='95% CI',
           color="purple",
           alpha=0.3
           )

    else:
       plt.fill_between(
           hmap_disagg[sorted_inds],
           hmap_ci_new_sis_up[sorted_inds],
           hmap_ci_old_sis_up[sorted_inds],
           where=(hmap_ci_new_sis_up[sorted_inds]<=hmap_ci_old_sis_up[sorted_inds]),
           label='95% CI',
           interpolate=True,
           color="gray",
           alpha=0.5
           )
    
       plt.fill_between(
           hmap_disagg[sorted_inds],
           hmap_ci_new_sis_up[sorted_inds],
           hmap_ci_new_sis_dw[sorted_inds],
           label='95% CI',
           color="purple",
           alpha=0.3
           )
    
       plt.fill_between(
           hmap_disagg[sorted_inds],
           hmap_ci_new_sis_dw[sorted_inds],
           hmap_ci_old_sis_dw[sorted_inds],
           where=(hmap_ci_new_sis_dw[sorted_inds]>=hmap_ci_old_sis_dw[sorted_inds]),
           interpolate=True,
           label='95% CI',
           color="gray",
           alpha=0.5
           )
    
    #ax.scatter(hmap_disagg, hmap_sis, s=15, edgecolor="black", color="midnightblue", alpha=0.5)
    ax.set_xlabel("Exact solution", fontsize=50)
    ax.set_ylabel("Monte Carlo estimate", fontsize=50)
    ax.set_xlim([hmap_disagg.min(), hmap_disagg.max()])
    ax.set_ylim([hmap_sis.min(), hmap_sis.max()])
    ax.plot(ax.get_xlim(), ax.get_ylim(), ls="--", lw=4, c="yellow")
    ax.scatter(hmap_disagg, hmap_sis, s=20, edgecolor="black", color="midnightblue", alpha=0.5)

    if return_time==475:
       plt.text(x=hmap_disagg.max()-0.5, y=0.5, s="N = {}".format(percentage),
             fontsize=45, color="darkblue")
       letter="a"
       plt.text(x=0.1, y=hmap_sis.max()-0.2, s="Analytical CI 95% (old IF)",
             fontsize=45, color="gray")
       plt.text(x=0.1, y=hmap_sis.max()-0.3, s="Analytical CI 95% (new IF)",
             fontsize=45, color="purple")
    elif return_time==2475:
       plt.text(x=hmap_disagg.max()-1, y=0.5, s="N = {}".format(percentage),
             fontsize=45, color="darkblue")
       letter="b"
       plt.text(x=0.1, y=hmap_sis.max()-0.3, s="Analytical CI 95% (old IF)",
             fontsize=45, color="gray")
       plt.text(x=0.1, y=hmap_sis.max()-0.5, s="Analytical CI 95% (new IF)",
             fontsize=45, color="purple")

    elif return_time==9975:
       plt.text(x=hmap_disagg.max()-2.5, y=0.5, s="N = {}".format(percentage),
             fontsize=45, color="darkblue")
       letter="c"
       plt.text(x=0.3, y=hmap_sis.max()-1, s="Analytical CI 95% (old IF)", 
             fontsize=45, color="gray")
       plt.text(x=0.3, y=hmap_sis.max()-1.35, s="Analytical CI 95% (new IF)", 
             fontsize=45, color="purple")
    
    plt.yticks(fontsize=50)
    plt.xticks(fontsize=50)
    city = get_city(inpdir)
    ax.text(-0.2, 1.08, "({})".format(letter), fontsize=50, color='black', transform=ax.transAxes,    ##000000
             bbox=dict(facecolor='#ffffff', edgecolor='black', pad=4.0))
    #ax.text(-0.1, 1.01, "({})".format(letter), fontsize=25, color='black', transform=ax.transAxes,    ##000000
    #         bbox=dict(facecolor='#ffffff', edgecolor='black', pad=6.0))
    ax.set_title('{} Flow depth [m]; ARP: {} years'.format(city, return_time), fontsize=50)
    fig.savefig(os.path.join(outdir, "ci_{}_mean_inundated_site{}_ARP{}y.png".format(percentage, inpdir, return_time)),
                    format='png', dpi=300, bbox_inches="tight")
    plt.show()
    
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

def local_parser():
    parser = argparse.ArgumentParser(description=sys.argv[0])
    parser.add_argument('--offshore_poi', default=None, required=True, help='Where local S-PTHA is evaluated (e.g. CT, SR,..)')
    parser.add_argument('--num_samples', default=None, required=True, help='Number of scenarios to be sampled from the ensemble')
    parser.add_argument('--arp', default=None, required=True, help='Average return period')

    args = parser.parse_args()

    if not sys.argv[1:]:
        print("Use -h or --help option for Help")
        sys.exit(0)
    return args
#-----------------------------------------
def from_local_parser():
    local_opts = local_parser()
    inpdir = local_opts.offshore_poi
    percentage = local_opts.num_samples
    arp = local_opts.arp
    return str(inpdir), int(percentage), int(arp)

#------------------------------------------------------------------------------------------------------------------------
inpdir, percentage, return_time = from_local_parser()

plot_hazard_map()


