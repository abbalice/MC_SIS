from Library_sampling import exceedance_curve, get_parameters, mc_sis#, update_rates_sis
from Library_plots import get_sis_data_onshore, get_city
import numpy as np
import argparse
import os
import sys
import pandas as pd
import seaborn as sns
from scipy import stats
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from mpl_toolkits.basemap import Basemap
from matplotlib.colors import LightSource
import cartopy
from cartopy import config
import cartopy.crs as ccrs
from cartopy.mpl.gridliner import LONGITUDE_FORMATTER, LATITUDE_FORMATTER
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import matplotlib
from numba import njit
import types
#=================================================================================================
def bottom_offset(self, bboxes, bboxes2):
    bottom = self.axes.bbox.ymin
    self.offsetText.set(va="top", ha="left", fontsize=60)
    oy = bottom - pad * self.figure.dpi / 72.0
    self.offsetText.set_position((1, oy))

@njit
def single_exceedance_curve(thresholds, max_stage_point, rates):
    
    # Annual Mean Rate of threshold exceedance
    lambda_exc = np.zeros_like(thresholds)
    for ith, threshold in enumerate(thresholds):
        ix_sel = np.argwhere(max_stage_point > threshold)[:]
        ix_sel = ix_sel[:,0]
        lambda_exc[ith] += rates[ix_sel].sum()
    #------------------------------------------------
    return lambda_exc

@njit
def update_rates_sis(mw_bins, mw_ensemble, mw_sis, annual_rates_ensemble, annual_rates_sis, proposal_offshore, proposal_sis, n_mw_sis):
    lmw_over_nmw = np.zeros((len(mw_sis),))
    phi = np.zeros((len(mw_sis),)) 
    for imw, mw_bin in enumerate(mw_bins):
       iscen_sampled = np.argwhere(mw_sis == mw_bin)[:]
       iscen_ensemble = np.argwhere(mw_ensemble == mw_bin)[:]
       iscen_sampled, iscen_ensemble = iscen_sampled[:,0], iscen_ensemble[:,0]
       tot_rates_bin = annual_rates_ensemble[iscen_ensemble].sum()
       tot_proposal_bin = proposal_offshore[iscen_ensemble].sum()
       n_bin = n_mw_sis[imw]
       lmw_over_nmw[iscen_sampled] = tot_rates_bin / n_bin
       phi[iscen_sampled] = (annual_rates_sis[iscen_sampled] / annual_rates_ensemble[iscen_ensemble].sum()) / (proposal_sis[iscen_sampled] / tot_proposal_bin)
    new_rates_sis = lmw_over_nmw * phi
    new_rates_sis[np.isnan(new_rates_sis)]=0
    return new_rates_sis


@njit
def mc_simulation(thresholds, n_iter, percentage, mw, rates, hmax_on, proposal_offshore, n_sis):
    lambda_sis = np.zeros((len(thresholds), n_iter))
    emp_sis_up = np.zeros((len(thresholds), n_iter))
    emp_sis_dw = np.zeros((len(thresholds), n_iter))
    var_sis    = np.zeros((len(thresholds), n_iter))

    for iteration in range(n_iter):
      # Importance sampling offshore
      scenum = percentage
      ix_sis, _ = mc_sis(mw, mw_bins, proposal_offshore, n_sis)
      
      mw_sis = mw[ix_sis]
      rates_sis = rates[ix_sis]
      hmax_sis = hmax_on[ix_sis]
      proposal_sis = proposal_offshore[ix_sis]

      rates_sis = update_rates_sis(mw_bins, mw, mw_sis, rates, rates_sis, proposal_offshore, proposal_sis, n_sis)
      mean_rates_sis = rates_sis
      tmp_lambda_sis = single_exceedance_curve(thresholds, hmax_sis, mean_rates_sis)

      lambda_sis[:, iteration] = tmp_lambda_sis
    return lambda_sis

#=================================================================
def cond_emp_variance_importance(mw_ensemble, mw_sampled, mw_bins, annual_rates_sampled, annual_rates_ensemble, hmax_poi_sampled, hmax_poi_ensemble, distance,
                                 hmax_sampled_offshore, distance_sampled, threshold, n_mw_sampled):
    cond_emp_variance = np.zeros_like(mw_bins)
    cond_lambda = np.zeros_like(mw_bins)
    for imw, mw_bin in enumerate(mw_bins):
        tmp_q = 0
        tmp_variance = 0
        iscen_sampled = np.argwhere(mw_sampled == mw_bin)[:]
        iscen_ensemble = np.argwhere(mw_ensemble == mw_bin)[:].squeeze()
        if np.size(iscen_sampled, axis=1)>1:
            iscen_sampled = iscen_sampled.squeeze()
        else:
            iscen_sampled = iscen_sampled.squeeze(axis=1)
        tot_rates_bin = annual_rates_ensemble[iscen_ensemble].sum() 
        importance = hmax_poi_ensemble[iscen_ensemble]*distance[iscen_ensemble]
        proposal_ensemble = importance*annual_rates_ensemble[iscen_ensemble]

        tot_proposal_bin = proposal_ensemble.sum()
        n_bin = n_mw_sampled[imw]
        if n_bin!=0 and tot_proposal_bin!=0 and tot_rates_bin!=0:
            for ix in iscen_sampled:
              if hmax_poi_sampled[ix] > threshold:
                w_ss = annual_rates_sampled[ix]/tot_rates_bin
                importance = hmax_sampled_offshore[ix]*distance_sampled[ix]
                w_sis = (importance*annual_rates_sampled[ix])/tot_proposal_bin
                tmp_fi = w_ss/w_sis
                tmp_q += tmp_fi
            q = tmp_q/n_bin
            cond_lambda[imw] = tot_rates_bin*q
            for ix in iscen_sampled:
                if hmax_poi_sampled[ix] > threshold:
                   w_ss = annual_rates_sampled[ix]/tot_rates_bin
                   importance = hmax_sampled_offshore[ix]*distance_sampled[ix]
                   w_sis = (importance*annual_rates_sampled[ix])/tot_proposal_bin
                   tmp_fi = w_ss/w_sis
                else:
                   tmp_fi = 0
                tmp_variance += ((tmp_fi - q)**2)/n_bin
            cond_emp_variance[imw] = tmp_variance*(tot_rates_bin**2)/n_bin
    return cond_emp_variance.sum(), cond_lambda.sum()

def empirical_variance_importance(mw_ensemble, mw_sampled, mw_bins, annual_rates_sampled, annual_rates_ensemble, hmax_poi_sampled, hmax_poi_ensemble, distance,
                                  hmax_sampled_offshore, distance_sampled, thresholds, n_mw_sampled):
    nthresh = len(thresholds)
    variance_mc = np.zeros((nthresh,))
    lambda_mc = np.zeros((nthresh,))
    for ith, threshold in enumerate(thresholds):
        tmp_variance, tmp_lambda = cond_emp_variance_importance(mw_ensemble, mw_sampled, mw_bins, annual_rates_sampled,
                                                                annual_rates_ensemble, hmax_poi_sampled, hmax_poi_ensemble, distance, hmax_sampled_offshore, distance_sampled,
                                                                threshold, n_mw_sampled)
        variance_mc[ith] = np.sqrt(tmp_variance)
        lambda_mc[ith] = tmp_lambda
   # Confidence Interval-----------------------------------------------------------
    # To check for the performance of the Monte Carlo simulation
    empirical95_up = lambda_mc + 1.96 * variance_mc
    empirical95_dw = lambda_mc - 1.96 * variance_mc
    return empirical95_up, empirical95_dw, lambda_mc, variance_mc

#----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------



def normal(mean, std):
    x = np.linspace(mean-4*std, mean+4*std, 200)
    p = stats.norm.pdf(x, mean, std)
    return x, p
#-----------------------------------------------Inizialization of arguments--------------------------------------------
def local_parser():
    parser = argparse.ArgumentParser(description=sys.argv[0])
    parser.add_argument('--offshore_poi', default=None, required=True, help='Where local S-PTHA is evaluated (e.g. CT, SR,..)')
    parser.add_argument('--test_site', default=None, required=True, help='Where local S-PTHA is evaluated (e.g. CT, SR,..)')
    parser.add_argument('--mc_samples', default=None, required=True, help='Number of Monte Carlo samples')
    #parser.add_argument('--num_samples', default=None, required=True, help='Number of scenarios to be sampled from the ensemble')
    parser.add_argument('--outdir', default=None, required=True, help='Name of the directory for the outputs')

    args = parser.parse_args()

    if not sys.argv[1:]:
        print("Use -h or --help option for Help")
        sys.exit(0)
    return args
#-----------------------------------------
def from_local_parser():
    local_opts = local_parser()
    inpdir = local_opts.offshore_poi
    test_site = local_opts.test_site
    n_iter = local_opts.mc_samples
    #percentage = local_opts.num_samples
    outdir = local_opts.outdir
    #return str(inpdir), int(test_site), int(n_iter), int(percentage), str(outdir)
    return str(inpdir), int(test_site), int(n_iter), str(outdir)
#------------------------------------------------------------------------------------------------------------------------
#inpdir, test_site, n_iter, percentage, outdir = from_local_parser()
inpdir, test_site, n_iter, outdir = from_local_parser()
#-------------------------------------------------------------------------------------------------------------------------
matplotlib.rcParams['font.family'] = 'serif'
matplotlib.rcParams['font.sans-serif'] = 'Times'

# Defining the directory for storing the results
#-------------------------------------------------------------------------------------------------------------------------
#outdir = outdir+'_perc' + str(percentage)+ '_{}'.format(inpdir)
outdir = outdir + '_{}'.format(inpdir)
#-------------------------------------------------------------------------------------------------------------------------
# Defining directory for outputs
try:
  os.makedirs(os.path.join("MC_RESULTS", outdir, inpdir))
except:
  os.path.exists(os.path.join("MC_RESULTS", outdir, inpdir))
  print("Directory for outputs already there!")
#----------------------------------------------------------------------------
outdir = os.path.join("MC_RESULTS", outdir, inpdir)
#---------------------------
# Coordinates of SIS POIs
if inpdir=="CT":
  poi_off = pd.read_csv('sis_poi_catania.txt', engine='python', header=None, index_col=False).to_numpy().squeeze()
  region_map_CT = [15, 15.17, 37.3, 37.550]
  city = "Catania"
elif inpdir=="SR":
  poi_off = pd.read_csv('sis_poi_siracusa.txt', engine='python', header=None, index_col=False).to_numpy().squeeze()
  region_map_SR = [15.15, 15.4, 37, 37.2]
  city = "Siracusa"

poi_on = pd.read_csv('HMAX_ONSHORE/poi_{}_{}_coord.txt'.format(test_site, inpdir), engine='python', header=None, index_col=False).to_numpy().squeeze()
lon_sis, lat_sis = poi_off[0], poi_off[1]
lon_on, lat_on = poi_on[0], poi_on[1]
#---------------------------
# Retrieve offshore height
pathHeight_OFF = 'HMAX_OFFSHORE/hmax_{}.txt'.format(inpdir)
data_off = pd.read_csv(pathHeight_OFF, engine='python', sep='\s+')
hmax_off = data_off["MIH"].to_numpy()
ids_off = data_off["IDs_sim"].to_numpy()

# Retrieve source parameters
mw, lon, lat = get_parameters(ids_off)

# Retrieve scenario rates
#if (inpdir == "CT") or (inpdir=="SR"):
fileRates_bs = "probs_BS_Sicily.txt"
fileRates_ps = "probs_PS_Sicily.txt"

prob_bs = pd.read_csv(fileRates_bs, engine='python', sep=',', header=None, index_col=False)
prob_ps = pd.read_csv(fileRates_ps, engine='python', sep=',', header=None, index_col=False)

id_bs = prob_bs.loc[:, 0].to_numpy(dtype='str')
id_ps = prob_ps.loc[:, 0].to_numpy(dtype='str')

rates_bs = prob_bs.loc[:,1:].to_numpy()
rates_ps = prob_ps.loc[:,1:].to_numpy()

# Check
ids_scen = np.concatenate((id_bs, id_ps), axis=0)

if (ids_off == ids_scen).all():
   pass
else:
   raise Exception("Data do not coincide")

rates = np.concatenate((rates_bs, rates_ps), axis=0)
mean_annual_rates = rates.mean(axis=1)

##
rates = rates.mean(axis=1)
##

# Retrieve arrival time
data_time = pd.read_csv("arrival_time_{}.txt".format(inpdir), engine="python", sep="\s+")
time_sec = data_time["AT"].to_numpy()
time_min = time_sec/60 # from seconds to hours
tmin = 1

time = np.array([max(t, tmin) for t in time_min])
time = 1/time

# Thresholds to be exceeded

thresholds = np.array([1.000e-02, 5.000e-02, 1.000e-01, 2.000e-01,
       3.000e-01, 5.000e-01, 8.000e-01, 1.000e+00,
       1.500e+00, 2.000e+00, 2.500e+00, 3.000e+00,
       3.600e+00, 4.320e+00, 5.180e+00, 6.220e+00,
       7.460e+00, 8.950e+00, 1.074e+01, 1.289e+01,
       1.547e+01, 1.856e+01, 2.227e+01])
#=================================================================
# Importance sampling
#=================================================================
# Magnitude binning

data_mw = {"Magnitude": mw, "Rates": mean_annual_rates}
df_mw = pd.DataFrame(data_mw)
mw_bins = np.array(list(df_mw.groupby("Magnitude").groups.keys()))
mw_counts = df_mw.groupby("Magnitude").count().to_numpy().squeeze()
#===========================================
# Flow depth
pathHeight_ON = 'HMAX_ONSHORE/hmax_onshore_{}{}.txt'.format(inpdir,test_site)
data_on = pd.read_csv(pathHeight_ON, engine='python', sep='\s+')
hmax_on = data_on["MIH"].to_numpy()
ids_on = data_on["IDs_sim"].to_numpy()


#===========================================
rates1 = np.concatenate((rates_bs, rates_ps), axis=0)
lambda_true = exceedance_curve(thresholds, hmax_on, rates1)
lambda_true_mean = lambda_true.mean(axis=1)
p2 = np.percentile(lambda_true, 2, axis=1)
p16 = np.percentile(lambda_true, 16, axis=1)
p84 = np.percentile(lambda_true, 84, axis=1)
p98 = np.percentile(lambda_true, 98, axis=1)

#===========================================
# Monte Carlo simulation
lambda_sis = np.zeros((len(thresholds), n_iter))

# Which one? 
#ix1 = np.argwhere(lambda_true_mean <= 2e-3)[0].squeeze()
ix1 = np.argwhere(lambda_true_mean <= 4e-4)[0].squeeze()
#ix1 = np.argwhere(lambda_true_mean <= 1e-4)[0].squeeze()


#if percentage==1500:
fig, axes = plt.subplots(figsize=(35,30), nrows=2, ncols=3)#, constrained_layout=True)

axes = axes.flatten()
ax1 = axes[0]
ax2 = axes[1]
ax3 = axes[2]
ax4 = axes[3]
ax5 = axes[4]
ax6 = axes[5]

rect1 = ax1.patch
rect2 = ax2.patch
rect3 = ax3.patch
rect4 = ax4.patch
rect5 = ax5.patch
rect6 = ax6.patch


rect1.set_facecolor("whitesmoke")
rect2.set_facecolor("whitesmoke")
rect3.set_facecolor("whitesmoke")
rect4.set_facecolor("whitesmoke")
rect5.set_facecolor("whitesmoke")
rect6.set_facecolor("whitesmoke")

#=======================================
# New
pathNew1 = "MC_RESULTS/results_perc1500_{}".format(inpdir)
df_nsis1 = pd.read_csv(os.path.join(pathNew1, 'optimal_nsamples_SIH.txt'), sep='\s+', engine='python')
n_sis1 = df_nsis1["nSamples"].to_numpy()

pathNew2 = "MC_RESULTS/results_perc6000_{}".format(inpdir)
df_nsis2 = pd.read_csv(os.path.join(pathNew2, 'optimal_nsamples_SIH.txt'), sep='\s+', engine='python')
n_sis2 = df_nsis2["nSamples"].to_numpy()

# Proposal
proposal_offshore = hmax_off * mean_annual_rates * time
lambda_sis1 = mc_simulation(thresholds, n_iter, 1500, mw, rates, hmax_on, proposal_offshore, n_sis1)
lambda_sis2 = mc_simulation(thresholds, n_iter, 6000, mw, rates, hmax_on, proposal_offshore, n_sis2)
#------------
# Old
pathOld1 = "MC_RESULTS/gareth_perc1500_{}".format(inpdir)
df_nsis1 = pd.read_csv(os.path.join(pathOld1, 'optimal_nsamples_SIH.txt'), sep='\s+', engine='python')
n_old1 = df_nsis1["nSamples"].to_numpy()

pathOld2 = "MC_RESULTS/gareth_perc6000_{}".format(inpdir)
df_nsis2 = pd.read_csv(os.path.join(pathOld2, 'optimal_nsamples_SIH.txt'), sep='\s+', engine='python')
n_old2 = df_nsis2["nSamples"].to_numpy()

# Proposal
proposal_old = hmax_off * mean_annual_rates 
lambda_sis_old1 = mc_simulation(thresholds, n_iter, 1500, mw, rates, hmax_on, proposal_old, n_old1)
lambda_sis_old2 = mc_simulation(thresholds, n_iter, 6000, mw, rates, hmax_on, proposal_old, n_old2)

lambda_sis_mean1 = lambda_sis1.mean(axis=1)
lambda_sis_mean2 = lambda_sis2.mean(axis=1)

# Analytical error onshore
_, _, _, _, _, an_var1, an_up1, an_dw1, _, _, an_var_old1, an_up_old1, an_dw_old1, _, _ = get_sis_data_onshore(1500, inpdir, test_site)
_, _, _, _, _, an_var2, an_up2, an_dw2, _, _, an_var_old2, an_up_old2, an_dw_old2, _, _ = get_sis_data_onshore(6000, inpdir, test_site)

old1500 = an_var_old1
#=======================================================================================================================
# 1500
#=======================================================================================================================
ax1.plot(thresholds, lambda_sis1[:, :500], color='tab:blue', linewidth=0.1, alpha=0.2, label='')
sis1, = ax1.plot(np.NaN, np.NaN, color='tab:blue', linewidth=4, label='SIS mean (1 iter.)')
true_mean, = ax1.plot(thresholds, lambda_true_mean, color='black', linewidth=4, label='Exact mean')
ci_new, = ax1.plot(thresholds, an_up1, color='darkblue', linewidth=4, linestyle='-', label='Analytical 95% \n c.i. (new IF)')
ax1.plot(thresholds, an_dw1, color='darkblue', linewidth=4, linestyle='-', label='')
ax1.fill_between(thresholds, an_dw_old1, an_dw1, color='tab:red', alpha=0.6)
ax1.fill_between(thresholds, an_up1, an_up_old1, color='tab:red', alpha=0.6)
ci_old, = ax1.fill(np.NaN, np.NaN, 'tab:red', alpha=0.6, label="Analytical 95% \n c.i. (old IF)")
sis_mean, = ax1.plot(thresholds, lambda_sis_mean1, color='gold', linewidth=4, linestyle='-.',  label='SIS mean (10 k iter.)')
ax1.set_box_aspect(1)
# Histograms
#=======================================
ax3.hist(lambda_sis1[ix1, :], bins=100, color="tab:blue", alpha=0.1, density=True, label="", edgecolor='tab:blue')
ax3.hist(lambda_sis_old1[ix1, :], bins=100, color="tab:red", alpha=0.3, density=True, label="", edgecolor='tab:red')
x_new, p_new = normal(lambda_true_mean[ix1], an_var1[ix1])
x_old, p_old = normal(lambda_true_mean[ix1], an_var_old1[ix1])
old1500, _ = normal(lambda_true_mean[ix1], old1500[ix1])

p_new, = ax3.plot(x_new, p_new, color="darkblue", linestyle='-.', linewidth=4, label="Analytical variance (new IF)")
p_old, = ax3.plot(x_old, p_old, color="tab:red", linestyle='-.', linewidth=4, label="Analytical variance (old IF)")
ax3.set_xlim([old1500.min(), old1500.max()])
ax3.axvline(x=lambda_true_mean[ix1], color='green', linestyle='-.')
plt.setp(ax3.get_yticklabels(), visible=True, fontsize=50)
ax3.set_ylabel("Density", fontsize=50)
ax6.ticklabel_format(style='sci', axis='x', scilimits=(0, 0))#0,0))
ax3.ticklabel_format(useOffset=False)
ax3.yaxis.offsetText.set_fontsize(50)

ax3.set_box_aspect(1)
# Epistemic
#=========================================
lambda_sis_mean, sis_p2, sis_p16, sis_p84, sis_p98, _, ci_new_up, ci_new_dw, mc_ci_new_up, mc_ci_new_dw, _, ci_old_up, ci_old_dw, mc_ci_old_up, mc_ci_old_dw = get_sis_data_onshore(1500, inpdir, test_site)
ax2.plot(thresholds, lambda_sis_mean, color="tab:blue", linewidth=4, linestyle='-', label='SIS mean (1 iter.)')
true_percentiles1, = ax2.plot(thresholds, p2, color="darkmagenta", linewidth=4, alpha=0.8, label='Exact p2 - p98')
ax2.plot(thresholds, p98, color="darkmagenta", linewidth=4, alpha=0.8, label='')
true_percentiles2, = ax2.plot(thresholds, p16, color="darkorange", linewidth=3.5, alpha=0.8, label='Exact p16 - p84')
ax2.plot(thresholds, p84, color="darkorange", linewidth=4, alpha=0.8, label='')
ax2.plot(thresholds, sis_p2, color="black", linestyle="-.", linewidth=4, label='SIS percentiles')
ax2.plot(thresholds, sis_p98, color="black", linestyle="-.", linewidth=4, label='')
sis_percentiles, = ax2.plot(thresholds, sis_p16, color="black", linestyle="-.", linewidth=4, label='SIS percentiles')
ax2.plot(thresholds, sis_p84, color="black", linestyle="-.", linewidth=4, label='')
ax2.fill_between(thresholds, mc_ci_new_dw, mc_ci_new_up, color="hotpink", alpha=0.4, label="Estimated 95%\nc.i. (new IF)")
sis_ci, = ax2.fill(np.NaN, np.NaN, 'hotpink', alpha=0.4, label="Estimated 95% \n c.i. (new IF)")

ax2.set_box_aspect(1)
#ax1.legend(handles=[true_mean, sis_mean, sis1, ci_new, ci_old,
#                    true_percentiles1, true_percentiles2, sis_percentiles, sis_ci,
#                    p_new, p_old], bbox_to_anchor=(-0.4, 1.15, 4.7, 0.3), loc="lower left",  mode="expand", borderaxespad=0, ncol=3, fontsize=50)

ax1.set_yscale('log')
ax1.set_ylim([1e-6, 1])
ax2.set_yscale('log')
ax2.set_ylim([1e-6, 1])

#if (inpdir=="CT") or (inpdir=="SR"):
ax1.set_xlim([0, 7.5])
ax2.set_xlim([0, 7.5])
#ax1.text(x=0.2, y=4e-1, s="N = 1500", color="darkblue", fontsize=50)
if inpdir=="SR":
   ax1.text(x=0.2, y=3e-6, s="N = 1500", color="black", fontsize=48, bbox=dict(edgecolor="black", facecolor="white", pad=8))
else:
   ax1.text(x=0.2, y=1e-3, s="N = 1500", color="black", fontsize=48, bbox=dict(edgecolor="black", facecolor="white", pad=8))

ax1.axvline(x=thresholds[ix1], color='black', linestyle=':')
# Annotate above the horizontal line
#ax1.text(x=thresholds[ix1]+0.2, y=1e-4, s= '{} m'.format(thresholds[ix1]), 
#             rotation=90, fontsize=35, color='green')
ax1.axhline(y=lambda_true_mean[ix1], color='tab:purple', linestyle='-.')
# Annotate above the horizontal line
ax1.text(x=7.8, y = lambda_true_mean[ix1], s=np.round(lambda_true_mean[ix1],5),  
             fontsize=35, color='tab:purple')

ax2.axhline(y=lambda_true_mean[ix1], color='tab:purple', linestyle='-.')
#ax1.text(-0.2, 1.12, "(a)", bbox=dict(facecolor='#ffffff', edgecolor='black', pad=4.0), transform=ax1.transAxes,
#         fontsize=50, va='top', ha='right')
#ax2.text(-0.2, 1.12, "(b)", bbox=dict(facecolor='#ffffff', edgecolor='black', pad=4.0), transform=ax2.transAxes,
#         fontsize=50, va='top', ha='right')
#ax3.text(-0.2, 1.12, "(c)", bbox=dict(facecolor='#ffffff', edgecolor='black', pad=4.0), transform=ax3.transAxes,
#         fontsize=50, va='top', ha='right')

ax1.text(0.16, 0.98, "(a)", bbox=dict(facecolor='white', edgecolor='grey', pad=8.0), transform=ax1.transAxes,
         fontsize=48, va='top', ha='right')
ax2.text(0.16, 0.98, "(b)", bbox=dict(facecolor='white', edgecolor='grey', pad=8.0), transform=ax2.transAxes,
         fontsize=48, va='top', ha='right')
ax3.text(0.16, 0.98, "(c)", bbox=dict(facecolor='white', edgecolor='grey', pad=8.0), transform=ax3.transAxes,
         fontsize=48, va='top', ha='right')

plt.setp(ax1.get_yticklabels(), visible=True, fontsize=50)
ax1.set_ylabel("Annual exceedance-rate", fontsize=50)
plt.setp(ax2.get_yticklabels(), visible=False)
plt.setp(ax1.get_xticklabels(), visible=False)
plt.setp(ax2.get_xticklabels(), visible=False)
plt.setp(ax3.get_xticklabels(), visible=False)
ax3.set_title('Flow depth = {} m'.format(round(thresholds[ix1], 2)), fontsize=50)

#ax2.text(x=0.3, y=0.98, s='Exc. rate = {}'.format(np.round(lambda_true_mean[ix1],5)), color='green', ha='left', va='top',
#            transform=ax2.get_xaxis_transform(), fontsize=50)

# Maps

#axins = inset_axes(ax1, width="50%", height="70%", loc="upper right")
lat_0 = lat.mean()
lon_0 = lon.mean()
if inpdir=="CT":
  axins = inset_axes(ax1, width="52%", height="72%", loc="upper right",
                   bbox_to_anchor=(0.43, 0.43, 0.52, 0.72), bbox_transform=ax1.transAxes)
  m = Basemap(llcrnrlon=15, llcrnrlat=37.3, urcrnrlon=15.17, urcrnrlat=37.550,
              lat_0=lat_0, lon_0=lon_0,
              projection="cyl",
              resolution="f")
#  parallels = [37.35, 37.5]
  # labels = [left,right,top,bottom]
#  m.drawparallels(parallels, labels=[1,0,0,0], fontsize=32)
#  meridians = [15.05, 15.15]
#  m.drawmeridians(meridians, labels=[0,0,0,1], fontsize=32)
elif inpdir=="SR":
  axins = inset_axes(ax1, width="70%", height="90%", loc="upper right",
                   bbox_to_anchor=(0.4, 0.45, 0.70, 0.90), bbox_transform=ax1.transAxes)
  m = Basemap(llcrnrlon=15.2, llcrnrlat=37, urcrnrlon=15.4, urcrnrlat=37.2,
              lat_0=lat_0, lon_0=lon_0,
              projection="cyl",
              resolution="f")
  #parallels = [37.05, 37.15]
  #m.drawparallels(parallels, labels=[1,0,0,0], fontsize=32)
  #meridians =  [15.25, 15.35]
  #m.drawmeridians(meridians, labels=[0,0,0,1], fontsize=32)
m.drawparallels([np.round(lat_on,2)], labels=[1,0,0,0], fontsize=35, color="#FF7F00", textcolor="#FF7F00")
m.drawmeridians([np.round(lon_on,2)], labels=[0,0,0,1], fontsize=35, color="#FF7F00", textcolor="#FF7F00")
# 
m.drawcoastlines()
m.drawrivers()
m.drawmapboundary(fill_color="azure")

# 
m.fillcontinents(color='lightgray',lake_color='azure')
m.scatter(np.round(lon_sis,2), np.round(lat_sis,2), marker = 'X', color='#000080', s=400, label='SIS POI')
m.scatter(np.round(lon_on,2), np.round(lat_on,2), marker = 'v', color='#FF7F00', s=400, label="Onshore POI")
plt.legend(loc="upper center", facecolor="white", edgecolor="#FF7F00", fontsize=30)

#-------------------------

#elif percentage==3000:
#   plt.setp(ax1.get_yticklabels(), visible=True, fontsize=50)
#   ax1.set_ylabel("Annual exceedance-rate", fontsize=50)
#   plt.setp(ax2.get_yticklabels(), visible=False)
#   plt.setp(ax1.get_xticklabels(), visible=False)
#   plt.setp(ax2.get_xticklabels(), visible=False)
#   plt.setp(ax3.get_xticklabels(), visible=False)
#   ax1.text(0.1, 1.10, "(d)", transform=ax1.transAxes,
#      fontsize=50, va='top', ha='right')
#   ax2.text(0.1, 1.10, "(e)", transform=ax2.transAxes,
#      fontsize=50, va='top', ha='right')
#   ax3.text(0.1, 1.10, "(f)", transform=ax3.transAxes,
#      fontsize=50, va='top', ha='right')

#else:
#=======================================================================================================================
# 6000
#=======================================================================================================================
ax4.plot(thresholds, lambda_sis2[:, :500], color='tab:blue', linewidth=0.1, alpha=0.2, label='')
ax4.plot(thresholds, lambda_true_mean, color='black', linewidth=4, label='Exact mean')
ax4.plot(thresholds, an_up2, color='darkblue', linewidth=4, linestyle='-', label='Analytical 95%\nc.i. (new IF)')
ax4.plot(thresholds, an_dw2, color='darkblue', linewidth=4, linestyle='-', label='')
ax4.fill_between(thresholds, an_dw_old2, an_dw2, color='tab:red', alpha=0.6)
ax4.fill_between(thresholds, an_up2, an_up_old2, color='tab:red', alpha=0.6)
ax4.plot(thresholds, lambda_sis_mean2, color='gold', linewidth=4, linestyle='-.',  label='SIS mean (10 k iter.)')

ax4.legend(handles=[true_mean, sis_mean, sis1, ci_new, ci_old], frameon=True, edgecolor="grey", 
           bbox_to_anchor=(0.01,1.04), loc="lower left", borderaxespad=0, ncol=1, fontsize=35)

ax4.set_box_aspect(1)
# Histograms
#=======================================
ax6.hist(lambda_sis2[ix1, :], bins=100, color="tab:blue", alpha=0.1, density=True, label="", edgecolor='tab:blue')
ax6.hist(lambda_sis_old2[ix1, :], bins=100, color="tab:red", alpha=0.3, density=True, label="", edgecolor='tab:red')
x_new, p_new = normal(lambda_true_mean[ix1], an_var2[ix1])
x_old, p_old = normal(lambda_true_mean[ix1], an_var_old2[ix1])

p_new, = ax6.plot(x_new, p_new, color="darkblue", linestyle='-.', linewidth=4, label="Analytical variance\n(new IF)")
p_old, = ax6.plot(x_old, p_old, color="tab:red", linestyle='-.', linewidth=4, label="Analytical variance\n(old IF)")
ax6.set_xlim([old1500.min(), old1500.max()])
ax6.axvline(x=lambda_true_mean[ix1], color='tab:purple', linestyle='-.')
plt.setp(ax6.get_yticklabels(), visible=True, fontsize=50)
ax6.set_ylabel("Density", fontsize=50)

ax6.legend(handles=[p_new, p_old], frameon=True, edgecolor="grey",
           bbox_to_anchor=(0.02, 1.12), loc="lower left", borderaxespad=0, ncol=1, fontsize=35)

ax6.set_box_aspect(1)
# Epistemic
#=========================================
lambda_sis_mean, sis_p2, sis_p16, sis_p84, sis_p98, _, ci_new_up, ci_new_dw, mc_ci_new_up, mc_ci_new_dw, _, ci_old_up, ci_old_dw, mc_ci_old_up, mc_ci_old_dw = get_sis_data_onshore(6000, inpdir, test_site)
ax5.plot(thresholds, lambda_sis_mean, color="tab:blue", linewidth=4, linestyle='-', label='SIS mean (1 iter.)')
ax5.plot(thresholds, p2, color="darkmagenta", linewidth=4, alpha=0.8, label='Exact p2 - p98')
ax5.plot(thresholds, p98, color="darkmagenta", linewidth=4, alpha=0.8, label='')
ax5.plot(thresholds, p16, color="darkorange", linewidth=3.5, alpha=0.8, label='Exact p16 - p84')
ax5.plot(thresholds, p84, color="darkorange", linewidth=4, alpha=0.8, label='')
ax5.plot(thresholds, sis_p2, color="black", linestyle="-.", linewidth=4, label='SIS percentiles')
ax5.plot(thresholds, sis_p98, color="black", linestyle="-.", linewidth=4, label='')
ax5.plot(thresholds, sis_p16, color="black", linestyle="-.", linewidth=4, label='SIS percentiles')
ax5.plot(thresholds, sis_p84, color="black", linestyle="-.", linewidth=4, label='')
ax5.fill_between(thresholds, mc_ci_new_dw, mc_ci_new_up, color="hotpink", alpha=0.4, label="Estimated 95%\nc.i. (new IF)")

ax5.legend(handles=[true_percentiles1, true_percentiles2, sis_percentiles, sis1, sis_ci], frameon=True, edgecolor="grey",
           bbox_to_anchor=(0.06, 1.08), loc="lower left", borderaxespad=0, ncol=1, fontsize=35)

ax5.set_box_aspect(1)
ax4.set_yscale('log')
ax4.set_ylim([1e-6, 1])
ax5.set_yscale('log')
ax5.set_ylim([1e-6, 1])

#if (inpdir=="CT") or (inpdir=="SR"):
ax4.set_xlim([0, 7.5])
ax5.set_xlim([0, 7.5])

if inpdir=="SR":
   ax4.text(x=0.2, y=3e-6, s="N = 6000", color="black", bbox=dict(facecolor="white", edgecolor="black", pad=8.0), fontsize=48)
else:
   ax4.text(x=0.2, y=1e-3, s="N = 6000", color="black", bbox=dict(facecolor="white", edgecolor="black", pad=8.0), fontsize=48)
ax4.text(0.16, 0.98, "(d)", bbox=dict(facecolor='white', edgecolor='grey', pad=8.0), transform=ax4.transAxes,
         fontsize=48, va='top', ha='right')
ax5.text(0.16, 0.98, "(e)", bbox=dict(facecolor='white', edgecolor='grey', pad=8.0), transform=ax5.transAxes,
         fontsize=48, va='top', ha='right')
ax6.text(0.16, 0.98, "(f)", bbox=dict(facecolor='white', edgecolor='grey', pad=8.0), transform=ax6.transAxes,
         fontsize=48, va='top', ha='right')

plt.setp(ax4.get_yticklabels(), visible=True, fontsize=50)
ax4.set_ylabel("Annual exceedance-rate", fontsize=50)
plt.setp(ax5.get_yticklabels(), visible=False)
plt.setp(ax5.get_xticklabels(), visible=True, fontsize=50)
   
for label in ax6.xaxis.get_ticklabels()[::2]:
    label.set_visible(False)
ax6.ticklabel_format(style='sci', axis='x', scilimits=(old1500.min(), old1500.max()))#0,0))
plt.setp(ax6.get_xticklabels(), visible=True, fontsize=50)
#ax6.ticklabel_format(useOffset=False)
ax6.xaxis.offsetText.set_fontsize(50)
pad = plt.rcParams["xtick.major.size"] + plt.rcParams["xtick.major.pad"]   
ax6.xaxis._update_offset_text_position = types.MethodType(bottom_offset, ax6.xaxis)
ax6.set_xlabel("Annual exceedance-rate", fontsize=50)
plt.setp(ax4.get_xticklabels(), visible=True, fontsize=50)
ax4.set_xlabel("Flow depth [m]", fontsize=50)
plt.setp(ax5.get_xticklabels(), visible=True, fontsize=50)
ax5.set_xlabel("Flow depth [m]", fontsize=50)

plt.subplots_adjust(wspace=0.5, hspace=0.2)
#plt.tight_layout()
plt.savefig("FD{}_poi{}.pdf".format(thresholds[ix1], test_site), format="pdf", bbox_inches='tight')
plt.show()

vr_new1 = np.round(100*abs(an_var1[ix1]-an_var_old1[ix1])/an_var_old1[ix1], 2)
df = pd.DataFrame([vr_new1])
df.to_csv(os.path.join(outdir, "variance_reduction_1500_{}m_poi{}.txt".format(thresholds[ix1], test_site)), sep=' ', index=False)

vr_new2 = np.round(100*abs(an_var2[ix1]-an_var_old2[ix1])/an_var_old2[ix1], 2)
df = pd.DataFrame([vr_new1])
df.to_csv(os.path.join(outdir, "variance_reduction_6000_{}m_poi{}.txt".format(thresholds[ix1], test_site)), sep=' ', index=False)

