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
#=================================================================================================
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
    parser.add_argument('--num_samples', default=None, required=True, help='Number of scenarios to be sampled from the ensemble')
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
    percentage = local_opts.num_samples
    outdir = local_opts.outdir
    return str(inpdir), int(test_site), int(n_iter), int(percentage), str(outdir)

#------------------------------------------------------------------------------------------------------------------------
inpdir, test_site, n_iter, percentage, outdir = from_local_parser()
#-------------------------------------------------------------------------------------------------------------------------
matplotlib.rcParams['font.family'] = 'serif'
matplotlib.rcParams['font.sans-serif'] = 'Times'


# Defining the directory for storing the results
#-------------------------------------------------------------------------------------------------------------------------
outdir = outdir+'_perc' + str(percentage)+ '_{}'.format(inpdir)
#-------------------------------------------------------------------------------------------------------------------------
# Defining directory for outputs

try:
  os.makedirs(os.path.join("MC_RESULTS", outdir))
except:
  os.path.exists(os.path.join("MC_RESULTS", outdir))
  print("Directory for outputs already there!")
#----------------------------------------------------------------------------
outdir = os.path.join("MC_RESULTS", outdir)
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

elif inpdir=="LK":
  poi_off = pd.read_csv('sis_poi_larnaka.txt', engine='python', header=None, index_col=False).to_numpy().squeeze()
  region_map_LK = [33.55, 33.75, 34.85, 35.05]
  city = "Larnaka"

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
if (inpdir == "CT") or (inpdir=="SR"):
   fileRates_bs = "probs_BS_Sicily.txt"
   fileRates_ps = "probs_PS_Sicily.txt"
elif (inpdir == "LK"):
   fileRates_bs = "probs_BS_Cyprus.txt"
   fileRates_ps = "probs_PS_Cyprus.txt"

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
if inpdir == "LK":
   tmin = 30
   #tmin = 1
else:
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
pathHeight_ON = 'HMAX_ONSHORE/hmax_onshore_{}{}.txt'.format(inpdir,test_site)
data_on = pd.read_csv(pathHeight_ON, engine='python', sep='\s+')
hmax_on = data_on["MIH"].to_numpy()
ids_on = data_on["IDs_sim"].to_numpy()

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
# Analytical error onshore
_, _, _, _, _, an_var, an_up, an_dw, _, _, an_var_old, an_up_old, an_dw_old, _, _ = get_sis_data_onshore(percentage, inpdir, test_site)
_, _, _, _, _, _, _, _, _, _, old1500, _, _, _, _ = get_sis_data_onshore(1500, inpdir, test_site)




fig, axes = plt.subplots(figsize=(35,15), ncols=3, constrained_layout=True)
ax1 = axes[0]
ax2 = axes[1]
ax3 = axes[2]
#=======================================
# New
pathNew = "MC_RESULTS/results_perc{}_{}".format(percentage, inpdir)
df_nsis = pd.read_csv(os.path.join(pathNew, 'optimal_nsamples_SIH.txt'), sep='\s+', engine='python')
n_sis = df_nsis["nSamples"].to_numpy()
# Proposal
proposal_offshore = hmax_off * mean_annual_rates * time
lambda_sis = mc_simulation(thresholds, n_iter, percentage, mw, rates, hmax_on, proposal_offshore, n_sis)
#------------
# Old
pathOld = "MC_RESULTS/gareth_perc{}_{}".format(percentage, inpdir)
df_nsis = pd.read_csv(os.path.join(pathOld, 'optimal_nsamples_SIH.txt'), sep='\s+', engine='python')
n_old = df_nsis["nSamples"].to_numpy()
# Proposal
proposal_old = hmax_off * mean_annual_rates 
lambda_sis_old = mc_simulation(thresholds, n_iter, percentage, mw, rates, hmax_on, proposal_old, n_old)
lambda_sis_mean = lambda_sis.mean(axis=1)




# Which one? 
#ix1 = np.argwhere(lambda_true_mean <= 2e-3)[0].squeeze()
ix1 = np.argwhere(lambda_true_mean <= 4e-4)[0].squeeze()
#ix1 = np.argwhere(lambda_true_mean <= 1e-4)[0].squeeze()



ax1.plot(thresholds, lambda_sis[:, :500], color='tab:blue', linewidth=0.1, alpha=0.2, label='')
true_mean, = ax1.plot(thresholds, lambda_true_mean, color='black', linewidth=3.5, label='Exact mean')
ci_new, = ax1.plot(thresholds, an_up, color='darkblue', linewidth=3.5, linestyle='-', label='Analytical c.i. 95% (new IF)')
ax1.plot(thresholds, an_dw, color='darkblue', linewidth=3.5, linestyle='-', label='')
ci_old, = ax1.plot(thresholds, an_up_old, color='tab:red', linewidth=3, linestyle='-', label='Analytical c.i. 95% (old IF)')
ax1.plot(thresholds, an_dw_old, color='tab:red', linewidth=3.5, linestyle='-', label='')
sis_mean, = ax1.plot(thresholds, lambda_sis_mean, color='mediumblue', linewidth=3.5, linestyle='-',  label='SIS mean')

# Histograms
#=======================================
ax3.hist(lambda_sis[ix1, :], bins=100, color="tab:blue", alpha=0.1, density=True, label="", edgecolor='darkblue')
ax3.hist(lambda_sis_old[ix1, :], bins=100, color="tab:red", alpha=0.3, density=True, label="", edgecolor='tab:red')
x_new, p_new = normal(lambda_true_mean[ix1], an_var[ix1])
x_old, p_old = normal(lambda_true_mean[ix1], an_var_old[ix1])
old1500, _ = normal(lambda_true_mean[ix1], old1500[ix1])

p_new, = ax3.plot(x_new, p_new, color="darkblue", linestyle='-.', linewidth=3.5, label="Analytical variance (new IF)")
p_old, = ax3.plot(x_old, p_old, color="tab:red", linestyle='-.', linewidth=3.5, label="Analytical variance (old IF)")
ax3.set_xlim([old1500.min(), old1500.max()])
ax3.axvline(x=lambda_true_mean[ix1], color='green', linestyle='-.')
plt.setp(ax3.get_yticklabels(), visible=True, fontsize=35)
plt.setp(ax3.get_xticklabels(), visible=True, fontsize=35)
ax3.set_ylabel("Density", fontsize=40)

# Epistemic
#=========================================
lambda_sis_mean, sis_p2, sis_p16, sis_p84, sis_p98, _, ci_new_up, ci_new_dw, mc_ci_new_up, mc_ci_new_dw, _, ci_old_up, ci_old_dw, mc_ci_old_up, mc_ci_old_dw = get_sis_data_onshore(percentage, inpdir, test_site)
sis_mean, = ax2.plot(thresholds, lambda_sis_mean, color="darkblue", linewidth=3.5, linestyle='-', label='')
true_percentiles1, = ax2.plot(thresholds, p2, color="darkmagenta", linewidth=3.5, alpha=0.6, label='Exact p2 - p98')
ax2.plot(thresholds, p98, color="darkmagenta", linewidth=3.5, alpha=0.6, label='')
true_percentiles2, = ax2.plot(thresholds, p16, color="darkorange", linewidth=3.5, alpha=0.6, label='Exact p16 - p84')
ax2.plot(thresholds, p84, color="darkorange", linewidth=3.5, alpha=0.6, label='')
ax2.plot(thresholds, sis_p2, color="black", linestyle=":", linewidth=3.5, label='SIS percentiles')
ax2.plot(thresholds, sis_p98, color="black", linestyle=":", linewidth=3.5, label='')
sis_percentiles, = ax2.plot(thresholds, sis_p16, color="black", linestyle=":", linewidth=3.5, label='')
ax2.plot(thresholds, sis_p84, color="black", linestyle=":", linewidth=3.5, label='')
ax2.fill_between(thresholds, mc_ci_new_dw, mc_ci_new_up, color="hotpink", alpha=0.2, label="Estimated 95% c.i. (new IF)")
sis_ci, = ax2.fill(np.NaN, np.NaN, 'hotpink', alpha=0.2, label="Estimated 95% c.i. (new IF)")


ax1.set_yscale('log')
ax1.set_ylim([1e-6, 1])
ax2.set_yscale('log')
ax2.set_ylim([1e-6, 1])


if (inpdir=="CT") or (inpdir=="SR"):
  ax1.set_xlim([0, 7.5])
  ax2.set_xlim([0, 7.5])
  ax1.text(x=5, y=5*1e-5, s="N = {}".format(percentage), color="darkblue", fontsize=40)
else:
  ax1.set_xlim([0,5])
  ax2.set_xlim([0, 5])
  ax1.text(x=3.5, y=5*1e-5, s="N = {}".format(percentage), color="darkblue", fontsize=40)

if (percentage==1500):
   plt.setp(ax1.get_yticklabels(), visible=True, fontsize=40)
   ax1.set_ylabel("Mean annual exceedance-rate", fontsize=40)
   plt.setp(ax2.get_yticklabels(), visible=False)
   plt.setp(ax1.get_xticklabels(), visible=False)
   plt.setp(ax2.get_xticklabels(), visible=False)
   plt.setp(ax3.get_xticklabels(), visible=False)
   ax3.set_title('Flow depth = {} m'.format(round(thresholds[ix1], 2)), fontsize=40)
   ax1.legend(handles=[true_mean, sis_mean, ci_new, ci_old, p_new, p_old, true_percentiles1, true_percentiles2, sis_percentiles, sis_ci], bbox_to_anchor=(0, 1.08, 3.5, 0.3), loc="lower left",
                mode="expand", borderaxespad=0, ncol=4, fontsize=33)
   ax2.text(x=1, y=0.98, s='Mean exceedance-rate = {}'.format(np.round(lambda_true_mean[ix1],4)), color='green', ha='left', va='top',
            transform=ax2.get_xaxis_transform(), fontsize=30)

   # Maps

   axins = inset_axes(ax1, width="45%", height="65%", loc="upper right")

   lat_0 = lat.mean()
   lon_0 = lon.mean()
   if inpdir=="CT":
     m = Basemap(llcrnrlon=15, llcrnrlat=37.3, urcrnrlon=15.17, urcrnrlat=37.550,
            lat_0=lat_0, lon_0=lon_0,
            projection="cyl",
            resolution="f")
     parallels = [37.35, 37.5]
     # labels = [left,right,top,bottom]
     m.drawparallels(parallels, labels=[1,0,0,0], fontsize=35)
     meridians = [15.05, 15.15]
     m.drawmeridians(meridians, labels=[0,0,0,1], fontsize=35)
   elif inpdir=="SR":
     m = Basemap(llcrnrlon=15.2, llcrnrlat=37, urcrnrlon=15.4, urcrnrlat=37.2,
            lat_0=lat_0, lon_0=lon_0,
            projection="cyl",
            resolution="f")
     parallels = [37.05, 37.15]
     m.drawparallels(parallels, labels=[1,0,0,0], fontsize=35)
     meridians =  [15.25, 15.35]
     m.drawmeridians(meridians, labels=[0,0,0,1], fontsize=35)
   elif inpdir =="LK":
     m = Basemap(llcrnrlon=33.55, llcrnrlat=34.85, urcrnrlon=33.75, urcrnrlat=35.05,
            lat_0=lat_0, lon_0=lon_0,
            projection="cyl",
            resolution="f")
     parallels = [34.9, 35]
     m.drawparallels(parallels, labels=[1,0,0,0], fontsize=35)
     meridians = [33.6, 33.7]
     m.drawmeridians(meridians, labels=[0,0,0,1], fontsize=35)
   # 
   m.drawcoastlines()
   m.drawrivers()
   m.drawmapboundary(fill_color="#A6CAE0")

   # 
   m.fillcontinents(color='grey',lake_color='#A6CAE0')
   m.scatter(lon_sis, lat_sis, marker = 'X', color='purple', s=300, label='')

   ax1.axvline(x=thresholds[ix1], color='black', linestyle=':')
   ax1.axhline(y=lambda_true_mean[ix1], color='green', linestyle='-.')
   ax2.axhline(y=lambda_true_mean[ix1], color='green', linestyle='-.')
   ax1.text(0.1, 1.04, "(a)", transform=ax1.transAxes,
      fontsize=32, va='top', ha='right')
   ax2.text(0.1, 1.04, "(b)", transform=ax2.transAxes,
      fontsize=32, va='top', ha='right')
   ax3.text(0.1, 1.04, "(c)", transform=ax3.transAxes,
      fontsize=32, va='top', ha='right')

   m.scatter(lon_on, lat_on, marker = 'v', color='darkred', s=300, label="Onshore POI")
   plt.legend(loc="upper center", fontsize=30)

#-------------------------

elif percentage==3000:
   plt.setp(ax1.get_yticklabels(), visible=True, fontsize=40)
   ax1.set_ylabel("Mean annual exceedance-rate", fontsize=40)
   plt.setp(ax2.get_yticklabels(), visible=False)
   plt.setp(ax1.get_xticklabels(), visible=False)
   plt.setp(ax2.get_xticklabels(), visible=False)
   plt.setp(ax3.get_xticklabels(), visible=False)
   ax1.text(0.1, 1.04, "(d)", transform=ax1.transAxes,
      fontsize=32, va='top', ha='right')
   ax2.text(0.1, 1.04, "(e)", transform=ax2.transAxes,
      fontsize=32, va='top', ha='right')
   ax3.text(0.1, 1.04, "(f)", transform=ax3.transAxes,
      fontsize=32, va='top', ha='right')

else:
   plt.setp(ax1.get_yticklabels(), visible=True, fontsize=40)
   ax1.set_ylabel("Mean annual exceedance-rate", fontsize=40)
   plt.setp(ax2.get_yticklabels(), visible=False)
   plt.setp(ax2.get_xticklabels(), visible=True, fontsize=40)
  
   for label in ax3.xaxis.get_ticklabels()[::2]:
    label.set_visible(False)
   ax3.set_xlabel("Mean annual exceedance-rate", fontsize=40)
   plt.setp(ax1.get_xticklabels(), visible=True, fontsize=40)
   ax1.set_xlabel("Flow depth [m]", fontsize=40)
   plt.setp(ax2.get_xticklabels(), visible=True, fontsize=40)
   ax2.set_xlabel("Flow depth [m]", fontsize=40)
   ax1.text(0.1, 1.04, "(g)", transform=ax1.transAxes,
      fontsize=32, va='top', ha='right')
   ax2.text(0.1, 1.04, "(h)", transform=ax2.transAxes,
      fontsize=32, va='top', ha='right')
   ax3.text(0.1, 1.04, "(i)", transform=ax3.transAxes,
      fontsize=32, va='top', ha='right')

plt.tight_layout()
plt.savefig(outdir+"/FD{}_poi{}.png".format(thresholds[ix1], test_site))


vr_new = np.round(100*abs(an_var[ix1]-an_var_old[ix1])/an_var_old[ix1], 2)

df = pd.DataFrame([vr_new])
df.to_csv(os.path.join(outdir, "variance_reduction_{}m_poi{}.txt".format(thresholds[ix1], test_site)), sep=' ', index=False)

