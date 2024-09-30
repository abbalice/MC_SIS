from Library_sampling import exceedance_curve, get_parameters, mc_sis, optimized_mc_sis#*
import numpy as np
import argparse
import os
import sys
import pandas as pd
from Library_plots import *
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

#-----------------------------------------------Inizialization of arguments--------------------------------------------
def local_parser():
    parser = argparse.ArgumentParser(description=sys.argv[0])
    parser.add_argument('--offshore_poi', default=None, required=True, help='Where local S-PTHA is evaluated (e.g. CT, SR,..)')
    parser.add_argument('--test_site', default=None, required=True, help='Where local S-PTHA is evaluated (e.g. CT, SR,..)')
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
    percentage = local_opts.num_samples
    outdir = local_opts.outdir
    return str(inpdir), int(test_site), int(percentage), str(outdir)

#------------------------------------------------------------------------------------------------------------------------
inpdir, test_site, percentage, outdir = from_local_parser()
#-------------------------------------------------------------------------------------------------------------------------
matplotlib.rcParams['font.family'] = 'serif'
matplotlib.rcParams['font.sans-serif'] = 'Times'

# Defining the directory for storing the results
#-------------------------------------------------------------------------------------------------------------------------
outdir = outdir+'_{}'.format(inpdir)
#-------------------------------------------------------------------------------------------------------------------------
# Defining directory for outputs

try:
  os.makedirs(os.path.join("MC_RESULTS2", outdir))
except:
  os.path.exists(os.path.join("MC_RESULTS2", outdir))
  print("Directory for outputs already there!")
outdir = os.path.join("MC_RESULTS2", outdir)

# Retrieve scenario rates
if (inpdir == "CT") or (inpdir=="SR"):
   fileRates_bs = "/work/volpe/PTHA_old/SICILIA_ORIENTALE/Scenarios/from_TSUMAPS1.1/refinement_BS/final_list_BS_probs99ALL.txt"
   fileRates_ps = "/work/volpe/PTHA_old/SICILIA_ORIENTALE/Scenarios/from_TSUMAPS1.1/refinement_PS/final_list_PS_probs99ALL.txt"
elif (inpdir == "LK"):
   fileRates_bs = "/nas/bandabardo/cat/CIPRO/scenario_list_BS_ALL_IDs98.probs"
   fileRates_ps = "/nas/bandabardo/cat/CIPRO/scenario_list_PS_ALL_IDs98.probs"

prob_bs = pd.read_csv(fileRates_bs, engine='python', sep=',', header=None, index_col=False)
prob_ps = pd.read_csv(fileRates_ps, engine='python', sep=',', header=None, index_col=False)

id_bs = prob_bs.loc[:, 0].to_numpy(dtype='str')
id_ps = prob_ps.loc[:, 0].to_numpy(dtype='str')

rates_bs = prob_bs.loc[:,1:].to_numpy()
rates_ps = prob_ps.loc[:,1:].to_numpy()

# Check
ids_scen = np.concatenate((id_bs, id_ps), axis=0)
rates = np.concatenate((rates_bs, rates_ps), axis=0)
mean_annual_rates = rates.mean(axis=1)

# Thresholds to be exceeded
thresholds = [1.000e-02, 5.000e-02, 1.000e-01, 2.000e-01,
       3.000e-01, 5.000e-01, 8.000e-01, 1.000e+00,
       1.500e+00, 2.000e+00, 2.500e+00, 3.000e+00,
       3.600e+00, 4.320e+00, 5.180e+00, 6.220e+00,
       7.460e+00, 8.950e+00, 1.074e+01, 1.289e+01,
       1.547e+01, 1.856e+01, 2.227e+01]


#===========================================
pathHeight_ON = 'HMAX_ONSHORE_New/hmax_onshore_{}{}.txt'.format(inpdir,test_site)
data_on = pd.read_csv(pathHeight_ON, engine='python', sep='\s+')
#data_on["MIH"].replace(np.nan,0)
hmax_on = data_on["MIH"].to_numpy()


#---------------------------------------------------------------------------------------
#  Original
lambda_true = exceedance_curve(thresholds, hmax_on, rates)
#lambda_true_mean = lambda_true.mean(axis=1)
p2 = np.percentile(lambda_true, 2, axis=1)
p16 = np.percentile(lambda_true, 16, axis=1)
p84 = np.percentile(lambda_true, 84, axis=1)
p98 = np.percentile(lambda_true, 98, axis=1)

p2 = 1-np.exp(-p2*50)
p16 = 1-np.exp(-p16*50)
p84 = 1-np.exp(-p84*50)
p98 = 1-np.exp(-p98*50)

#---------------------------------
# SIS
lambda_sis_mean, sis_p2, sis_p16, sis_p84, sis_p98, _, ci_new_up, ci_new_dw, mc_ci_new_up, mc_ci_new_dw, _, ci_old_up, ci_old_dw, mc_ci_old_up, mc_ci_old_dw = get_sis_data_onshore(percentage, inpdir, test_site)
lambda_sis_mean = 1-np.exp(-lambda_sis_mean*50)
sis_p2 = 1-np.exp(-sis_p2*50)
sis_p16 = 1-np.exp(-sis_p16*50)
sis_p84 = 1-np.exp(-sis_p84*50)
sis_p98 = 1-np.exp(-sis_p98*50)
mc_ci_new_up = 1-np.exp(-mc_ci_new_up*50)
mc_ci_new_dw = 1-np.exp(-mc_ci_new_dw*50)

city = get_city(inpdir)

fig, ax = plt.subplots(figsize=(25,21))
sis_mean, = ax.plot(thresholds, lambda_sis_mean, color="darkblue", linewidth=3.5, linestyle='-', label='SIS mean PoE')

true_percentiles1, = ax.plot(thresholds, p2, color="darkmagenta", linewidth=3.5, alpha=0.6, label='Exact p2 - p98')
ax.plot(thresholds, p98, color="darkmagenta", linewidth=3.5, alpha=0.6, label='')
true_percentiles2, = ax.plot(thresholds, p16, color="darkorange", linewidth=3.5, alpha=0.6, label='Exact p16 - p84')
ax.plot(thresholds, p84, color="darkorange", linewidth=3.5, alpha=0.6, label='')
ax.plot(thresholds, sis_p2, color="black", linestyle=":", linewidth=3.5, label='SIS percentiles PoE')
ax.plot(thresholds, sis_p98, color="black", linestyle=":", linewidth=3.5, label='')
sis_percentiles, = ax.plot(thresholds, sis_p16, color="black", linestyle=":", linewidth=3.5, label='')
ax.plot(thresholds, sis_p84, color="black", linestyle=":", linewidth=3.5, label='')
ax.fill_between(thresholds, mc_ci_new_dw, mc_ci_new_up, color="hotpink", alpha=0.2, label="Estimated 95% c.i. (new IF)")
ax.axhline(y=1e-1, color="green", linestyle="-.", linewidth=2, alpha=0.5)
ax.axhline(y=1e-2, color="green", linestyle="-.", linewidth=2, alpha=0.5)
ax.axhline(y=5e-3, color="green", linestyle="-.", linewidth=2, alpha=0.5)


ax.set_yscale('log')
ax.set_ylim([5*1e-4, 1])
if (inpdir=="CT") or (inpdir=="SR"):
  ax.set_xlim([0, 7.5])
else:
  ax.set_xlim([0,5])
plt.setp(ax.get_xticklabels(), visible=True, fontsize=45)

if percentage==1500:
   ax.set_yticks(ticks=[1e-1, 2e-2, 5e-3], labels=["10%", "2%", "0.5%"])
   ax.tick_params(axis="y", left=True, right=False, labelleft=True, labelright=False)
   plt.setp(ax.get_yticklabels(), visible=True, fontsize=42)
   ax.set_ylabel("PoE (50 years)", fontsize=42)
elif percentage==6000:
   
   ax.set_yticks(ticks=[1e-1, 1e-2, 5e-3], labels=["475", "2475", "99975"])
   ax.tick_params(axis="y", left=False, right=True, labelleft=False, labelright=True, labelcolor="green")
   plt.setp(ax.get_yticklabels(), visible=True, fontsize=45)
   ax.set_ylabel("ARP [y]", fontsize=42, color="green")
   ax.yaxis.set_label_position("right")
else:
   plt.setp(ax.get_yticklabels(), visible=False)
ax.set_xlabel("Flowdepth [m]", fontsize=45)

#plt.tight_layout()
plt.legend(fontsize=38, loc="upper right", frameon=False)
#ax.set_title("{}, N = {}".format(city, percentage), fontsize=25)
plt.savefig(os.path.join(outdir, "poi_{}_perc{}.png".format(test_site, percentage)))
#plt.show()


