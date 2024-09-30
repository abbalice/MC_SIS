from Library_sampling import *
import seaborn as sns
from scipy import stats

#-----------------------------------------------Inizialization of arguments--------------------------------------------
def local_parser():
    parser = argparse.ArgumentParser(description=sys.argv[0])
    parser.add_argument('--offshore_poi', default=None, required=True, help='Where local S-PTHA is evaluated (e.g. CT, SR,..)')
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
    n_iter = local_opts.mc_samples
    percentage = local_opts.num_samples
    outdir = local_opts.outdir
    return str(inpdir), int(n_iter), int(percentage), str(outdir)

#------------------------------------------------------------------------------------------------------------------------
inpdir, n_iter, percentage, outdir = from_local_parser()
#-------------------------------------------------------------------------------------------------------------------------
# Defining the directory for storing the results
#-------------------------------------------------------------------------------------------------------------------------
outdir = outdir+'_perc' + str(percentage)+ '_{}'.format(inpdir)
#-------------------------------------------------------------------------------------------------------------------------
# Defining directory for outputs

try:
  os.makedirs(os.path.join("MC_RESULTS2", outdir))
except:
  os.path.exists(os.path.join("MC_RESULTS2", outdir))
  print("Directory for outputs already there!")
#----------------------------------------------------------------------------
outdir = os.path.join("MC_RESULTS2", outdir)
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

if (ids_off == ids_scen).all():
   pass
else:
   raise Exception("Data do not coincide")

rates = np.concatenate((rates_bs, rates_ps), axis=0)
mean_annual_rates = rates.mean(axis=1)

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

# Proposal
proposal_offshore = hmax_off * mean_annual_rates
#===========================================
scenum = percentage
ix_sis, n_sis = optimized_mc_sis(mw, mw_bins, mw_counts, thresholds, mean_annual_rates, proposal_offshore, hmax_off, scenum)

# Store samples (with repetitions)

data = {"Sampled_Scenarios": ix_sis}
df_ixsis = pd.DataFrame(data)
df_ixsis.to_csv(os.path.join(outdir, 'index_scenarios_SIH.txt'), sep=' ', index=False)

# Store optimal number of scenarios sampled per magnitude bin
data = {"mwBins": mw_bins, "nSamples": n_sis}
df_nsis = pd.DataFrame(data)
df_nsis.to_csv(os.path.join(outdir, 'optimal_nsamples_SIH.txt'), sep=' ', index=False)

# Update rates
mw_sis = mw[ix_sis]
ids_sis = ids_scen[ix_sis]
rates_sis = rates[ix_sis]
mean_rates_sis = mean_annual_rates[ix_sis]
hmax_off_sis = hmax_off[ix_sis]
proposal_sis = proposal_offshore[ix_sis]

new_rates_sis = update_rates_sis(mw_bins, mw, mw_sis, rates, rates_sis, proposal_offshore, proposal_sis, n_sis)

# Store updated rates
data = new_rates_sis
df_rates_sis = pd.DataFrame(data)
df_rates_sis.to_csv(os.path.join(outdir, 'new_rates_SIH.txt'), sep=' ', header=None, index=False)

# Store scenario ids (without repetition) - for simulations  
ix = np.unique(ix_sis)
data = {"Sampled_Scenarios": ids_scen[ix]}
df_scen_sis = pd.DataFrame(data)
df_scen_sis.to_csv(os.path.join(outdir, 'scenarios_SIH_unique.txt'), sep=' ', index=False)
#---------------------------------------------------------------------------------------------------
# CT 12, SR 4

for i in range(12):
  pathHeight_ON = 'HMAX_ONSHORE2/hmax_onshore_{}{}.txt'.format(inpdir,i)
  data_on = pd.read_csv(pathHeight_ON, engine='python', sep='\s+')
  #data_on["MIH"].replace(np.nan,0)
  hmax_on = data_on["MIH"].to_numpy()
  ids_on = data_on["IDs_sim"].to_numpy()


  lambda_true = exceedance_curve(thresholds, hmax_on, rates)
  lambda_true_mean = lambda_true.mean(axis=1)

  var_sis = analytical_variance_importance(mw, mw_bins, mean_annual_rates, hmax_on, proposal_offshore, thresholds, n_sis)
  hmax_on_sis = hmax_on[ix_sis]
  #mw_sis = mw[ix_sis]
  #time_sis = time[ix_sis]
  #emp_sis_up, emp_sis_dw, lambda_sis, var_sis = empirical_variance_importance(mw, mw_sis, mw_bins,
  #                          mean_rates_sis, mean_annual_rates, hmax_sis, hmax_off, time, hmax_off_sis, time_sis, thresholds, n_sis)
  lambda_sis = exceedance_curve(thresholds, hmax_on_sis, new_rates_sis)
  lambda_sis_mean = lambda_sis.mean(axis=1)
  an_sis_up = lambda_true_mean + 1.96 * var_sis
  an_sis_dw = lambda_true_mean - 1.96 * var_sis

  dt = 50
  lambda_true_mean = 1-np.exp(-lambda_true_mean*dt)
  lambda_sis_mean = 1-np.exp(-lambda_sis_mean*dt)
  #emp_sis_up = 1-np.exp(-emp_sis_up*dt)
  #emp_sis_dw = 1-np.exp(-emp_sis_dw*dt)
  an_sis_up = 1-np.exp(-an_sis_up*dt)
  an_sis_dw = 1-np.exp(-an_sis_dw*dt)
 #---------------------
  #data= {"Thresholds": thresholds, "SIH": lambda_sis,
  #     "AnSx": an_sis_up, "AnDx": an_sis_dw}
  #df_sis = pd.DataFrame(data)
  #df_sis.to_csv(os.path.join(outdir, 'Dataset_SIH_Onshore{}.txt'.format(i+1)), sep=' ', index=False)


  fig, ax = plt.subplots(figsize=(18,15))
  ax.plot(thresholds, lambda_true_mean, color='black', linewidth=2, label='True')
  ax.plot(thresholds, lambda_sis_mean, color='darkred', linestyle='-.', linewidth=2, label='SIS')
  ax.plot(thresholds, an_sis_up, color='black', linewidth=1, linestyle='-.',  label='95% analytical CI')
  ax.plot(thresholds, an_sis_dw, color='black', linewidth=1, linestyle='-.',  label='')
  ax.fill_between(thresholds, an_sis_dw, an_sis_up, color='darkred', alpha=0.2)
  #ax.plot(thresholds, emp_sis_up, color='darkred', linewidth=3, linestyle='-.', alpha=0.3, label='95% analytical CI (offshore)')
  #ax.plot(thresholds, emp_sis_dw, color='darkred', linewidth=3, linestyle='-.', alpha=0.3, label='')

  ax.set_yscale('log')
  ax.set_ylim([5*1e-5, 0])
  ax.set_xlabel("Maximum wave height [m]", fontsize=25)
  ax.set_ylabel("Probability of exceedance (50 years)", fontsize=25)
  plt.legend(fontsize=25)
  plt.xticks(fontsize=25)
  plt.yticks(fontsize=25)
  plt.title("Onshore point {}".format(i+1), fontsize=30)
  plt.savefig(outdir+'/onshore_{}_perc{}.png'.format(i+1,percentage))

