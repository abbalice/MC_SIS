#################################################################################################################
                         # FIGURE 2
#################################################################################################################

from Library_sampling import *
from Library_plots import get_city
import netCDF4
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from mpl_toolkits.basemap import Basemap
from matplotlib.colors import LightSource
import cartopy
from cartopy import config
import cartopy.crs as ccrs
from cartopy.mpl.gridliner import LONGITUDE_FORMATTER, LATITUDE_FORMATTER
#import pyart
import xarray as xr
#-----------------------------------------------Inizialization of arguments--------------------------------------------
def local_parser():
    parser = argparse.ArgumentParser(description=sys.argv[0])
    parser.add_argument('--location', default=None, required=True, help='Where local S-PTHA is evaluated (e.g. CT, SR,..)')
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
    return str(inpdir)

#------------------------------------------Stratified Sampling----------------------------------------------------------
inpdir = from_local_parser()

plotDir = "VISUALIZATION"
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

# Retrieve source parameters 
mw_bs, lon_bs, lat_bs = get_parameters(id_bs)
mw_ps, lon_ps, lat_ps = get_parameters(id_ps)

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
mean_annual_rates_bs = rates_bs.mean(axis=1)
mean_annual_rates_ps = rates_ps.mean(axis=1)

# Retrieve arrival time
data_time = pd.read_csv("arrival_time_{}.txt".format(inpdir), engine="python", sep="\s+")
time_sec = data_time["AT"].to_numpy()
time_min = time_sec/60 # from seconds to hours
if inpdir == "LK":
   tmin = 30
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

lambda_true = exceedance_curve(thresholds, hmax_off, rates)
lambda_true_mean = lambda_true.mean(axis=1)
#-----------------------------------------------------
# Instantaneous sources
mean_annual_rates1 = mean_annual_rates[time_min<=1]
rates1 = rates[time_min<=1]
hmax_off1 = hmax_off[time_min<=1]

p_sigma1 = np.zeros((len(mean_annual_rates), len(thresholds)))

for ith, threshold in enumerate(thresholds):
    indSel = np.argwhere(hmax_off1 > threshold)[:]
    indSel = indSel[:,0]
    p_sigma1[indSel, ith] += mean_annual_rates1[indSel]/lambda_true_mean[ith]
#-------------------------------------------------------
# Local sources
mean_annual_rates2 = mean_annual_rates[(time_min>1) & (time_min<30)]
rates2 = rates[(time_min>1) & (time_min<30)]
hmax_off2 = hmax_off[(time_min>1) & (time_min<30)]
time2 = time_min[(time_min>1) & (time_min<30)]

p_sigma2 = np.zeros((len(mean_annual_rates), len(thresholds)))

for ith, threshold in enumerate(thresholds):
    indSel = np.argwhere(hmax_off2 > threshold)[:]
    indSel = indSel[:,0]
    p_sigma2[indSel, ith] += mean_annual_rates2[indSel]/lambda_true_mean[ith]

#-------------------------------------------------------
# Regional  sources
mean_annual_rates3 = mean_annual_rates[time_min>30]
rates3 = rates[time_min>30]
hmax_off3 = hmax_off[time_min>30]
time3 = time_min[time_min>30]
p_sigma3 = np.zeros((len(mean_annual_rates), len(thresholds)))

for ith, threshold in enumerate(thresholds):
    indSel = np.argwhere(hmax_off3 > threshold)[:]
    indSel = indSel[:,0]
    p_sigma3[indSel, ith] += mean_annual_rates3[indSel]/lambda_true_mean[ith]
    #print(time3[indSel])

p_sigma1 = np.sum(p_sigma1, axis=0)
p_sigma2 = np.sum(p_sigma2, axis=0)
p_sigma3 = np.sum(p_sigma3, axis=0)

matplotlib.rcParams['font.family'] = 'serif'
matplotlib.rcParams['font.sans-serif'] = 'Times'

city = get_city(inpdir)
data = {"RS": p_sigma3[:15]*100, "LS": p_sigma2[:15]*100, "IS": p_sigma1[:15]*100}
df = pd.DataFrame(data, index=thresholds[:15])
colors = ['purple', 'plum', 'lightblue']
ax = df.plot.barh(align='center', stacked=True, figsize=(10, 6), color=colors, edgecolor="black")

font_color = 'k'

for label in (ax.get_xticklabels() + ax.get_yticklabels()):
    label.set_fontsize(18)
plt.xticks(color=font_color, fontsize=18)
plt.yticks(color=font_color, fontsize=18)

if inpdir=="LK":
   letter = "c"
   plt.xlabel('Hazard contribution [%]', fontsize=18)
else:
   plt.xticks([])
plt.ylabel('Maximum wave amplitude [m]', fontsize=18)

if inpdir=="CT":
   letter="a"
   
   plt.subplots_adjust(top=0.8)
   title = plt.title('{}'.format(city), pad=22, fontsize=20, color=font_color)
   title.set_position([.5, 1.08])
   legend = plt.legend(loc='center',
       frameon=False,
       bbox_to_anchor=(0., 1.15, 1., .102), 
       mode='expand', 
       ncol=3, 
       borderaxespad=-.46,
       prop={'size': 15})
   for text in legend.get_texts():
       plt.setp(text, color=font_color) # legend font color
else:
   plt.legend()
   ax.get_legend().remove()
   title = plt.title('{}'.format(city), pad=22, fontsize=20, color=font_color)
   title.set_position([.5, 1.08])

if inpdir=="SR":
   letter = "b"
ax.text(-0.1, 1.08, "({})".format(letter), fontsize=18, color='black', transform=ax.transAxes,    ##000000
             bbox=dict(facecolor='#ffffff', edgecolor='black', pad=4.0))

plt.xlim([0, 100])
          
plt.savefig(os.path.join(plotDir, "disaggregation_{}.png".format(inpdir)))#, dpi=300)
plt.show()


