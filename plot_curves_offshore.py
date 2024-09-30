from Library_sampling import *
from Library_plots import get_sis_data_offshore
import netCDF4
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from mpl_toolkits.basemap import Basemap
from matplotlib.colors import LightSource
import cartopy
from cartopy import config
import cartopy.crs as ccrs
from cartopy.mpl.gridliner import LONGITUDE_FORMATTER, LATITUDE_FORMATTER

#==========================================================================================================================
def local_parser():
    parser = argparse.ArgumentParser(description=sys.argv[0])
    parser.add_argument('--num_samples', default=None, required=True, help='Number of samples')
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
    percentage = local_opts.num_samples
    return int(percentage)
#===========================================================================================================================
percentage = from_local_parser()

# ==========================================================================================
# Retrieve all Data
# ==========================================================================================
matplotlib.rcParams['font.family'] = 'serif'
matplotlib.rcParams['font.sans-serif'] = 'Times'

offshore_poi = ["CT", "SR", "LK"]
city = ["Catania", "Siracusa", "Larnaka"]
letters = ["(a)", "(b)", "(c)", "(d)", "(e)", "(f)", "(g)", "(h)", "(i)"]
fig, axes = plt.subplots(nrows=3, ncols=3, sharex=True, sharey=True, figsize=(21,18), constrained_layout=True)

for i, inpdir in enumerate(offshore_poi):
  # MIH at offshore POI
  pathHeight_OFF = 'HMAX_OFFSHORE/hmax_{}.txt'.format(inpdir)
  data_off = pd.read_csv(pathHeight_OFF, engine='python', sep='\s+')
  hmax_off = data_off["MIH"].to_numpy()
  ids_off = data_off["IDs_sim"].to_numpy()
  mw, lat, lon = get_parameters(ids_off)
  # Coordinates of SIS POIs
  if inpdir=="CT":
    poi_off = pd.read_csv('sis_poi_catania.txt', engine='python', header=None, index_col=False).to_numpy().squeeze()
    region_map_CT = [15, 15.17, 37.3, 37.550]

  elif inpdir=="SR":
    poi_off = pd.read_csv('sis_poi_siracusa.txt', engine='python', header=None, index_col=False).to_numpy().squeeze()
    region_map_SR = [15.15, 15.4, 37, 37.2]

  elif inpdir=="LK":
    poi_off = pd.read_csv('sis_poi_larnaka.txt', engine='python', header=None, index_col=False).to_numpy().squeeze()
    region_map_LK = [33.55, 33.75, 34.85, 35.05]

  lon_sis, lat_sis = poi_off[0], poi_off[1]

  letter1=letters[i]
  letter2=letters[i+3]
  letter3=letters[i+6]
  ax1 = axes.flat[i]
  ax2 = axes.flat[i+3]
  ax3 = axes.flat[i+6]

  thresholds = [1.000e-02, 5.000e-02, 1.000e-01, 2.000e-01,
       3.000e-01, 5.000e-01, 8.000e-01, 1.000e+00,
       1.500e+00, 2.000e+00, 2.500e+00, 3.000e+00,
       3.600e+00, 4.320e+00, 5.180e+00, 6.220e+00,
       7.460e+00, 8.950e+00, 1.074e+01, 1.289e+01,
       1.547e+01, 1.856e+01, 2.227e+01]
  #=====================================================================
  dt = 50 
  lambda_sis_mean, sis_p2, sis_p16, sis_p84, sis_p98, ci_new_up, ci_new_dw, mc_ci_new_up, mc_ci_new_dw, ci_old_up, ci_old_dw, mc_ci_old_up, mc_ci_old_dw = get_sis_data_offshore(percentage,inpdir)
  #=================================================================
  path_exact = "MC_RESULTS"
  df_exact =  pd.read_csv(os.path.join(path_exact, "exact_statistics_{}_offshore.txt".format(inpdir)), engine="python", sep="\s+", index_col=False)
  lambda_true_mean = df_exact["poe"].to_numpy()
  p2 =  df_exact["p2"].to_numpy()
  p16 = df_exact["p16"].to_numpy()
  p84 = df_exact["p84"].to_numpy()
  p98 = df_exact["p98"].to_numpy()

 #=================================================================
  # Plots
  
  true_poe, = ax1.plot(thresholds, lambda_true_mean, color='black', linewidth=2, label='Exact mean')
  sis_poe, = ax1.plot(thresholds, lambda_sis_mean, color='darkblue', linewidth=2, label='SIS mean (new IF)')
  new_if, = ax1.plot(thresholds, ci_new_up, color='darkblue', linestyle='-.', linewidth=1, label='Analytical 95% c.i. (new IF)')
  ax1.plot(thresholds, ci_new_dw, color='darkblue', linestyle='-.', linewidth=1, label='')
  ax1.fill_between(thresholds, ci_new_dw, ci_new_up, color='tab:blue', alpha=0.2)

  old_if, = ax1.plot(thresholds, ci_old_up, color='tab:red', linestyle='-', linewidth=2, label='Analytical 95% c.i. (old IF)')
  ax1.plot(thresholds, ci_old_dw, color='tab:red', linestyle='-', linewidth=2, label='')
  ax1.set_ylim([1e-6, 1]) 
  ax1.set_yscale('log')
  ax1.set_xlim([0, 6])
  ax1.text(x=1, y=3*1e-1, s="N = {}".format(percentage), color="darkblue", fontsize=22)
  ax1.text(x=0.5, y=3*1e-1, s="{}".format(letter1), color="darkblue", fontsize=22)
  ax1.set_title("{}".format(city[i]), fontsize=25)
  
  plt.xticks(fontsize=25)
  #--------------------------
  ax2.text(x=0.5, y=3*1e-1, s="{}".format(letter2), color="darkblue", fontsize=22)
  ax2.fill_between(thresholds, ci_new_dw, ci_new_up, color='tab:blue', alpha=0.2)
  mc_new_if, = ax2.plot(thresholds, mc_ci_new_up, color='hotpink', linestyle='-', linewidth=2, label='Estimated 95% c.i. (new IF)')
  ax2.plot(thresholds, mc_ci_new_dw, color='hotpink', linestyle='-', linewidth=2, label='')
  #---------------------------
  true_percentiles1, = ax3.plot(thresholds, p2, color="darkmagenta", linewidth=2, alpha=0.6, label='Exact p2 - p98')
  ax3.plot(thresholds, p98, color="darkmagenta", linewidth=2, alpha=0.6, label='')
  true_percentiles2, = ax3.plot(thresholds, p16, color="darkorange", linewidth=2, alpha=0.6, label='Exact p16 - p84')
  ax3.plot(thresholds, p84, color="darkorange", linewidth=2, alpha=0.6, label='')
  ax3.plot(thresholds, sis_p2, color="black", linestyle=":", linewidth=2, label='')
  ax3.plot(thresholds, sis_p98, color="black", linestyle=":", linewidth=2, label='')
  sis_percentiles, = ax3.plot(thresholds, sis_p16, color="black", linestyle=":", linewidth=2, label='SIS percentiles')
  ax3.plot(thresholds, sis_p84, color="black", linestyle=":", linewidth=2, label='')
  ax3.text(x=0.5, y=3*1e-1, s="{}".format(letter3), color="darkblue", fontsize=22)
  
  if (i==0):
    ax1.set_ylabel("Annual exceedance-rate", fontsize=25)
    plt.setp(ax1.get_yticklabels(), visible=True, fontsize=25)
    ax2.set_ylabel("Annual exceedance-rate", fontsize=25)
    plt.setp(ax2.get_yticklabels(), visible=True, fontsize=25)
    ax3.set_ylabel("Annual exceedance-rate", fontsize=25)
    plt.setp(ax3.get_yticklabels(), visible=True, fontsize=25)
  elif (i==1) and (i+6 == 7):
    ax3.set_xlabel("Maximum wave amplitude [m]", fontsize=25)
  else:
    plt.setp(ax1.get_yticklabels(), visible=False)
    plt.setp(ax2.get_yticklabels(), visible=False)
    plt.setp(ax3.get_yticklabels(), visible=False)
  plt.setp(ax1.get_xticklabels(), visible=False)
  plt.setp(ax2.get_xticklabels(), visible=False)
  plt.setp(ax3.get_xticklabels(), visible=True, fontsize=25)
  
 #======================================================================

  axins = inset_axes(ax1, width="50%", height="70%", loc="upper right")

  lat_0 = lat.mean()
  lon_0 = lon.mean()
  if inpdir=="CT":
    m = Basemap(llcrnrlon=15, llcrnrlat=37.3, urcrnrlon=15.17, urcrnrlat=37.550,
            lat_0=lat_0, lon_0=lon_0,
            projection="cyl",
            resolution="f")
    parallels = [37.35, 37.5]
    # labels = [left,right,top,bottom]
    m.drawparallels(parallels, labels=[1,0,0,0], fontsize=20)
    meridians = [15.05, 15.15]
    m.drawmeridians(meridians, labels=[0,0,0,1], fontsize=20)
  elif inpdir=="SR":
    m = Basemap(llcrnrlon=15.2, llcrnrlat=37, urcrnrlon=15.4, urcrnrlat=37.2,
            lat_0=lat_0, lon_0=lon_0,
            projection="cyl",
            resolution="f")
    parallels = [37.05, 37.15]
    m.drawparallels(parallels, labels=[1,0,0,0], fontsize=20)
    meridians =  [15.25, 15.35] 
    m.drawmeridians(meridians, labels=[0,0,0,1], fontsize=20)

  elif inpdir =="LK":
    m = Basemap(llcrnrlon=33.55, llcrnrlat=34.85, urcrnrlon=33.75, urcrnrlat=35.05,
            lat_0=lat_0, lon_0=lon_0,
            projection="cyl",
            resolution="f")
    parallels = [34.9, 35]
    m.drawparallels(parallels, labels=[1,0,0,0], fontsize=20)
    meridians = [33.6, 33.7] 
    m.drawmeridians(meridians, labels=[0,0,0,1], fontsize=20)
  
  m.drawcoastlines()
  m.drawrivers()
  m.drawmapboundary(fill_color="#A6CAE0")

  
  m.fillcontinents(color='grey',lake_color='#A6CAE0')
  m.scatter(lon_sis, lat_sis, marker = 'X', color='purple', s=250, label='SIS POI')
  
  plt.legend(loc="upper center", fontsize=20)

ax1 = axes.flat[0]
ax1.legend(handles=[true_poe, sis_poe, old_if, new_if, mc_new_if, true_percentiles1, true_percentiles2, sis_percentiles],  bbox_to_anchor=(0, 1.2, 3.1, 0.3), 
           mode="expand", borderaxespad=0, loc="lower left",
           ncol=4, fontsize=22)

plt.subplots_adjust(wspace=0.3)

fig.tight_layout()
plt.savefig("VISUALIZATION/offshore_curves_N{}.png".format(percentage))#, dpi=300)
plt.show()


