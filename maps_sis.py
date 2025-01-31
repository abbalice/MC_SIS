############################################################################################################################
                               # FIGURES 3 and 4
###########################################################################################################################

from Library_sampling import *
from Library_plots import *
import pygmt

def local_parser():
    parser = argparse.ArgumentParser(description=sys.argv[0])
    parser.add_argument('--location', default=None, required=True, help='Where local S-PTHA is evaluated (e.g. CT, SR,..)')
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
#---------------------------------------------------------------------------------------------------------------------------
inpdir = from_local_parser()
   
#===========================================================================================================================
city = get_city(inpdir)

plotDir = "VISUALIZATION"
# Retrieve offshore height
pathHeight_OFF = 'HMAX_OFFSHORE/hmax_{}.txt'.format(inpdir)
data_off = pd.read_csv(pathHeight_OFF, engine='python', sep=' ')
hmax_off = data_off["MIH"].to_numpy()
ids_off = data_off["IDs_sim"].to_numpy()

# Retrieve source parameters
mw, lon, lat = get_parameters(ids_off)

# Retrieve scenario rates
fileRates_bs = "probs_BS_Sicily.txt"
fileRates_ps = "probs_PS_Sicily.txt"

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

len_bs = len(mean_annual_rates_bs)-1
len_ps = len(mean_annual_rates_ps)-1

#==============================
num_scen = [1500, 3000, 6000]

for n in num_scen:
  
  pathNew = "MC_RESULTS/results_perc{}_{}".format(n, inpdir)
  pathOld = "MC_RESULTS/gareth_perc{}_{}".format(n, inpdir)
  
  lat_bs_new, lon_bs_new, lat_ps_new, lon_ps_new, new_rates_bs, new_rates_ps = retrieve_sampling_data(pathNew, lat, lon, len_bs, len_ps)
  lat_bs_old, lon_bs_old, lat_ps_old, lon_ps_old, old_rates_bs, old_rates_ps = retrieve_sampling_data(pathOld, lat, lon, len_bs, len_ps)
  df_bs_new = cumulate_rates(lon_bs_new, lat_bs_new, new_rates_bs)
  df_ps_new = cumulate_rates(lon_ps_new, lat_ps_new, new_rates_ps)
  df_bs_old = cumulate_rates(lon_bs_old, lat_bs_old, old_rates_bs)
  df_ps_old = cumulate_rates(lon_ps_old, lat_ps_old, old_rates_ps)
  globals()['df_bs_new_{}'.format(n)] = df_bs_new
  globals()['df_ps_new_{}'.format(n)] = df_ps_new
  globals()['df_bs_old_{}'.format(n)] = df_bs_old
  globals()['df_ps_old_{}'.format(n)] = df_ps_old

df_bs = cumulate_rates(lon_bs, lat_bs, rates_bs)
df_ps = cumulate_rates(lon_ps, lat_ps, rates_ps)
#=======================================================================================================================
region_map = [12, 33.5, 30, 42]
#--------------------
pygmt.config(FONT="15p,Times-Roman",MAP_FRAME_TYPE="plain")
pygmt.show_versions()
fig = pygmt.Figure()

with fig.subplot(
    nrows=3,
    ncols=2,
    figsize=("21c", "23.5c"),
    #autolabel="(a)",
    sharex="b",
    sharey="l",
    margins=["0.1c", "0.0c"]
):
   with fig.set_panel(panel=0):
      panel(fig, inpdir, region_map, df_bs, df_bs_old_1500, 'Old', 1500, 1, "(a)")
   with fig.set_panel(panel=1):
     panel(fig, inpdir, region_map, df_bs,  df_bs_new_1500, 'New', 1500, 2, "(b)")
   with fig.set_panel(panel=2):
     panel(fig, inpdir, region_map, df_bs,  df_bs_old_3000, 'Old', 3000, 3, "(c)")
   with fig.set_panel(panel=3):
     panel(fig, inpdir, region_map, df_bs,  df_bs_new_3000, 'New', 3000, 4, "(d)")
   with fig.set_panel(panel=4):
     panel(fig, inpdir, region_map, df_bs,  df_bs_old_6000, 'Old', 6000, 5, "(e)")
   with fig.set_panel(panel=5):
     panel(fig, inpdir, region_map, df_bs,  df_bs_new_6000, 'New', 6000, 6, "(f)")
   
   fig.legend(spec="legend_map_sis_BS.txt", position="JTR+o0.8c/13.5c+w0.1c/0.5c")
   fig.colorbar(Q=True,frame=["a1f2g3p", "x+lCumulative annual rate (at each geometrical center)", "y+lSelected BS"], position="JMR+o1.7c/5.5c+w15c/0.4c")
   fig.colorbar(cmap="GMT_abyss.cpt", frame=["x1500f500+lSea depth (m)"], position="JBC+w10c/0.4c+h")

#fig.savefig(os.path.join("VISUALIZATION", "map_sis_{}.png".format(inpdir)))#, dpi=447)
fig.savefig(os.path.join("VISUALIZATION", "map_sis_{}_BS.eps".format(inpdir)))
fig.show()
#======================================================================================================================================================================================================================
pygmt.config(FONT="15p,Times-Roman", FONT_TITLE="16p,Times-Bold", MAP_FRAME_TYPE="plain")
pygmt.show_versions()
fig = pygmt.Figure()

with fig.subplot(
    nrows=3,
    ncols=2,
    figsize=("21c", "23.5c"),
    #autolabel="(a)",
    sharex="b",
    sharey="l",
    margins=["0.1c", "0.0c"]
):

   with fig.set_panel(panel=0):
      panel(fig, inpdir, region_map, df_ps, df_ps_old_1500, 'Old', 1500, 1, "(a)")
   with fig.set_panel(panel=1):
     panel(fig, inpdir, region_map, df_ps,  df_ps_new_1500, 'New', 1500, 2, "(b)")
   with fig.set_panel(panel=2):
     panel(fig, inpdir, region_map, df_ps,  df_ps_old_3000, 'Old', 3000, 3, "(c)")
   with fig.set_panel(panel=3):
     panel(fig, inpdir, region_map, df_ps,  df_ps_new_3000, 'New', 3000, 4, "(d)")
   with fig.set_panel(panel=4):
     panel(fig, inpdir, region_map, df_ps,  df_ps_old_6000, 'Old', 6000, 5, "(e)")
   with fig.set_panel(panel=5):
     panel(fig, inpdir, region_map, df_ps,  df_ps_new_6000, 'New', 6000, 6, "(f)")
   fig.legend(spec="legend_map_sis_PS.txt", position="JTR+o0.8c/13.5c+w0.1c/0.5c")
   fig.colorbar(Q=True,frame=["a1f2g3p", "x+lCumulative annual rate (at each geometrical center)", "y+lSelected PS"], position="JMR+o1.7c/5.5c+w15c/0.4c")
   fig.colorbar(cmap="GMT_abyss.cpt", frame=["x1500f500+lSea depth (m)"], position="JBC+w10c/0.4c+h")

#fig.savefig(os.path.join("VISUALIZATION", "map_sis_{}.png".format(inpdir)))#, dpi=447)
fig.savefig(os.path.join("VISUALIZATION", "map_sis_{}_PS.eps".format(inpdir)))
fig.show()
