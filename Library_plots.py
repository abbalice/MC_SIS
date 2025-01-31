import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sys
import argparse
from netCDF4 import Dataset
import os
import matplotlib
import warnings
warnings.filterwarnings("ignore")
import random
#from numba import njit
import time
import netCDF4
import xarray as xr
#import mpu


def get_distance(lat_bs, lat_ps, lon_bs, lon_ps, lat_poi, lon_poi):
   dist_bs = []
   dist_ps = []

   for i in range(len(lon_bs)):
     dist_bs.append(mpu.haversine_distance((lat_bs[i], lon_bs[i]), (lat_poi, lon_poi)))
   for i in range(len(lon_ps)):
     dist_ps.append(mpu.haversine_distance((lat_ps[i], lon_ps[i]), (lat_poi, lon_poi)))
   return dist_bs, dist_ps

def get_seismicity_new_rates(index, len_bs, len_ps, new_rates):
   index_bs = [ix for ix in index if ix<=len_bs]
   index_ps = [ix for ix in index if ix>len_bs]
   print(len(index), len_bs, len_ps)
   new_rates_bs = new_rates[index<=len_bs, :]
   new_rates_ps = new_rates[index>len_bs, :]

   return np.array(index_bs), np.array(index_ps), new_rates_bs, new_rates_ps


def retrieve_sampling_data(pathMC, lat_disagg, lon_disagg, len_bs, len_ps):
  # Retrieve optimal number of scenarios sampled per magnitude bin(new_sis)
  new_rates = pd.read_csv(os.path.join(pathMC, 'new_rates_SIH.txt'), sep='\s+', engine='python', header=None, index_col=False).to_numpy()
  tmp_index =  pd.read_csv(os.path.join(pathMC, 'index_scenarios_SIH.txt'), sep='\s+', engine='python', index_col=False)
  scen_index = tmp_index["Sampled_Scenarios"].to_numpy()
  
  mean_new_rates = new_rates.mean(axis=1)
  scen_index_bs, scen_index_ps, new_rates_bs, new_rates_ps = get_seismicity_new_rates(scen_index, len_bs, len_ps, new_rates)

  lon_bs, lat_bs = lon_disagg[scen_index_bs], lat_disagg[scen_index_bs]
  lon_ps, lat_ps = lon_disagg[scen_index_ps], lat_disagg[scen_index_ps]
  return lat_bs, lon_bs, lat_ps, lon_ps, new_rates_bs, new_rates_ps

def cumulate_rates(lon, lat, rates):
  df = pd.DataFrame(data={"longitude": lon, "latitude": lat, "rates": rates.mean(axis=1)})
  df["rates"] = df.groupby(['longitude', 'latitude'])['rates'].transform(lambda x: x.cumsum())
  return df

def get_city(inpdir):
  if inpdir=="CT":
    city = "Catania"
  elif inpdir=="SR":
    city = "Siracusa"
  return city


def panel(fig, inpdir, region_map, df_disagg, df_sis, IF, N, panel, letter):
    import pygmt
    fig.basemap(
               region=region_map, projection="M10c",
               frame=["a5f5"]
               )
    grid_map = pygmt.datasets.load_earth_relief(
              resolution="30s",
              region=region_map
              )
    # Plot the downloaded grid with color-coding based on the elevation
    if (panel==1) or (panel==2):
       fig.grdimage(grid=grid_map, region=region_map, projection="M10c", cmap="GMT_abyss.cpt", shading=True, transparency=20, frame=["+t{} IF".format(IF)])
    else:
       fig.grdimage(grid=grid_map, region=region_map, projection="M10c", cmap="GMT_abyss.cpt", shading=True, transparency=20)
    fig.coast(region=region_map,projection='M10c',shorelines='0.5p,black', land="lightgrey")
    hc_dir_disagg = "DISAGG_MAPS/{}".format(inpdir)
    #fileBathy
    grdfile = os.path.join(hc_dir_disagg,'grid_bathy.grd')
    fgrid = xr.open_dataset(grdfile)
    xgrid = fgrid['x'].values
    ygrid = fgrid['y'].values

    xmin = np.amin(xgrid)
    xmax = np.amax(xgrid)
    ymin = np.amin(ygrid)
    ymax = np.amax(ygrid)
    
    inset_region = [xmin -0.05, 15.4, ymin-0.015, ymax+0.035] 
    inundated_area = [[xmin, ymin, xmax, ymax]]
  
    if panel==1:
     
       city = get_city(inpdir)
       poi_off = pd.read_csv('sis_poi_{}.txt'.format(city), engine='python', header=None, index_col=False).to_numpy().squeeze()
       poi = poi_off[0], poi_off[1]
       position="jTR+w3.8c/2.5c+o1.0c/0.2c"
 
       with fig.inset(position=position, box="+plightgreen,+gwhite", margin=0.0): 
         fig.basemap(
                   region=inset_region, projection="M?"
                   )
         grid_map = pygmt.datasets.load_earth_relief(
                   resolution="01s",
                   region=inset_region
                   )
  
         fig.grdimage(grid=grid_map, region = inset_region, projection="M?", cmap="GMT_abyss.cpt", transparency=20)
         fig.coast(region=inset_region,projection='M?',shorelines='0.5p,black', land="lightgrey")
        
         fig.plot(region=inset_region, projection="M?",x=poi[0], y=poi[1], style="x0.2c", pen="1.5p,black", color="magenta4")
         if inpdir=="CT":
            fig.text(region=inset_region, projection="M?",x=poi[0]+0.1, y=poi[1] + 0.12, text="Catania", font="11p,Times-Bold", fill="white")
         else:
            fig.text(region=inset_region, projection="M?",x=poi[0]-0.09, y=poi[1] + 0.066, text="Siracusa", font="11p,Times-Bold",fill="white")

         fig.plot(data=inundated_area, region=inset_region, projection="M?", style="r+s", pen="0.8p,magenta4")
 

    fig.plot(
            x=df_disagg.longitude,
            y=df_disagg.latitude,
            color="240/210/255",
            pen="200/160/220", #230/230/250",
            transparency=20,
            style="c0.12c",
            region=region_map,
            projection="M10c"
            )

    pygmt.makecpt(cmap="GMT_hot.cpt", series=[-10, -3, 1], log=True, reverse=True)

    fig.plot(
            x=df_sis.longitude,
            y=df_sis.latitude,
            color=df_sis.rates,
            cmap=True,
            style="c0.12c",
            pen="black",
            region=region_map,
            projection="M10c"
            )
    #if IF=='old':
    #  fig.text(region=region_map, projection="M6c",x=29.5, y=33.5, text="{} (Old IF)".format(seismicity), font="8.5p,Times-Bold,black", fill="white")
    #else:
    #  fig.text(region=region_map, projection="M6c",x=29.5, y=33.5, text="{} (New IF)".format(seismicity), font="8.5p,Times-Bold,black", fill="white")
    fig.text(region=region_map, projection="M10c",x=29.5, y=31.5, text="N = {}".format(N), font="15p,Times-Bold,black", fill="white")
    fig.text(region=region_map, projection="M10c",x=13, y=41.2, text="{}".format(letter), font="15p,Times-Roman,black", fill="white") 
    ###### Box encompassing the large area ######
    rectangle = [[inset_region[0], inset_region[2], inset_region[1], inset_region[3]]]
    fig.plot(data=rectangle, region=region_map, projection="M10c", style="r+s", pen="1.2p,lightgreen")
  
    return

def get_sis_data_onshore(n, inpdir, poi):

  pathNew = "MC_RESULTS/results_perc{}_{}".format(n, inpdir)
  pathOld = "MC_RESULTS/gareth_perc{}_{}".format(n, inpdir)

  df_new = pd.read_csv(os.path.join(pathNew, "statistics_onshore_poi_{}.txt".format(poi)), engine="python", sep="\s+", index_col=False)
  lambda_sis_mean = df_new["poe"].to_numpy()
  sis_p2 =  df_new["p2"].to_numpy()
  sis_p16 =  df_new["p16"].to_numpy()
  sis_p84 =  df_new["p84"].to_numpy()
  sis_p98 =  df_new["p98"].to_numpy()
  
  an_var_new = df_new["AnVar"].to_numpy()

  ci_new_up = df_new["CIanUp"].to_numpy()
  ci_new_dw = df_new["CIanDW"].to_numpy()

  mc_ci_new_up = df_new["CIempUp"].to_numpy()
  mc_ci_new_dw = df_new["CIempDW"].to_numpy()

  df_old = pd.read_csv(os.path.join(pathOld, "statistics_onshore_poi_{}.txt".format(poi)), engine="python", sep="\s+", index_col=False)
  
  an_var_old = df_old["AnVar"].to_numpy()

  ci_old_up = df_old["CIanUp"].to_numpy()
  ci_old_dw = df_old["CIanDW"].to_numpy()

  mc_ci_old_up = df_old["CIempUp"].to_numpy()
  mc_ci_old_dw = df_old["CIempDW"].to_numpy()
  return lambda_sis_mean, sis_p2, sis_p16, sis_p84, sis_p98, an_var_new, ci_new_up, ci_new_dw, mc_ci_new_up, mc_ci_new_dw, an_var_old, ci_old_up, ci_old_dw, mc_ci_old_up, mc_ci_old_dw

def get_sis_data_offshore(n, inpdir):

  pathNew = "MC_RESULTS/results_perc{}_{}".format(n, inpdir)
  pathOld = "MC_RESULTS/gareth_perc{}_{}".format(n, inpdir)

  df_new = pd.read_csv(os.path.join(pathNew, "statistics_offshore.txt"), engine="python", sep="\s+", index_col=False)
  lambda_sis_mean = df_new["poe"].to_numpy()
  sis_p2 =  df_new["p2"].to_numpy()
  sis_p16 =  df_new["p16"].to_numpy()
  sis_p84 =  df_new["p84"].to_numpy()
  sis_p98 =  df_new["p98"].to_numpy()

  ci_new_up = df_new["CIanUp"].to_numpy()
  ci_new_dw = df_new["CIanDW"].to_numpy()

  mc_ci_new_up = df_new["CIempUp"].to_numpy()
  mc_ci_new_dw = df_new["CIempDW"].to_numpy()

  df_old = pd.read_csv(os.path.join(pathOld, "statistics_offshore.txt"), engine="python", sep="\s+", index_col=False)

  ci_old_up = df_old["CIanUp"].to_numpy()
  ci_old_dw = df_old["CIanDW"].to_numpy()

  mc_ci_old_up = df_old["CIempUp"].to_numpy()
  mc_ci_old_dw = df_old["CIempDW"].to_numpy()
  return lambda_sis_mean, sis_p2, sis_p16, sis_p84, sis_p98, ci_new_up, ci_new_dw, mc_ci_new_up, mc_ci_new_dw, ci_old_up, ci_old_dw, mc_ci_old_up, mc_ci_old_dw

