###########################################################
                 # FIGURE 1
###########################################################

from Library_sampling import *
import pygmt
import netCDF4
import xarray as xr

offshore_poi = ["CT", "CT", "LK", "LK"]
#city = ["Catania", "Siracusa", "Larnaka"]

hc_dir_disagg_ct = "DISAGG_MAPS/CT"
#fileBathy
grdfile = os.path.join(hc_dir_disagg_ct,'grid_bathy.grd')
fgrid = xr.open_dataset(grdfile)
xgrid = fgrid['x'].values
ygrid = fgrid['y'].values

xmin_ct = np.amin(xgrid)
xmax_ct = np.amax(xgrid)
ymin_ct = np.amin(ygrid)
ymax_ct = np.amax(ygrid)

hc_dir_disagg_sr = "DISAGG_MAPS/SR"
#fileBathy
grdfile = os.path.join(hc_dir_disagg_sr,'grid_bathy.grd')
fgrid = xr.open_dataset(grdfile)
xgrid = fgrid['x'].values
ygrid = fgrid['y'].values

xmin_sr = np.amin(xgrid)
xmax_sr = np.amax(xgrid)
ymin_sr = np.amin(ygrid)
ymax_sr = np.amax(ygrid)

hc_dir_disagg_lk = "DISAGG_MAPS/LK"
#fileBathy
grdfile = os.path.join(hc_dir_disagg_lk,'grid_bathy.grd')
fgrid = xr.open_dataset(grdfile)
xgrid = fgrid['x'].values
ygrid = fgrid['y'].values

xmin_lk = np.amin(xgrid)
xmax_lk = np.amax(xgrid)
ymin_lk = np.amin(ygrid)
ymax_lk = np.amax(ygrid)

plotDir = "VISUALIZATION"

pygmt.config(FONT="8p,Times-Roman",MAP_FRAME_TYPE="plain")

fig = pygmt.Figure()

with fig.subplot(
    nrows=4,
    ncols=1,
    figsize=("6.5c", "18c"),
    autolabel="(a)",
    margins=["0.1c", "0.3c"],
):
 for i in range(0,4,2):
   inpdir=offshore_poi[i]
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

   rates = np.concatenate((rates_bs, rates_ps), axis=0)
   mean_annual_rates = rates.mean(axis=1)
   mean_annual_rates_bs = rates_bs.mean(axis=1)
   mean_annual_rates_ps = rates_ps.mean(axis=1)

   #==============================
   df_coord_bs = pd.DataFrame(data={"longitude": lon_bs, "latitude": lat_bs, "weight": mean_annual_rates_bs})
   df_coord_bs["sum_weight"] = df_coord_bs.groupby(['longitude', 'latitude'])['weight'].transform(lambda x: x.cumsum()) #x.cumsum()
   df_bs = df_coord_bs

   df_coord_ps = pd.DataFrame(data={"longitude": lon_ps, "latitude": lat_ps, "weight": mean_annual_rates_ps})
   df_coord_ps["sum_weight"] = df_coord_ps.groupby(['longitude', 'latitude'])['weight'].transform(lambda x: x.cumsum()) #x.cumsum()
   df_ps = df_coord_ps

   
   if (inpdir=="CT") or (inpdir=="SR"):
      poi_ct = [15.158333, 37.355221]
      poi_sr = [15.325, 37.037843]
      region_map = [12, 33.5, 30, 42]
   elif (inpdir=="LK"):
      poi_lk = [33.7, 34.953295]
      region_map = [17, 38.5, 30, 42]
      
 
   with fig.set_panel(panel=i):

      fig.basemap(
                  region=region_map, projection="M6c",
                  frame=["ya5f5", "xf"]
                  )
      grid_map = pygmt.datasets.load_earth_relief(
                  resolution="30s",
                  region=region_map
                  )
      # 
      fig.grdimage(grid=grid_map, region=region_map, projection="M6c", cmap="gray", transparency=60)
      fig.coast(region=region_map,projection='M6c',shorelines='0.25p,black')
      pygmt.makecpt(cmap="hot", series=[-10,-3,1], log=True, reverse=True)

      fig.plot(
                 x=df_bs.longitude,
                 y=df_bs.latitude,
                 color=df_bs.sum_weight,
                 cmap=True,
                 style="c0.065c",
                 region=region_map,
                 projection="M6c"
              )

      if (inpdir=="CT") or (inpdir=="SR"):
        fig.text(region=region_map, projection="M6c",x=29.5, y=32.5, text="Eastern Sicily", font="8.5p,Times-Bold,white")
        fig.text(region=region_map, projection="M6c",x=29.5, y=31.5, text="BS", font="9.0p,Times-Bold,white")
        position="jTR+w1.5c/2.3c+o0.9c/0.2c"

      elif inpdir=="LK":
        fig.text(region=region_map, projection="M6c",x=22.5, y=34, text="Larnaka", font="9.5p,Times-Bold,white")
        fig.text(region=region_map, projection="M6c",x=22.5, y=33, text="BS", font="9.0p,Times-Bold,white")
        position= "jTR+w2.0c/1.85c+o0.3c/0.3c"

      with fig.inset(position=position, box="+gwhite", margin=0.03): 
        # 
        if (inpdir=="CT") or (inpdir=="SR"):
          inset_region = [xmin_ct -0.05, 15.4, ymin_sr-0.015, ymax_ct+0.015] 
          inundated_ct = [[xmin_ct, ymin_ct, xmax_ct, ymax_ct]]
          inundated_sr = [[xmin_sr, ymin_sr, xmax_sr, ymax_sr]]

        elif inpdir=="LK":
          inset_region = [xmin_lk - 0.05, xmax_lk + 0.05, ymin_lk - 0.015, ymax_lk + 0.015]
          inundated_lk = [[xmin_lk, ymin_lk, xmax_lk, ymax_lk]]

        fig.basemap(
                  region=inset_region, projection="M?"
                   )
        grid_map = pygmt.datasets.load_earth_relief(
                   resolution="01s",
                   region=inset_region
                   )
        fig.grdcontour(grid=grid_map, region=inset_region, projection="M?", annotation=50, limit=[-150, -50], pen="0.30p,darkorange")
        # 
        fig.grdimage(grid=grid_map, region = inset_region, projection="M?", cmap="gray", transparency=60)
        fig.coast(region=inset_region,projection='M?',shorelines='0.25p,black')
 
        if (inpdir=="CT") or (inpdir=="SR"):
          fig.plot(region=inset_region, projection="M?",x=poi_ct[0], y=poi_ct[1], style="c0.07c", pen=".1p,black", color="magenta4")
          fig.text(region=inset_region, projection="M?",x=poi_ct[0] + 0.05, y=poi_ct[1] + 0.035, text="SIS POI", font="4p,Times-Bold")
          fig.plot(region=inset_region, projection="M?",x=poi_sr[0], y=poi_sr[1], style="c0.08c", pen=".1p,black", color="magenta4")
          fig.text(region=inset_region, projection="M?",x=poi_sr[0], y=poi_sr[1] + 0.035, text="SIS POI", font="4p,Times-Bold")
          fig.plot(data=inundated_ct, region=inset_region, projection="M?", style="r+s", pen="0.30p,magenta4")
          fig.plot(data=inundated_sr, region=inset_region, projection="M?", style="r+s", pen="0.30p,magenta4")

          fig.text(region=inset_region, projection="M?",x=xmin_ct - 0.02, y=ymin_ct + 0.05, text="Catania", angle=90, font="4p,Times-Bold")
          fig.text(region=inset_region, projection="M?",x=xmin_sr, y=ymax_sr + 0.02, text="Siracusa", font="4p,Times-Bold")
          fig.text(region=inset_region, projection="M?",x=xmin_ct+0.04, y=ymin_ct + 0.10, angle=90, text="Inund. PTHA", font="4p,Times-Bold")

        elif inpdir=="LK":
          fig.plot(region=inset_region, projection="M?",x=poi_lk[0], y=poi_lk[1], style="c0.08c", pen=".1p,black", color="magenta4")
          fig.text(region=inset_region, projection="M?",x=poi_lk[0]-0.02, y=poi_lk[1] + 0.02, text="SIS POI", font="5p,Times-Bold")
          fig.plot(data=inundated_lk, region=inset_region, projection="M?", style="r+s", pen="0.30p,magenta4")

          fig.text(region=inset_region, projection="M?",x=xmin_lk+0.04, y=ymin_lk + 0.08, angle=66, text="Inund. PTHA", font="5p,Times-Bold")
      ###### Box encompassing the large area ######
      rectangle = [[inset_region[0], inset_region[2], inset_region[1], inset_region[3]]]
      fig.plot(data=rectangle, region=region_map, projection="M6c", style="r+s", pen="0.5p,blue")

   
#======================================================================================================================================================================================================================
   with fig.set_panel(panel=i+1):

      fig.basemap(
                  region=region_map, projection="M6c",
                  frame=["a5f5"]
                  )
      grid_map = pygmt.datasets.load_earth_relief(
                  resolution="30s",
                  region=region_map
                  )
      #
      fig.grdimage(grid=grid_map, region=region_map, projection="M6c", cmap="gray", transparency=60)
      fig.coast(region=region_map,projection='M6c',shorelines='0.25p,black')
      pygmt.makecpt(cmap="hot", series=[-10, -3, 1], log=True, reverse=True)

      fig.plot(
                 x=df_ps.longitude,
                 y=df_ps.latitude,
                 color=df_ps.sum_weight,
                 cmap=True,
                 style="c0.065c",
                 region=region_map,
                 projection="M6c"
              )

      if (inpdir=="CT") or (inpdir=="SR"):
        fig.text(region=region_map, projection="M6c",x=29.5, y=32.5, text="Eastern Sicily", font="8.5p,Times-Bold,white")
        fig.text(region=region_map, projection="M6c",x=29.5, y=31.5, text="PS", font="9.0p,Times-Bold,white")
      elif inpdir=="LK":
        fig.text(region=region_map, projection="M6c",x=22.5, y=34, text="Larnaka", font="9.5p,Times-Bold,white")
        fig.text(region=region_map, projection="M6c",x=22.5, y=33, text="PS", font="9.0p,Times-Bold,white")
      fig.plot(data=rectangle, region=region_map, projection="M6c", style="r+s", pen="0.5p,blue")
 
 fig.colorbar(Q=True,frame=["a1f2g3p", "x+lCumulative annual rate (at each geometrical center)"], position="JMR+o1.2c/7.5c+w15c/0.3c")     
 fig.savefig(os.path.join(plotDir, "cumulated_annual_rates.png"))#, dpi=447)

fig.show()

