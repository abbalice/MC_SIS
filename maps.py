###########################################################
                 # FIGURE 1
###########################################################

from Library_sampling import *
import pygmt
import netCDF4
import xarray as xr


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

plotDir = "VISUALIZATION"

pygmt.config(FONT="16p,Times-Roman",MAP_FRAME_TYPE="plain")

fig = pygmt.Figure()

with fig.subplot(
    nrows=2,
    ncols=1,
    figsize=("10c", "14.5c"),
    margins=["0.1c", "0.3c"],
    title="Full ensemble",
    sharex="b"
):
 
   inpdir="CT"
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

   
   poi_ct = [15.158333, 37.355221]
   poi_sr = [15.325, 37.037843]
   region_map = [12, 33.5, 30, 42]
      
 
   with fig.set_panel(panel=0):

      fig.basemap(
                  region=region_map, projection="M10c",
                  frame=["a5f5"]
                  )
      grid_map = pygmt.datasets.load_earth_relief(
                  resolution="30s",
                  region=region_map
                  )
      # 
      fig.grdimage(grid=grid_map, region=region_map, projection="M10c", cmap="GMT_abyss.cpt",shading=True, transparency=20)
      fig.coast(region=region_map,projection='M10c',shorelines='0.5p,black', land="lightgrey")
      pygmt.makecpt(cmap="GMT_hot.cpt", series=[-10,-3,1], log=True, reverse=True)

      fig.plot(
                 x=df_bs.longitude,
                 y=df_bs.latitude,
                 color=df_bs.sum_weight,
                 cmap=True,
                 style="c0.12c",
                 region=region_map,
                 projection="M10c"
              )

      fig.text(region=region_map, projection="M10c",x=29.5, y=32.5, text="BS", font="15.0p,Times-Bold,black", fill="white")
      position="jTR+w2.6c/4.0c+o1.0c/0.2c"
      with fig.inset(position=position, box="+plightgreen,+gwhite", margin=0.0): 
        inset_region = [xmin_ct -0.05, 15.4, ymin_sr-0.015, ymax_ct+0.015] 
        inundated_ct = [[xmin_ct, ymin_ct, xmax_ct, ymax_ct]]
        inundated_sr = [[xmin_sr, ymin_sr, xmax_sr, ymax_sr]]

        fig.basemap(
                  region=inset_region, projection="M?"
                   )
        grid_map = pygmt.datasets.load_earth_relief(
                   resolution="01s",
                   region=inset_region
                   )
  
        fig.grdimage(grid=grid_map, region = inset_region, projection="M?", cmap="GMT_abyss.cpt", transparency=20)
        fig.coast(region=inset_region,projection='M?',shorelines='0.5p,black', land="lightgrey")
        
        fig.plot(region=inset_region, projection="M?",x=poi_ct[0], y=poi_ct[1], style="x0.18c", pen="1.2p,black", color="magenta4")
        fig.text(region=inset_region, projection="M?",x=poi_ct[0]+0.1, y=poi_ct[1] + 0.12, text="Catania", font="11p,Times-Bold", fill="white")
        fig.plot(region=inset_region, projection="M?",x=poi_sr[0], y=poi_sr[1], style="x0.18c", pen="1.2p,black", color="magenta4")
        fig.text(region=inset_region, projection="M?",x=poi_sr[0]-0.22, y=poi_sr[1] + 0.09, text="Siracusa", font="11p,Times-Bold",fill="white")
        fig.plot(data=inundated_ct, region=inset_region, projection="M?", style="r+s", pen="1.2p,magenta4")
        fig.plot(data=inundated_sr, region=inset_region, projection="M?", style="r+s", pen="1.2p,magenta4")
        
      ###### Box encompassing the large area ######
      rectangle = [[inset_region[0], inset_region[2], inset_region[1], inset_region[3]]]
      fig.plot(data=rectangle, region=region_map, projection="M10c", style="r+s", pen="1p,lightgreen")
      fig.text(region=region_map, projection="M10c", x=13, y=41.2, text="(a)", font="16p,Times-Roman,black", fill="white") 
#======================================================================================================================================================================================================================
   with fig.set_panel(panel=1):

      fig.basemap(
                  region=region_map, projection="M10c",
                  frame=["a5f5"]
                  )
      grid_map = pygmt.datasets.load_earth_relief(
                  resolution="30s",
                  region=region_map
                  )
      #
      fig.grdimage(grid=grid_map, region=region_map, projection="M10c", cmap="GMT_abyss.cpt", shading=True, transparency=20)
      fig.coast(region=region_map,projection='M10c',shorelines='0.5p,black', land="lightgrey")
      pygmt.makecpt(cmap="GMT_hot.cpt", series=[-10, -3, 1], log=True, reverse=True)

      fig.plot(
                 x=df_ps.longitude,
                 y=df_ps.latitude,
                 color=df_ps.sum_weight,
                 cmap=True,
                 style="c0.12c",
                 region=region_map,
                 projection="M10c"
              )

      
      #fig.text(region=region_map, projection="M10c",x=29.5, y=32.5, text="Eastern Sicily", font="8.5p,Times-Bold,black", fill="white")
      fig.text(region=region_map, projection="M10c",x=29.5, y=32.5, text="PS", font="16p,Times-Bold,black", fill="white")
      fig.text(region=region_map, projection="M10c",x=13, y=41.2, text="(b)", font="16p,Times-Roman,black", fill="white")
      fig.plot(data=rectangle, region=region_map, projection="M10c", style="r+s", pen="1p,lightgreen")

   #legend
   fig.legend(spec='legend_map.txt',  position='JTR+o0.8c/7.5c+w0.1c/0.5c')
   fig.colorbar(Q=True,frame=["a1f2g3p", "x+lCumulative annual rate (at each geometrical center)"], position="JMR+o1.6c/2.6c+w11c/0.4c")    
   fig.colorbar(cmap="GMT_abyss.cpt", frame=["x1500f500+lSea depth (m)"], position="JBC+o0.0c/1.5c+w9c/0.4c+h")
fig.savefig(os.path.join(plotDir, "cumulated_annual_rates.png"), dpi=300)
fig.savefig(os.path.join(plotDir, "cumulated_annual_rates.eps")) 
fig.show()

