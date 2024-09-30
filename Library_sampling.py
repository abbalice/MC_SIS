#from Library_SIR import *
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
from numba import njit
# Set a Random seed for reproducibility
np.random.seed()
import time

def get_parameters(fileScen):

# The function will provide three numpy arrays containg the synthetic
# earthquake parameters:
# 1. Magnitude
# 2. Geographic coordinates (Latitude and Longitude of the cells centers)
# 
# retrieved from the scenarios IDs.

# INPUTS:
# fileScen = List of scenarios IDs

  Magn=[]
  Lon=[]
  Lat=[]
  #fileScen = pd.read_csv(fileScen_PS, engine='python', sep='\s+', header=None, index_col=False).loc[:,0].to_numpy()
  for scenario in fileScen:
    scenario = str(scenario)
    if str.isdigit(scenario.split('M')[1].split('_')[0]):
      for_magn=scenario.split('M')[1].split('_')[0]
      for_coord=scenario.split('M')[1].split('_')[1].split('_')[0]
    else:
      for_magn=scenario.split('M')[2].split('_')[0]
      for_coord=scenario.split('M')[2].split('_')[1].split('_')[0]
    for_lat=for_coord[2:6]
    for_lon=for_coord[7:11]
    if str.isdigit(for_magn) and str.isdigit(for_lat) and str.isdigit(for_lon):
      Magn.append(float(for_magn[0]+'.'+for_magn[1:]))
      Lat.append(float(for_lat[:2]+'.'+for_lat[2:]))
      Lon.append(float(for_lon[:2]+'.'+for_lon[2:]))
  return np.array(Magn), np.array(Lat), np.array(Lon)
#-----------------------------------------------------------------------------------------------------------------------

def exceedance_curve(thresholds, max_stage_point, rates):
    nEnsemble = np.size(rates, axis=1)
    # Annual Mean Rate of threshold exceedance
    lambda_exc = np.zeros((len(thresholds), nEnsemble))
    for ith, threshold in enumerate(thresholds):
        ix_sel = np.argwhere(max_stage_point > threshold)[:]
        ix_sel = ix_sel[:,0]
        for j in range(nEnsemble):
            lambda_exc[ith, j] = lambda_exc[ith, j] +  rates[ix_sel,j].sum(axis=0)
    #------------------------------------------------
    return lambda_exc

#------------------------------------------Stratified Sampling----------------------------------------------------------
@njit
def mc_sis(mw, mw_bins, proposal_offshore, scenum):
    list_samples = []
    n_mw = np.zeros_like(mw_bins)
    #----------------
    for imw, mw_bin in enumerate(mw_bins):
       iscen = np.argwhere(mw == mw_bin)[:]
       iscen = iscen[:,0]
       n_bin = round(scenum[imw])
       n_mw[imw] = n_bin
       proposal_bin = proposal_offshore[iscen]
       if (proposal_bin.sum()!=0) and (n_bin!=0):
         w_sis = proposal_bin/np.sum(proposal_bin)
         samples = iscen[np.searchsorted(np.cumsum(w_sis), np.random.rand(n_bin))]
         for isample, sample in enumerate(samples):
           list_samples.append(sample)
    return np.array(list_samples), n_mw

def optimized_mc_sis(mw, mw_bins, mw_counts, thresholds, mean_annual_rates, proposal_offshore, hmax_poi_offshore, scenum):

    n_tot1 = np.zeros_like(mw_counts)
    tmp_num = round(0.25*scenum/len(mw_counts))
    n_tot1 = np.array([tmp_num for i in range(len(mw_counts))])
    mih_optimal = [1, 1.5, 2, 2.5, 3, 3.5, 4, 4.5, 5] 
    n_tot2 = round(0.75*scenum)
    ix_th = np.array([np.argmin(abs(mih-thresholds)) for mih in mih_optimal])

    mih_optimal = thresholds[ix_th]
    n_sis2 = optimized_sis_per_threshold(mw, mw_bins, mih_optimal, mean_annual_rates, proposal_offshore, hmax_poi_offshore, n_tot2)
    n_tot = n_tot1 + np.round(n_sis2.mean(axis=0))
    ix_sis, n_sis = mc_sis(mw, mw_bins, proposal_offshore, n_tot)
    return ix_sis, n_sis

def optimized_sis_per_threshold(mw, mw_bins, mih_optimal, mean_annual_rates, proposal_offshore, hmax_poi_offshore, n_tot):
    n_mw = np.zeros((len(mih_optimal), len(mw_bins)))
    for ith, threshold in enumerate(mih_optimal):
       n_th = optimal_number_samples_per_bin(mw, mw_bins, mean_annual_rates, proposal_offshore, hmax_poi_offshore, threshold, n_tot)
       n_mw[ith, :] = n_th
    return n_mw

def optimal_number_samples_per_bin(mw, mw_bins, mean_annual_rates, proposal_offshore, hmax_poi_offshore, threshold, scenum):
    alpha = np.zeros_like(mw_bins)
    for imw, mw_bin in enumerate(mw_bins):
        tmp_pmw = 0
        tmp_alpha = 0
        iscen = np.argwhere(mw == mw_bin)[:]
        if np.size(iscen, axis=1)>1:
             iscen = iscen.squeeze()
        else:
             iscen = iscen.squeeze(axis=1)
        tot_rates_bin = mean_annual_rates[iscen].sum()
        for ix in iscen:
            if hmax_poi_offshore[ix] > threshold:
              tmp_pmw += mean_annual_rates[ix]
        p_mw = tmp_pmw/tot_rates_bin
        for ix in iscen:
            if hmax_poi_offshore[ix] > threshold:
              w_ss = mean_annual_rates[ix]/tot_rates_bin
              w_sis = proposal_offshore[ix]/np.sum(proposal_offshore[iscen])
              tmp_fi = w_ss/w_sis
            else:
              tmp_fi = 0
              w_sis = proposal_offshore[ix]/np.sum(proposal_offshore[iscen])
            tmp_alpha += ((tmp_fi - p_mw)**2)*w_sis
        alpha[imw] = tmp_alpha*(tot_rates_bin**2)   
    alpha_sqrt = np.sqrt(alpha)
    n_samples = np.zeros_like(mw_bins)
    tot_alpha = alpha_sqrt.sum()
    for imw, mw_bin in enumerate(mw_bins): 
        n_tmp = scenum * (alpha_sqrt[imw]/tot_alpha)
        if np.isnan(n_tmp)==False:
           n_samples[imw] = round(n_tmp)
        else:
           n_samples[imw] = 0
    return n_samples

@njit
def update_rates_sis(mw_bins, mw_ensemble, mw_sis, annual_rates_ensemble, annual_rates_sis, proposal_offshore, proposal_sis, n_mw_sis):
    nEnsemble = 1000
    lmw_over_nmw = np.zeros((len(mw_sis), nEnsemble))
    phi = np.zeros((len(mw_sis), nEnsemble))
    for imw, mw_bin in enumerate(mw_bins):
       iscen_sampled = np.argwhere(mw_sis == mw_bin)[:]
       iscen_ensemble = np.argwhere(mw_ensemble == mw_bin)[:]
       iscen_sampled, iscen_ensemble = iscen_sampled[:,0], iscen_ensemble[:,0]
       tot_rates_bin = annual_rates_ensemble[iscen_ensemble, :].sum(axis=0) 
       tot_proposal_bin = proposal_offshore[iscen_ensemble].sum()
       n_bin = n_mw_sis[imw]
       lmw_over_nmw[iscen_sampled] = tot_rates_bin / n_bin
       for col in range(nEnsemble):
          phi[iscen_sampled,col] = (annual_rates_sis[iscen_sampled, col] / annual_rates_ensemble[iscen_ensemble, col].sum(axis=0)) / (proposal_sis[iscen_sampled] / tot_proposal_bin)
    new_rates_sis = lmw_over_nmw * phi
    new_rates_sis = new_rates_sis.flatten()
    new_rates_sis[np.isnan(new_rates_sis)]=0
    new_rates_sis = np.reshape(new_rates_sis, (len(mw_sis), nEnsemble))
    return new_rates_sis
#-----------------------------------------------Monte Carlo estimates----------------------------------------------------------------------------------------
def cond_mc_variance_sis(mw_ensemble, mw_sampled, mw_bins, annual_rates_sampled, annual_rates_ensemble, hmax_poi_sampled, hmax_poi_ensemble, proposal_ensemble,
                                 hmax_sampled_offshore, proposal_sampled, threshold, n_mw_sampled):
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
        tot_rates_bin = annual_rates_ensemble[iscen_ensemble].sum() #iscen_ensemble
        tot_proposal_bin = proposal_ensemble[iscen_ensemble].sum()
        n_bin = n_mw_sampled[imw]
        if n_bin!=0 and tot_proposal_bin!=0 and tot_rates_bin!=0:
           for ix in iscen_sampled:
              if hmax_poi_sampled[ix] > threshold:
                w_ss = annual_rates_sampled[ix]/tot_rates_bin
                w_sis = proposal_sampled[ix]/tot_proposal_bin
                tmp_fi = w_ss/w_sis
                tmp_q += tmp_fi
           q = tmp_q/n_bin
           for ix in iscen_sampled:
                if hmax_poi_sampled[ix] > threshold:
                   w_ss = annual_rates_sampled[ix]/tot_rates_bin
                   w_sis = proposal_sampled[ix]/tot_proposal_bin
                   tmp_fi = w_ss/w_sis
                else:
                   tmp_fi = 0
                tmp_variance += ((tmp_fi - q)**2)/n_bin
           cond_emp_variance[imw] = tmp_variance*(tot_rates_bin**2)/n_bin
    return cond_emp_variance.sum()

def mc_variance_sis(mw_ensemble, mw_sampled, mw_bins, annual_rates_sampled, annual_rates_ensemble, hmax_poi_sampled, hmax_poi_ensemble, proposal_ensemble,
                                  hmax_sampled_offshore, proposal_sampled, thresholds, n_mw_sampled):
    nthresh = len(thresholds)
    variance_mc = np.zeros((nthresh,))
    for ith, threshold in enumerate(thresholds):
        #print("MIH {}: -------> {}".format(ith, threshold))
        tmp_variance = cond_mc_variance_sis(mw_ensemble, mw_sampled, mw_bins, annual_rates_sampled,
                                                                annual_rates_ensemble, hmax_poi_sampled, hmax_poi_ensemble, proposal_ensemble, hmax_sampled_offshore, proposal_sampled,
                                                                threshold, n_mw_sampled)
        variance_mc[ith] = np.sqrt(tmp_variance)
    return variance_mc

@njit
def cond_anal_variance_importance(mw, mw_bins, mean_annual_rates, hmax_poi_onshore, proposal_offshore, threshold, n_mw):
    nbins = len(mw_bins)
    analytical_sigma2 = np.zeros((nbins,))
    for imw, mw_bin in enumerate(mw_bins):
        tmp_pmw = 0.0
        tmp_alpha = 0.0
        iscen = np.argwhere(mw == mw_bin)[:]
        iscen = iscen[:,0]
        tot_rates_bin = mean_annual_rates[iscen].sum()
        tot_proposal_bin = proposal_offshore[iscen].sum()
        n_bin = n_mw[imw]
        if (tot_rates_bin!=0) and (tot_proposal_bin!=0):
          # prima parte tra parentesi
          for ix in iscen:
            if hmax_poi_onshore[ix] > threshold:
                tmp_pmw += mean_annual_rates[ix]
          p_mw = tmp_pmw/tot_rates_bin
          for ix in iscen:
            if (hmax_poi_onshore[ix] > threshold) and (proposal_offshore[ix]!=0):
              w_ss = mean_annual_rates[ix]/tot_rates_bin
              w_sis = proposal_offshore[ix]/tot_proposal_bin
              tmp_fi = w_ss/w_sis
            else:
              tmp_fi = 0.0
              w_sis = proposal_offshore[ix]/tot_proposal_bin
            tmp_alpha += ((tmp_fi - p_mw)**2)*w_sis
          if n_bin!=0:
            analytical_sigma2[imw] = tmp_alpha*(tot_rates_bin**2)/n_bin
    return analytical_sigma2.sum()

@njit
def analytical_variance_importance(mw, mw_bins, mean_annual_rates, hmax_poi_onshore, proposal_offshore, thresholds, n_mw):
    nthresh = len(thresholds)
    analytical_variance = np.zeros((nthresh,))
    for ith, threshold in enumerate(thresholds):
        #print("MIH {}: ------> {}m".format(ith, threshold))
        tmp_variance = cond_anal_variance_importance(mw, mw_bins, mean_annual_rates, hmax_poi_onshore, proposal_offshore, threshold, n_mw)
        analytical_variance[ith] = np.sqrt(tmp_variance)
    return analytical_variance


#-----------------------------------------------------------------------------------Analytical errors estimate----------------------------------------------------------------------

