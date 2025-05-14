"""
Functions for filtering and smoothing tracking data.
"""

import numpy as np
import pandas as pd
import scipy as sp
from swap_correction import utils, metrics

def filter_data(rawData : pd.DataFrame) -> pd.DataFrame:
    '''Apply the default filter used by the analysis pipeline'''
    return filter_gaussian(rawData,3)
    #return filter_sgolay(data,45,4)
    #return filter_median(data,10)

def filter_sgolay(rawData : pd.DataFrame, window : int = 45, order : int = 4) -> pd.DataFrame:
    '''Apply Savitzky-Golay filter to the position data'''
    data = rawData.copy()

    for col in utils.flatten(metrics.POSDICT.values()):
        data[col] = sp.signal.savgol_filter(data[col].to_numpy(), window, order)
    
    return data

def filter_gaussian(rawData : pd.DataFrame, sigma : float = 3) -> pd.DataFrame:
    '''Apply Gaussian filter to the position data'''
    data = rawData.copy()

    for col in utils.flatten(metrics.POSDICT.values()):
        data[col] = sp.ndimage.gaussian_filter1d(data[col].to_numpy(),sigma)
    
    return data

def filter_meanmed(rawData : pd.DataFrame, medWin : int = 15, meanWin : int | None = None) -> pd.DataFrame:
    '''Filter the position data by taking a rolling median followed by a rolling mean'''
    data = rawData.copy()

    if meanWin is None : meanWin = medWin
    for col in utils.flatten(metrics.POSDICT.values()):
        med = sp.ndimage.median_filter(data[col].to_numpy(),medWin)
        avg = sp.ndimage.uniform_filter(med,meanWin)
        data[col] = avg
    
    return data

def filter_median(rawData : pd.DataFrame, win : int = 5) -> pd.DataFrame:
    '''Filter the position data using a rolling median'''
    data = rawData.copy()

    for col in utils.flatten(metrics.POSDICT.values()):
        data[col] = sp.ndimage.median_filter(data[col].to_numpy(),win)
    
    return data 