import numpy as np
import pandas as pd
import os
import json
import utils


# ----- Static Parameters -----

ANALYZED_DATA = 'analysis.csv' # default name of analyzed data file
FILTERED_DATA = 'data_filtered.csv' # default name of filtered PiVR file
MANUAL_DATA = 'filtered_data.csv'
DISTANCE_TO_SOURCE = 'distance_to_source.csv' # file containing PiVR distance to source data
PIVR_SETTINGS = 'experiment_settings.json' # file containing PiVR experiment settings
PIVR_SETTINGS_UPDATED = 'experiment_settings_updated.json' # file containing PiVR experiment settings

SWAP_DATA = 'swaps.json'


# position data columns (must alternate x,y)
PIVRCOLS = [ # columns of interest in raw PiVR file
    'X-Head','Y-Head','X-Tail','Y-Tail','X-Midpoint','Y-Midpoint','X-Centroid','Y-Centroid',
    'Xmin-bbox','Ymin-bbox','Xmax-bbox','Ymax-bbox'
    ]
NEWCOLS = [ # new column names
    'xhead','yhead','xtail','ytail','xmid','ymid','xctr','yctr',
    'xmin','ymin','xmax','ymax'
    ]
POSCOLS = NEWCOLS[:8] # position column names


# ----- Data Import -----

def get_sample_directories(sourceDir : str) -> list[str]:
    '''
    Return a list of sub-directories that contain valid PiVR data
    NOTE: only checks for "...data.csv" file (not image data, etc)
    '''
    valid = []
    folders = utils.get_dirs(sourceDir)
    for folder in folders:
        try:
            fname = utils.find_file(folder,'data.csv')
            valid.append(folder)
        except:
            pass
    return valid


def load_raw_data(mainPath : str, fileName : str | None = None, px2mm : bool = True) -> pd.DataFrame:
    """
    Load raw data from csv and convert from PiVR format to analysis pipeline format

    fileName: name of file; if None, search for and load original PiVR output file
    px2mm: convert position data from px to mm and centers trajectory on source
    """
    # load raw data
    if fileName is None : rawData, _ = _retrieve_raw_data(mainPath) # find and load PiVR file
    else : rawData = utils.read_csv(os.path.join(mainPath,fileName))
    _fps, ppmm, source = get_settings(mainPath,mmSource=False)
    data = pd.DataFrame()

    # TODO: add compatability for multiple stim channels
    # load stimulation data (two different possible column names? Legacy PiVR version?)
    for key in ['stimulation','Stim Ch1']:
        if key in rawData.keys():
            data['stimulus'] = rawData[key]
            break

    # load position data
    # optionally move origin to odor source and convert from px to mm
    for i in range(len(PIVRCOLS)):
        idx = i % 2 # alternates between 0 and 1 -> x and y
        if px2mm : data[NEWCOLS[i]] = (rawData[PIVRCOLS[i]].to_numpy() - source[idx]) / ppmm
        else : data[NEWCOLS[i]] = rawData[PIVRCOLS[i]].to_numpy()

    return data


def _retrieve_raw_data(mainPath : str, dataTag = 'data.csv', exclude = ['filtered','omit']) -> pd.DataFrame:
    '''Locate raw data file and import to a pandas dataframe'''
    dataPath = utils.find_file(mainPath,dataTag,exclude)
    rawData = utils.read_csv(dataPath)
    return rawData, dataPath


def import_analysed_data(sourceDir : str, fileName : str = ANALYZED_DATA) -> pd.DataFrame:
    """
    Retrieve analysis results csv file for data
    """
    filePath = os.path.join(sourceDir,fileName)
    data = utils.read_csv(filePath)
    return data


def export_to_PiVR(sourceDir : pd.DataFrame, data : pd.DataFrame, 
                   suffix : str = 'level1', mm2px : bool = True) -> None:
    """
    Export data to PiVR-compatible csv file

    sourceDir: directory containing reference data
    data: data to export
    fileName: name of exported file
    mm2px: convert data from mm to px
    """
    # import raw data
    _fps, ppmm, source = get_settings(sourceDir,mmSource=False)
    rawData, dataPath = _retrieve_raw_data(sourceDir)

    # transform filtered position data and copy back into source dataframe 
    for i in range(len(PIVRCOLS)):
        idx = i % 2 # alternates between 0 and 1
        if mm2px : rawData[PIVRCOLS[i]] = data[NEWCOLS[i]].to_numpy() * ppmm + source[idx]
        else : rawData[PIVRCOLS[i]] = data[NEWCOLS[i]].to_numpy()

    rawDataFilename = os.path.basename(dataPath)
    name = rawDataFilename.split('.csv')[0]
    newFileName = f"{name}_{suffix}.csv"

    # export
    filePath = os.path.join(sourceDir,newFileName)
    rawData.to_csv(filePath, index=False) # export csv file


# ----- Settings / Supplemetal Data Import -----

def get_all_settings(mainPath : str, fileName : str = PIVR_SETTINGS) -> dict:
    '''
    get dictionary of assay settings
    TODO: store file name in json
    '''
    settingsPath = os.path.join(mainPath,fileName)
    try:
        with open(settingsPath) as json_file:
            settings = json.load(json_file)
        return settings
    except:
        return None


def get_settings(mainPath : str, mmSource : bool = False) -> tuple:
    '''
    retrieves fps, ppmm, and source position (mm) from experiment settings json
    mainPath: directory containing settings json
    mmSource: convert source from pixels to mm
    '''
    settings = get_all_settings(mainPath)
    settings_updated = get_all_settings(mainPath,PIVR_SETTINGS_UPDATED)

    fps = settings['Framerate']
    if settings_updated is not None:
        ppmm = settings_updated['Pixel per mm']
    else: ppmm = settings['Pixel per mm']

    # if no source entry (odor-less assay), set source to (0,0)
    if 'Source x' in settings.keys():
        source = np.array([settings['Source x'],settings['Source y']])
        if mmSource : source /= ppmm
    else:
        source = np.zeros(2)

    return (fps,ppmm,source)


def import_distance_data(data : pd.DataFrame, sourceDir: str, fileName = DISTANCE_TO_SOURCE) -> pd.Series:
    '''
    Retrieve distance from source calculation from csv generated by PiVR
    

    '''
    distancePath = sourceDir+fileName
    startFrame = data['Frame'][0] # data and distance records start on different frames
    distance = utils.read_csv(distancePath).iloc[startFrame:,1]
    distance = distance.reset_index(drop=True) # reset indices so distances assigned to correct frames
    print(type(distance))
    return distance


# ----- Multi-Sample Settings Import -----

def get_fps(sourceDir : str) -> int:
    '''Get the frame rate of given sample or of the first sample in a collection'''
    samples = utils.get_dirs(sourceDir)
    if len(samples) > 0: # collection
        return get_all_settings(samples[0])['Framerate']
    else: # single sample
        return get_all_settings(sourceDir)['Framerate']
    

def get_led_data(sourceDir : str, fileName : str = ANALYZED_DATA) -> np.ndarray:
    """
    Returns led stim sequence for experimental group
    TODO: store file name in json
    """
    samples = utils.get_dirs(sourceDir)
    data = import_analysed_data(samples[0],fileName)
    return data['stimulus'].to_numpy()

