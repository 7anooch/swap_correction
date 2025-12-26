import tkinter.filedialog as fd
import matplotlib.pyplot as plt
import numpy as np
import os
from swap_correction import plotting, utils, metrics
from swap_correction import tracking_correction as tc
from swap_correction import pivr_loader as loader


FILE_NAME = loader.FILTERED_DATA # name of new file to export corrected data to

FIX_SWAPS = True # correct head-tail swaps using single-frame flags
VALIDATE = False # attempt to correct missed swaps using segment-based metrics (NOTE: currently not recommended!)
REMOVE_ERRORS = True # set position values in frames where head / tail overlap to NaN
INTERPOLATE = True # interpolate over short overlap segments
FILTER_DATA = False # filter data before exporting (not recommended)

DEBUG = False # print debug messages
DIAGNOSTIC_PLOTS = True # generate and save diagnostic figures
SHOW_PLOTS = True # display diagnostic figures after saving (if generated)
TIMES = None#(200,230) # start and end times to show on plots (None -> show entire trajectory)


def compare_filtered_trajectories(mainPath : str, outputPath : str = None,
            fileName : str = 'compare_trajectories.png', times : tuple = None, show : bool = True) -> None:
    '''
    Compare trajectories from raw and filtered position data
    Note: analysis csv file must hve been generated

    mainPath: directory containing data for one sample
    outputPath: directory to export image file to; if None, do not save the image
    fileName: name of the image file
    times: range of times to display data for; if none, display data for entire sample
    show: display the figure after saving
    '''
    # ----- Data Retrieval -----
    suffix = 'level1'
    _, dataPath = loader._retrieve_raw_data(mainPath)
    rawDataFilename = os.path.basename(dataPath)
    name = rawDataFilename.split('.csv')[0]
    newFileName = f"{name}_{suffix}.csv"

    fps = loader.get_all_settings(mainPath)['Framerate']
    rawData = loader.load_raw_data(mainPath)
    processedData = loader.load_raw_data(mainPath,newFileName)

    # ----- Figure ------
    fig, axs = plt.subplots(1,2,squeeze=True,figsize=(8, 4)) # axes: left, bottom, width, height

    titles = ['Raw','Processed']#,'Filtered']
    data = [rawData,processedData]

    # plot
    for i, ax in enumerate(axs):
        plotting.plot_trajectory(ax,data[i],fps,times)
        xlim = metrics.get_df_bounds([processedData],['xhead','xtail'])
        ylim = metrics.get_df_bounds([processedData],['yhead','ytail'])
        ax.set_xlim(xlim)
        ax.set_ylim(ylim)
        ax.axis('square')
        ax.set_title(titles[i])

    # finish
    outPath = outputPath if outputPath else mainPath
    plotting.save_figure(fig,fileName,outPath,show=show)


if __name__ == '__main__':
    # open dialogue to get target directory
    msg = 'Select a PiVR trial folder or a parent folder containing multiple trials.'
    sourceDir = fd.askdirectory(title=msg)

    # retrieve sample directories / detect if target directory is single sample
    samples = utils.get_dirs(sourceDir) # get list of directories in target directory
    if len(samples) == 0: # single trial should not contain sub-folders
        samples = [sourceDir]
    nsamples = len(samples)
    
    # filter data
    end = '\n' if DEBUG else '\r' # ensure debug messages don't overwrite sample count
    for i, sample in enumerate(samples):
        print('Sample %d/%d' % (i+1,nsamples),end=end)

        if True: # set to false to skip filtering and just plot
            data = loader.load_raw_data(sample)
            fps = loader.get_all_settings(sample)['Framerate']
            data = tc.tracking_correction(
                data, fps,
                filterData=FILTER_DATA,
                swapCorrection=FIX_SWAPS,
                validate=VALIDATE,
                removeErrors=REMOVE_ERRORS,
                interp=INTERPOLATE,
                debug=DEBUG
                )
            loader.export_to_PiVR(sample,data) #,FILE_NAME)

        if DIAGNOSTIC_PLOTS:
            compare_filtered_trajectories(sample,show=SHOW_PLOTS,times=TIMES)
            if SHOW_PLOTS : plt.show()

    print('Finished ')
