import tkinter.filedialog as fd
import matplotlib.pyplot as plt
import numpy as np
import os
from swap_correction import plotting, utils, metrics
from swap_correction.tracking import tracking_correction
from swap_correction import pivr_loader as loader
import logging

FILE_NAME = loader.FILTERED_DATA # name of new file to export corrected data to
FILE_SUFFIX = 'level1'
# FILE_SUFFIX = 'test'

FIX_SWAPS = True # correct head-tail swaps using single-frame flags
VALIDATE = False # attempt to correct missed swaps using segment-based metrics (NOTE: currently not recommended!)
REMOVE_ERRORS = True # set position values in frames where head / tail overlap to NaN
INTERPOLATE = True # interpolate over short overlap segments
FILTER_DATA = False # filter data before exporting (not recommended)

DEBUG = False # print debug messages
DIAGNOSTIC_PLOTS = True # generate and save diagnostic figures
SHOW_PLOTS = False # display diagnostic figures after saving (if generated)
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
    _, dataPath = loader._retrieve_raw_data(mainPath)
    rawDataFilename = os.path.basename(dataPath)
    name = rawDataFilename.split('.csv')[0]
    newFileName = f"{name}_{FILE_SUFFIX}.csv"

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


# def compare_filtered_distributions(mainPath : str, outputPath : str = None,
#             fileName : str = 'compare_distributions.png', show : bool = True) -> None:
#     """
#     Compare distributions from raw and filtered position data
#     Note: analysis csv file must have been generated

#     mainPath: directory containing data for one sample
#     outputPath: directory to export image file to; if None, do not save the image
#     fileName: name of the image file
#     times: range of times to display data for; if none, display data for entire sample
#     show: display the figure after saving
#     """
#     # ----- Data Retrieval -----
#     suffix = 'level1'
#     _, dataPath = loader._retrieve_raw_data(mainPath)
#     rawDataFilename = os.path.basename(dataPath)
#     name = rawDataFilename.split('.csv')[0]
#     newFileName = f"{name}_{suffix}.csv"

#     rawData = loader.load_raw_data(mainPath)
#     processedData = loader.load_raw_data(mainPath,newFileName)
#     data = [rawData, processedData]


#     # ----- Figure ------
#     fig, axs = plt.subplots(3,2,squeeze=True,figsize=(8, 4)) # axes: left, bottom, width, height
#     fig.subplots_adjust(left=0.15, bottom=0.10, right=0.95, top=0.90, wspace=0.3, hspace=0.6)

#     titles = ['Head-Tail Separation (mm)','Body Orientation (rad)','Reorientation Rate (rad/s)']
#     spans = [(0,5),(0,np.pi),(0,1)]

#     # plot
#     for j, df in enumerate(data):
#         # Calculations
#         dist = metrics.get_delta_in_frame(df,'head','tail')
#         ba = metrics.get_orientation(df)
#         rrate = np.abs(np.diff(ba))
#         vals = [dist,ba,rrate]

#         # Figure
#         for i, ax in enumerate(axs[:,j]):
#             plotting.histogram(ax,vals[i],100,spans[i],True)
#             ax.set_title(titles[i])
#             if i == 2 : ax.set_ylim(0,100)

#     # finish
#     outPath = outputPath if outputPath else mainPath
#     plotting.save_figure(fig,fileName,outPath,show=show)


# def examine_flags(mainPath : str, outputPath : str = None, show : bool = True,
#             fileName : str = 'flags.png', times : tuple = None, labelFrames : bool = False) -> None:
#     """
#     Compare flagged frames and verified swap frames

#     mainPath: directory containing data for one sample
#     outputPath: directory to export image file to; if None, do not save the image
#     show: display the figure after saving
#     fileName: name of the image file
#     times: range of times to display data for; if none, display data for entire sample
#     labelFrames: label x-axis in units of frames instead of seconds
#     manual: csv with manual corrections
#     useManual: compare flags against mandual corrections
#     """

#     # ----- Data Retrieval -----
#     fps = loader.get_all_settings(mainPath)['Framerate']
#     rawData = loader.load_raw_data(mainPath)
#     processedData = loader.load_raw_data(mainPath,FILE_NAME)

#     rawData = tc.remove_edge_frames(rawData)


#     # ----- Flags -----
#     debug = False
#     #dh = tc.flag_discontinuities(rawData,'head',fps=fps,debug=debug)
#     dt = tc.flag_discontinuities(rawData,'tail',fps=fps,debug=debug)

#     olap = tc.flag_overlaps(rawData,debug=debug)
#     sr = tc.flag_sign_reversals(rawData,debug=debug)
#     dm = tc.flag_delta_mismatches(rawData,debug=debug)
#     #mdm = tc.flag_min_delta_mismatches(rawData,debug=debug)
    
#     cosr = tc.flag_overlap_sign_reversals(rawData,debug=debug)
#     #com = tc.flag_overlap_mismatches(rawData,debug=debug)
#     comm = tc.flag_overlap_minimum_mismatches(rawData,debug=debug)
#     print()

#     # filter out overlaps
#     filt = utils.merge(olap,olap+1)
#     dm = utils.filter_array(dm,filt)
#     sr = utils.filter_array(sr,filt)

#     # convert overlap frames to segments
#     overlaps = utils.get_consecutive_ranges(olap)


#     # ----- Curl Detection & Validation -----
#     # create intermediate corrected data
#     intData = tc.correct_tracking_errors(rawData)

#     # curl-based validation
#     debug = False
#     # finish
#     outPath = outputPath if outputPath else mainPath
#     plotting.save_figure(fig,fileName,outPath,show=show)


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
        logging.info('Sample %d/%d' % (i+1,nsamples),end=end)

        if True: # set to false to skip filtering and just plot
            data = loader.load_raw_data(sample)
            fps = loader.get_all_settings(sample)['Framerate']
            data = tracking_correction(
                data, fps,
                filterData=FILTER_DATA,
                swapCorrection=FIX_SWAPS,
                validate=VALIDATE,
                removeErrors=REMOVE_ERRORS,
                interp=INTERPOLATE,
                debug=DEBUG
            )
            loader.export_to_PiVR(sample, data, suffix=FILE_SUFFIX)

        # generate diagnostic plots
        if DIAGNOSTIC_PLOTS:
            compare_filtered_trajectories(sample,show=SHOW_PLOTS,times=TIMES)
            # compare_filtered_distributions(sample,show=SHOW_PLOTS)
            #examine_flags(sample,show=SHOW_PLOTS,times=TIMES,labelFrames=True)
            if SHOW_PLOTS : plt.show()

    logging.info('Finished ')
