import tkinter.filedialog as fd
import matplotlib.pyplot as plt
import numpy as np
import os
import plotting
import utils
import metrics
import tracking_correction as tc
import pivr_loader as loader


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
    data = [rawData,processedData]#,filteredData]

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


def compare_filtered_distributions(mainPath : str, outputPath : str = None,
            fileName : str = 'compare_distributions.png', show : bool = True) -> None:
    """
    Compare distributions from raw and filtered position data
    Note: analysis csv file must have been generated

    mainPath: directory containing data for one sample
    outputPath: directory to export image file to; if None, do not save the image
    fileName: name of the image file
    times: range of times to display data for; if none, display data for entire sample
    show: display the figure after saving
    """
    # ----- Data Retrieval -----
    suffix = 'level1'
    _, dataPath = loader._retrieve_raw_data(mainPath)
    rawDataFilename = os.path.basename(dataPath)
    name = rawDataFilename.split('.csv')[0]
    newFileName = f"{name}_{suffix}.csv"

    rawData = loader.load_raw_data(mainPath)
    processedData = loader.load_raw_data(mainPath,newFileName)
    data = [rawData, processedData]


    # ----- Figure ------
    fig, axs = plt.subplots(3,2,squeeze=True,figsize=(8, 4)) # axes: left, bottom, width, height
    fig.subplots_adjust(left=0.15, bottom=0.10, right=0.95, top=0.90, wspace=0.3, hspace=0.6)

    titles = ['Head-Tail Separation (mm)','Body Orientation (rad)','Reorientation Rate (rad/s)']
    spans = [(0,5),(0,np.pi),(0,1)]

    # plot
    for j, df in enumerate(data):
        # Calculations
        dist = metrics.get_delta_in_frame(df,'head','tail')
        ba = metrics.get_orientation(df)
        rrate = np.abs(np.diff(ba))
        vals = [dist,ba,rrate]

        # Figure
        for i, ax in enumerate(axs[:,j]):
            plotting.histogram(ax,vals[i],100,spans[i],True)
            ax.set_title(titles[i])
            if i == 2 : ax.set_ylim(0,100)

    # finish
    outPath = outputPath if outputPath else mainPath
    plotting.save_figure(fig,fileName,outPath,show=show)


def examine_flags(mainPath : str, outputPath : str = None, show : bool = True,
            fileName : str = 'flags.png', times : tuple = None, labelFrames : bool = False) -> None:
    """
    Compare flagged frames and verified swap frames

    mainPath: directory containing data for one sample
    outputPath: directory to export image file to; if None, do not save the image
    show: display the figure after saving
    fileName: name of the image file
    times: range of times to display data for; if none, display data for entire sample
    labelFrames: label x-axis in units of frames instead of seconds
    manual: csv with manual corrections
    useManual: compare flags against mandual corrections
    """

    # ----- Data Retrieval -----
    fps = loader.get_all_settings(mainPath)['Framerate']
    rawData = loader.load_raw_data(mainPath)
    processedData = loader.load_raw_data(mainPath,FILE_NAME)

    rawData = tc.remove_edge_frames(rawData)


    # ----- Flags -----
    debug = False
    #dh = tc.flag_discontinuities(rawData,'head',fps=fps,debug=debug)
    dt = tc.flag_discontinuities(rawData,'tail',fps=fps,debug=debug)

    olap = tc.flag_overlaps(rawData,debug=debug)
    sr = tc.flag_sign_reversals(rawData,debug=debug)
    dm = tc.flag_delta_mismatches(rawData,debug=debug)
    #mdm = tc.flag_min_delta_mismatches(rawData,debug=debug)
    
    cosr = tc.flag_overlap_sign_reversals(rawData,debug=debug)
    #com = tc.flag_overlap_mismatches(rawData,debug=debug)
    comm = tc.flag_overlap_minimum_mismatches(rawData,debug=debug)
    print()

    # filter out overlaps
    filt = utils.merge(olap,olap+1)
    dm = utils.filter_array(dm,filt)
    sr = utils.filter_array(sr,filt)

    # convert overlap frames to segments
    overlaps = utils.get_consecutive_ranges(olap)


    # ----- Curl Detection & Validation -----
    # create intermediate corrected data
    intData = tc.correct_tracking_errors(rawData)

    # curl-based validation
    debug = False
    curls = tc.get_curl_segments(intData,overlaps=False,debug=debug)
    remSwaps = tc.flag_swaps_after_curl(intData,debug=debug)


    # ----- Figure ------
    # TODO: display overlaps and curls as shaded segments instead of lines every frame
    fig, axs = plt.subplots(2,1,squeeze=True,figsize=(12, 6)) # axes: left, bottom, width, height
    fig.subplots_adjust(left=0.15, bottom=0.10, right=0.95, top=0.95, wspace=0.3, hspace=0.6)

    errframes = [[],[],dm,cosr,comm,sr]
    segments = [curls,remSwaps,[],[],overlaps,[]]
    cols = ['black','crimson','fuchsia','cyan','blue','lime']

    time = np.arange(0,len(rawData['xhead'])) if labelFrames else utils.get_time_axis(len(rawData['xhead']),fps)
    dt = time[1]-time[0]
    delta = dt/20
    titles = ['Flags','Overlaps and Curls']
    axes = ['Raw BA','Corrected BA']
    data = [rawData,processedData]

    # plot
    for j, ax in enumerate(axs):
        # line plot for reference
        ba = metrics.get_orientation(data[j])
        ax.plot(time,ba,c='grey')

        # use vertical lines to indicate where errors detected
        if j == 0:
            for k, flag in enumerate(errframes):
                for fr in flag:
                    ax.axvline(time[fr]+delta*k,c=cols[k],alpha=1.0,lw=1.5)

        # use shaded regions to indicate overlaps, curls, etc
        else:
            for k, segs in enumerate(segments):
                for a, b in segs:
                    c = min(b+1,rawData.shape[0]-1)
                    ax.fill_between(time[a:c],np.pi,0,color=cols[k],alpha=0.2)
                    #ax.axvline(time[a]+delta*k,c=cols[k],alpha=1.0,lw=1.5)
                    #ax.axvline(time[c]+delta*k,c=cols[k],alpha=1.0,lw=1.5)

        if times:
            if labelFrames : ax.set_xlim(times[0]*fps,times[1]*fps)
            else : ax.set_xlim(times)
        ax.set_ylim([0,np.pi])
        ax.set_xlabel('Frame' if labelFrames else 'Time (s)')
        ax.set_ylabel(axes[j])
        ax.set_title(titles[j])

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
            # compare_filtered_distributions(sample,show=SHOW_PLOTS)
            #examine_flags(sample,show=SHOW_PLOTS,times=TIMES,labelFrames=True)
            if SHOW_PLOTS : plt.show()

    print('Finished ')
