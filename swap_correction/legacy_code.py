"""
Legacy Code Archive

This file contains deprecated or unused code that has been removed from the main codebase
but is preserved here for reference. This code is NOT executed and should NOT be imported.

Source: swap_correct.py (lines 68-174, 208-209)
Date Removed: 2024
Reason: Code cleanup - these functions were commented out and are no longer used.
        They are preserved here in case they need to be referenced for future development.

Functions:
- compare_filtered_distributions: Compare distributions from raw and filtered position data
- examine_flags: Compare flagged frames and verified swap frames (incomplete implementation)
"""

# ============================================================================
# compare_filtered_distributions
# ============================================================================
# Original location: swap_correct.py, lines 68-115
# Status: Commented out, not used in main pipeline

# def compare_filtered_distributions(mainPath : str, outputPath : str = None,
#             fileName : str = 'compare_distributions.png', show : bool = True) -> None:
#     """
#     Compare distributions from raw and filtered position data
#     Note: analysis csv file must have been generated
#
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
#
#     rawData = loader.load_raw_data(mainPath)
#     processedData = loader.load_raw_data(mainPath,newFileName)
#     data = [rawData, processedData]
#
#
#     # ----- Figure ------
#     fig, axs = plt.subplots(3,2,squeeze=True,figsize=(8, 4)) # axes: left, bottom, width, height
#     fig.subplots_adjust(left=0.15, bottom=0.10, right=0.95, top=0.90, wspace=0.3, hspace=0.6)
#
#     titles = ['Head-Tail Separation (mm)','Body Orientation (rad)','Reorientation Rate (rad/s)']
#     spans = [(0,5),(0,np.pi),(0,1)]
#
#     # plot
#     for j, df in enumerate(data):
#         # Calculations
#         dist = metrics.get_delta_in_frame(df,'head','tail')
#         ba = metrics.get_orientation(df)
#         rrate = np.abs(np.diff(ba))
#         vals = [dist,ba,rrate]
#
#         # Figure
#         for i, ax in enumerate(axs[:,j]):
#             plotting.histogram(ax,vals[i],100,spans[i],True)
#             ax.set_title(titles[i])
#             if i == 2 : ax.set_ylim(0,100)
#
#     # finish
#     outPath = outputPath if outputPath else mainPath
#     plotting.save_figure(fig,fileName,outPath,show=show)


# ============================================================================
# examine_flags
# ============================================================================
# Original location: swap_correct.py, lines 118-173
# Status: Commented out, incomplete implementation

# def examine_flags(mainPath : str, outputPath : str = None, show : bool = True,
#             fileName : str = 'flags.png', times : tuple = None, labelFrames : bool = False) -> None:
#     """
#     Compare flagged frames and verified swap frames
#
#     mainPath: directory containing data for one sample
#     outputPath: directory to export image file to; if None, do not save the image
#     show: display the figure after saving
#     fileName: name of the image file
#     times: range of times to display data for; if none, display data for entire sample
#     labelFrames: label x-axis in units of frames instead of seconds
#     manual: csv with manual corrections
#     useManual: compare flags against mandual corrections
#     """
#
#     # ----- Data Retrieval -----
#     fps = loader.get_all_settings(mainPath)['Framerate']
#     rawData = loader.load_raw_data(mainPath)
#     processedData = loader.load_raw_data(mainPath,FILE_NAME)
#
#     rawData = tc.remove_edge_frames(rawData)
#
#
#     # ----- Flags -----
#     debug = False
#     #dh = tc.flag_discontinuities(rawData,'head',fps=fps,debug=debug)
#     dt = tc.flag_discontinuities(rawData,'tail',fps=fps,debug=debug)
#
#     olap = tc.flag_overlaps(rawData,debug=debug)
#     sr = tc.flag_sign_reversals(rawData,debug=debug)
#     dm = tc.flag_delta_mismatches(rawData,debug=debug)
#     #mdm = tc.flag_min_delta_mismatches(rawData,debug=debug)
#     
#     cosr = tc.flag_overlap_sign_reversals(rawData,debug=debug)
#     #com = tc.flag_overlap_mismatches(rawData,debug=debug)
#     comm = tc.flag_overlap_minimum_mismatches(rawData,debug=debug)
#     print()
#
#     # filter out overlaps
#     filt = utils.merge(olap,olap+1)
#     dm = utils.filter_array(dm,filt)
#     sr = utils.filter_array(sr,filt)
#
#     # convert overlap frames to segments
#     overlaps = utils.get_consecutive_ranges(olap)
#
#
#     # ----- Curl Detection & Validation -----
#     # create intermediate corrected data
#     intData = tc.correct_tracking_errors(rawData)
#
#     # curl-based validation
#     debug = False
#     # finish
#     outPath = outputPath if outputPath else mainPath
#     plotting.save_figure(fig,fileName,outPath,show=show)


# ============================================================================
# Additional commented code from main block
# ============================================================================
# Original location: swap_correct.py, lines 208-209
# Status: Commented out function calls in main execution block

# compare_filtered_distributions(sample,show=SHOW_PLOTS)
# examine_flags(sample,show=SHOW_PLOTS,times=TIMES,labelFrames=True)

