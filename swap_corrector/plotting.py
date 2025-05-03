import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.colors import LinearSegmentedColormap
from sklearn.mixture import GaussianMixture
from sklearn.neighbors import KernelDensity
from scipy import stats, signal
from . import utils
import warnings


def save_figure(fig, name : str, outputDir : str | None = None, show : bool = False, block : bool = False) -> None:
    """
    Save and/or display the current figure

    name: name of file
    outputDir: directory to save file to; if None, do not save
    show: show plot after saving
    block: figure will block script from continuing until closed
    """
    if outputDir is not None:
        fileout = os.path.join(outputDir,name)
        plt.savefig(fileout)
    if show : plt.show(block=block)
    else : plt.close(fig)


def plot_trajectory(ax : plt.Axes, data : pd.DataFrame, fps : int = 1,
            times : tuple | None = None, source : tuple[float,float] = (0,0),
            ledData : np.ndarray | None = None, headTail : bool = True,
            limits : tuple[float,float] | None = (-60,60), legend : bool = True) -> None:
    """
    Plot a single trajectory in detail

    ax: pyplot axis
    data: dataframe containing position data
    fps: frame rate (only used if times == True)
    times: (optional) start and end times for drawing trajectory segment
    source: location of odor source
    ledData: (optional) if not None, used to color segments of trajectory based on LED activity
    headTail: indicate head and tail alongside centroid
    limits: axis limits
    legend: generate a figure legend
    """
    hcol = 'b' # head
    tcol = 'r' # tail
    ccol = 'grey' # centroid
    anncol = 'g' # annotation color (odor, start)

    if times is not None:
        # get start, end indices
        a, b = times
        a *= fps
        b *= fps

        # get position data
        xhead = data.loc[a:b-1,'xhead'].values
        yhead = data.loc[a:b-1,'yhead'].values
        xtail = data.loc[a:b-1,'xtail'].values
        ytail = data.loc[a:b-1,'ytail'].values
        xctr = data.loc[a:b-1,'xctr'].values
        yctr = data.loc[a:b-1,'yctr'].values

        # get first non-NaN position to indicate start
        xo = xctr[~np.isnan(xctr)][0]
        yo = yctr[~np.isnan(yctr)][0]

        # assign centroid color using LED data
        if ledData is not None : ccol = _get_colors_from_LED(ledData[a:b])

        if headTail:
            ax.plot(xhead,yhead,c=hcol,alpha=0.5,label='head')
            ax.plot(xtail,ytail,c=tcol,alpha=0.5,label='tail')
        ax.scatter(xctr,yctr,c=ccol,s=0.1,label='ctrd')
        ax.scatter(xo,yo,c=anncol,marker='x',s=10,label='start')

    else:
        # get first non-NaN position to indicate start
        xo = data['xctr'].loc[data['xctr'].first_valid_index()]
        yo = data['yctr'].loc[data['yctr'].first_valid_index()]

        if headTail:
            ax.plot(data['xhead'],data['yhead'],c=hcol,alpha=0.5,label='head')
            ax.plot(data['xtail'],data['ytail'],c=tcol,alpha=0.5,label='tail')
        if ledData is None:
            ax.plot(data['xctr'],data['yctr'],c=ccol,label='ctrd')
        else:
            ccol = _get_colors_from_LED(ledData)
            ax.scatter(data['xctr'],data['yctr'],c=ccol,s=0.1,label='ctrd')
        ax.scatter(xo,yo,c=anncol,marker='x',s=10,label='start')
        
    ax.scatter(*source,c=anncol,marker='o',s=10,label='odor')
    ax.set_title('Trajectory (mm)')
    if limits is not None:
        ax.set_xlim(*limits)
        ax.set_ylim(*limits)
    ax.axis('square')
    if legend : ax.legend(loc="upper right")


def plot_stacked_trajectories(ax : plt.Axes, dfs : list[pd.DataFrame],
            timeSpectrum : bool = True, source : tuple[float,float] = (0,0),
            ledData : np.ndarray | None = None) -> None:
    """
    Plot trajectories for all input samples on the same plot

    ax: figure axes
    dfs (list[pd.DataFrame]): a list of dataframes to retrieve trajectories from
    timeSpectrum (bool): if true, color-code time; otherwise, color-code samples
    source (list): the coordinates [x,y] of the odor source
    ledData (np.ndarray): array indicating LED activity
    """
    # indicate led activity
    if ledData is not None:
        for i, data in enumerate(dfs):
            cols = _get_colors_from_LED(data['stimulus']) # in case data different lengths ...
            ax.scatter(data['xctr'],data['yctr'],c=cols,s=0.1)
    # indicate time with color gradient
    elif timeSpectrum:
        nframes = dfs[0].shape[0]
        cmap = plt.get_cmap('viridis')
        cols = cmap(np.linspace(0,1,nframes)) # create a spectrum of colors across timepoints
        cols = np.flip(cols,0) # make color-coding more intuitive
        for i, data in enumerate(dfs):
            ax.scatter(data['xctr'],data['yctr'],c=cols,s=0.1)
    # indicate samples sequentially with color gradient
    else:
        cols = cmap(np.linspace(0,1,len(dfs))) # create a spectrum of colors across samples
        for i, data in enumerate(dfs):
            ax.plot(data['xctr'],data['yctr'],c=cols[i])
    
    ax.scatter(*source, c='b',marker='o',s=10)
    ax.set_title('Trajectories (mm)')
    ax.axis('square')


def _get_colors_from_LED(ledData : np.ndarray) -> list:
    '''Convert LED data to color array on black -> red'''
    colors = [(0, 0, 0), (1, 0, 0)] # first color is black, last is red
    cmap = LinearSegmentedColormap.from_list("Custom",colors,N=10)
    
    ledmax = np.max(ledData)
    if ledmax == 0 : ledmax = 1

    return cmap(ledData / ledmax)


def single_timeseries(ax : plt.Axes, time : np.ndarray, var : np.ndarray, ledData : np.ndarray,
                      xlim : tuple | None = None, ylim : tuple | None = None) -> None:
    '''
    Plot the timeseries against LED activity for a single vector
    
    ax: plotting axes
    time: time vector
    var: data vector
    ledData: LED activity vector
    xlim: x-axis limits
    ylim: y-axis limits
    '''
    ledCol = 'r'
    ledAlpha = 0.3
    sampleCol = 'b'

    ax.plot(time,var,c=sampleCol)
    ybounds = utils.get_bounds(var,floor=0)
    ax.fill_between(time,ledData * ybounds[1]/100,facecolor=ledCol,alpha=ledAlpha) # display LED activity

    if xlim is not None : ax.set_xlim(xlim)
    else : ax.set_xlim(time[0],time[-1])
    if ylim is not None : ax.set_ylim(ylim)
    else : ax.set_ylim(ybounds)
    ax.set_xlabel('Time (s)')


def multi_timeseries(ax : plt.Axes, time : np.ndarray, data : np.ndarray, ledData : np.ndarray,
                     axis : int = 0, indiv : bool = False, median : bool = False, avg : bool = True,
                     ylim : tuple | None = None, legend : bool = True) -> None:
    '''
    plot mean / median / sample value against LED activity
    
    ax: plotting axes
    time: time vector
    data: 2D array of data vectors of shape (nlarvae x nframes)
    ledData: vector indicating LED activity
    axis: axis to calculate mean, median, etc on
    indiv: plot individual trials (in grey)
    median: plot median value
    avg: plot average value
    ylim: y-axis limits
    legend: add legend
    '''
    # TODO: accomodate times not starting from zero
    nframes = data.shape[1]
    t = time[:nframes]
    led = ledData[:nframes]

    icol = 'grey'
    ialpha = 0.5
    ilw = 0.3
    avgcol = 'b'
    errcol = 'b'
    errAlpha = 0.3
    medcol = 'k'
    lw = 0.5
    ledCol = 'r'
    ledAlpha = 0.3
    labels = []
    elements = []

    # plot LED activity
    ax.fill_between(t,led * 100,facecolor=ledCol,alpha=ledAlpha) # TODO: implement better scaling

    if indiv: # plot individual trials
        for sample in data:
            ax.plot(t,sample,icol,lw=ilw,alpha=ialpha)

        labels.append('sample')
        elements.append(Line2D([0], [0], color=icol, alpha=ialpha))
    
    if avg: # plot average and standard deviation
        with warnings.catch_warnings(): # suppress "all-NaN slice encountered"
            warnings.simplefilter("ignore", category=RuntimeWarning)
            y = np.nanmean(data,axis)
            err = np.nanstd(data,axis)

        ax.fill_between(t, y-err, y+err ,alpha=errAlpha, facecolor=errcol)
        ax.plot(t,y,avgcol,lw=lw)

        labels.append('average')
        elements.append(Line2D([0], [0], color=avgcol))

    if median: # plot median
        with warnings.catch_warnings(): # suppress "all-NaN slice encountered"
            warnings.simplefilter("ignore", category=RuntimeWarning)
            med = np.nanmedian(data,axis)

        ax.plot(t,med,medcol,lw=lw)

        labels.append('median')
        elements.append(Line2D([0], [0], color=medcol))

    ax.set_xlim(time[0],time[-1])
    if ylim : ax.set_ylim(ylim)
    ax.set_xlabel('Time(s)')
    if legend : ax.legend(elements, labels, loc="upper left")


# ----- Probability Density Extimation -----

def get_kernel_density_estimate(data : np.ndarray, span : tuple[float,float], resolution : int = 1000,
                   kernel : str = 'gaussian', bandwidth : float | str = 1.0) -> tuple[np.ndarray,np.ndarray]:
    '''
    Get the kernel density estimate of the input data
    Returns x-values and density estimates

    data: 1D array of samples
    span: expected range / range of interest for the data
    resolution: number of points to use in generating the density function
    kernel: kernel shape (see sklearn.neighbors.KernelDensity)
    bandwidth: kernel bandwidth or estimation method (see sklearn.neighbors.KernelDensity)
    '''
    data2d = data[~np.isnan(data),np.newaxis] # convert to 2d array (n samples, 1 feature) and filter NaN's
    xvals = np.linspace(*span,resolution)[:,np.newaxis]
    kde = KernelDensity(kernel=kernel, bandwidth=bandwidth).fit(data2d)
    logdens = kde.score_samples(xvals)
    dens = np.exp(logdens)
    return xvals, dens


def kernel_density_estimate(ax: plt.axes, data : np.ndarray | list[np.ndarray], span : tuple[float,float],
                   ylim : tuple[float,float] | None = None, labels : list[str] | None = None,
                   **kwargs) -> None:
    '''
    Plot the kernel density estimate of the input data

    ax: plotting axis
    data: one or a list of multiple 1D arrays of values
    span: expected range / range of interest for the data
    ylim: y-limits of axis
    labels: labels for different vectors
    kwargs: additional optional arguments for plotting.get_kernel_density()
    '''
    # convert single argument to list
    if type(data) is not list : data = [data]

    # calculate and plot kernel density for each data vector
    for i, vec in enumerate(data):
        xvals, kde = get_kernel_density_estimate(vec,span,**kwargs)
        label = '' if labels is None else labels[i]
        ax.plot(xvals,kde,linestyle="-",label=label)

    # finish
    ax.set_xlim(*span)
    if ylim is not None : ax.set_ylim(ylim)
    if labels is not None : ax.legend()


def histogram(ax : plt.Axes, data : np.ndarray, span : tuple, nbins : int = 100,
            scaleBars : bool = False, width : float = 1.0) -> None:
    '''
    Plot a histogram of the input data

    ax (plt.Axes): plotting axis
    data (np.ndarray): input data
    span (tuple): x limits of histogram
    nbins (int): number of bins to use
    scaleBars (bool): multiply bin counts by bin value to normalize
    width (float): bar width
    '''
    counts, edges = np.histogram(data,bins=nbins,range=span)
    w = np.diff(edges)[0] * width

    if scaleBars : ax.bar(edges[:nbins],counts*edges[1:],w,align='edge')
    else : ax.bar(edges[:nbins],counts,w,align='edge')

    ax.set_xlim(span)


def _mix_pdf(x, loc, scale, weights):
    d = np.zeros_like(x)
    for mu, sigma, pi in zip(loc, scale, weights):
        d += pi * stats.norm.pdf(x, loc=mu, scale=sigma)
    return d

def gmm(ax : plt.Axes, data : np.ndarray, npeaks : int, nbins : int, xlim : tuple) -> None:
    '''
    Fit a GMM against the distribution of values in the input array and plot the result

    a: axes to plot on
    data: input vector
    npeaks: number of peaks to fit to
    nbins: number of bins to use in histogram
    xlim: range of values to consider
    '''
    vals = data[data <= xlim[1]]
    vals = vals[vals >= xlim[0]]
    vals = vals[vals != np.NaN]
    vals = vals.reshape((vals.size,1))

    # fit GMM
    mix = GaussianMixture(n_components=npeaks, random_state=1, max_iter=100).fit(vals)
    pi = mix.weights_.flatten()
    mu = mix.means_.flatten()
    sigma = np.sqrt(mix.covariances_.flatten())

    # get pdf
    xpdf = np.linspace(xlim[0], xlim[1], 1000)
    pdf = _mix_pdf(xpdf, mu, sigma, pi)

    # get minima
    pdfmin = signal.argrelextrema(pdf, np.less)
    print('GMM local minima:',pdfmin[0])

    # plot
    ax.hist(vals, bins=nbins, density=True, alpha=0.2)
    ax.plot(xpdf, pdf)
    #ax.axvline(pdfmin,c='grey')
    ax.set_xlim(xlim)


def get_fft(data : np.ndarray, fps : int = 30) -> tuple[np.ndarray,np.ndarray]:
    '''
    Get the one-sided FFT of the input vector
    Returns frequencies and amplitudes

    ax: axes to plot on
    data: vector to analyse
    fps: sample rate (frame rate)
    '''
    N = data.shape[0] # number of elements
    t = np.linspace(0, N / fps, N)
    T = t[1] - t[0]
    #f = np.linspace(0, 1 / T, N)

    # replace NaNs and INFs with zeros
    filtData = np.nan_to_num(data,True,0,0,0)

    # perform FFT
    fft = np.fft.fft(filtData)
    fftfreq = np.fft.fftfreq(N,T)

    # get the one-sided specturm
    n_oneside = N//2
    amp = np.abs(fft[:n_oneside])
    freqs = fftfreq[:n_oneside]

    return freqs, amp

def fft(ax : plt.Axes, data : np.ndarray, fps : int = 30,
        xlim : tuple | None = None, ylim : tuple | None = None) -> None:
    '''
    Plot the FFT of the input vector

    ax: axes to plot on
    data: vector to analyse
    fps: sample rate (frame rate)
    xlim: x limits
    ylim: y limits
    '''
    # get FFT
    freqs, amp = get_fft(data,fps)

    # generate figure
    ax.plot(freqs, amp)
    if xlim is not None : ax.set_xlim(xlim)
    if ylim is not None : ax.set_ylim(ylim)
    else:
        ymax = np.max(amp[10:]) * 1.1
        ax.set_ylim((0,ymax))
    ax.set_ylabel("Amplitude")
    ax.set_xlabel("Frequency (Hz)")


def get_power_spectrum(data : np.ndarray, fps : int = 30) -> tuple[np.ndarray,np.ndarray,np.ndarray]:
    '''
    Extract the mean and std dev of the power spectrum of the input vectors
    returns: frequency, average spectrum, std dv spectrum
    TODO: add Welch vs default option

    data: 1D or 2D (nsamples,nframes) vector to analyse
    fps: sample rate (frame rate)
    '''
    filtData = np.nan_to_num(data,True,0,0,0)
    f, pwelch = signal.welch(filtData, fps)
    avg = np.mean(pwelch,axis=0)
    err = np.std(pwelch,axis=0)
    return f, avg, err


def power_spectrum(ax : plt.Axes, data : np.ndarray, fps : int = 30,
                   xlim : tuple | None = None, ylim : tuple | None = None) -> None:
    '''
    Plot the Welch power spectrum of the inpt vectors

    ax: axes to plot on
    data: 1D or 2D (nsamples,nframes) vector to analyse
    fps: sample rate (frame rate)
    xlim: x limits
    ylim: y limits
    '''
    # get power spectrum
    f, avg, err = get_power_spectrum(data,fps)

    # generate figure
    ax.fill_between(f,avg-err,avg+err,facecolor='grey',alpha=0.2)
    ax.plot(f, avg)
    #ax.semilogy(f, avg)
    ax.set_xlabel('Frequency (Hz)')
    ax.set_ylabel('PSD')
    if xlim is not None : ax.set_xlim(xlim)
    if ylim is not None : ax.set_ylim(ylim)


def report_data(ax, data : np.ndarray) -> None:
    '''
    Writes counts and durations of some activity on a plot

    ax: axes to plot on
    data: vector of activity durations
    '''
    msg1 = 'Count: %d' % len(data)
    msg2 = 'Avg Length: %.2f (%.2f) s' % (np.nanmean(data),np.nanstd(data))
    msg3 = 'Total Duration: %.2f s' % np.sum(data)
    msg = msg1+'\n'+msg2+'\n'+msg3

    ax.text(0.40, 0.80, msg,
        verticalalignment='center', horizontalalignment='left',
        transform=ax.transAxes, fontsize=7)

def generate_trajectory_plot(data: pd.DataFrame, fps: int, 
                           times: tuple = None, source: tuple = (0, 0),
                           led_data: np.ndarray = None) -> plt.Figure:
    """
    Generate a plot of the trajectory.
    
    Args:
        data: DataFrame containing position data
        fps: Frame rate
        times: Optional tuple of (start_time, end_time)
        source: Tuple of (x, y) coordinates for the source
        led_data: Optional array of LED activity data
        
    Returns:
        matplotlib Figure object
    """
    fig, ax = plt.subplots(figsize=(8, 8))
    plot_trajectory(ax, data, fps, times, source, led_data)
    return fig

def generate_distribution_plot(data: pd.DataFrame, fps: int) -> plt.Figure:
    """
    Generate a plot of the position distributions.
    
    Args:
        data: DataFrame containing position data
        fps: Frame rate
        
    Returns:
        matplotlib Figure object
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))
    
    # Plot x-position distribution
    kernel_density_estimate(ax1, data['xctr'].values, 
                          span=(data['xctr'].min(), data['xctr'].max()))
    ax1.set_title('X-position Distribution')
    ax1.set_xlabel('X Position (mm)')
    
    # Plot y-position distribution
    kernel_density_estimate(ax2, data['yctr'].values,
                          span=(data['yctr'].min(), data['yctr'].max()))
    ax2.set_title('Y-position Distribution')
    ax2.set_xlabel('Y Position (mm)')
    
    return fig

def generate_flag_plot(data: pd.DataFrame, flags: dict) -> plt.Figure:
    """
    Generate a plot visualizing the flags.
    
    Args:
        data: DataFrame containing position data
        flags: Dictionary containing flag data
        
    Returns:
        matplotlib Figure object
    """
    fig, ax = plt.subplots(figsize=(12, 6))
    
    # Plot time series of flags
    for i, (flag_name, flag_data) in enumerate(flags.items()):
        if flag_data is not None:
            ax.plot(np.arange(len(flag_data)), flag_data + i, label=flag_name)
    
    ax.set_title('Flag Time Series')
    ax.set_xlabel('Frame')
    ax.set_ylabel('Flag Type')
    ax.legend()
    
    return fig

def generate_comparison_plot(raw_data: pd.DataFrame, processed_data: pd.DataFrame,
                           fps: int) -> plt.Figure:
    """
    Generate a plot comparing raw and processed data.
    
    Args:
        raw_data: DataFrame containing raw position data
        processed_data: DataFrame containing processed position data
        fps: Frame rate
        
    Returns:
        matplotlib Figure object
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))
    
    # Plot raw trajectory
    plot_trajectory(ax1, raw_data, fps)
    ax1.set_title('Raw Trajectory')
    
    # Plot processed trajectory
    plot_trajectory(ax2, processed_data, fps)
    ax2.set_title('Processed Trajectory')
    
    return fig