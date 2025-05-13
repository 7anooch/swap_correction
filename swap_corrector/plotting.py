"""
Plotting utilities for swap correction pipeline.
"""

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
import scipy.stats as sp
from . import metrics
from scipy.stats import gaussian_kde
import scipy.ndimage as sp
from typing import Optional, Tuple, Dict, Any, Union

# Column name mappings
COLUMN_MAPPINGS = {
    'xhead': 'X-Head',
    'yhead': 'Y-Head',
    'xtail': 'X-Tail',
    'ytail': 'Y-Tail',
    'xctr': 'X-Center',
    'yctr': 'Y-Center',
    'speed': 'Speed',
    'acceleration': 'Acceleration',
    'curvature': 'Curvature'
}

def normalize_column_names(df: pd.DataFrame) -> pd.DataFrame:
    """Normalize column names to standard format.
    
    Args:
        df: Input DataFrame
        
    Returns:
        DataFrame with normalized column names
    """
    df = df.copy()
    for old_col, new_col in COLUMN_MAPPINGS.items():
        if old_col in df.columns:
            df[new_col] = df[old_col]
            df = df.drop(columns=[old_col])
    return df

def save_figure(fig, name: str, output_dir: str | None = None, show: bool = False, block: bool = False) -> None:
    """Save and/or display the current figure.
    
    Args:
        name: Name of file
        output_dir: Directory to save file to; if None, do not save
        show: Show plot after saving
        block: Figure will block script from continuing until closed
    """
    if output_dir is not None:
        os.makedirs(output_dir, exist_ok=True)
        fileout = os.path.join(output_dir, name)
        plt.savefig(fileout)
    if show:
        plt.show(block=block)
    else:
        plt.close(fig)

def plot_trajectory(ax: plt.Axes, data: pd.DataFrame, fps: float) -> None:
    """Plot trajectory with optional flags.
    
    Args:
        ax: Axes to plot on
        data: DataFrame with trajectory data
        fps: Frames per second
    """
    # Plot head trajectory
    ax.plot(data['X-Head'], data['Y-Head'], 'b-', alpha=0.5, label='Head')
    
    # Plot tail trajectory
    ax.plot(data['X-Tail'], data['Y-Tail'], 'r-', alpha=0.5, label='Tail')
    
    # Plot midpoints if available
    if 'X-Midpoint' in data.columns and 'Y-Midpoint' in data.columns:
        ax.plot(data['X-Midpoint'], data['Y-Midpoint'], 'g-', alpha=0.5, label='Midpoint')
    
    # Add labels
    ax.set_xlabel('X Position (mm)')
    ax.set_ylabel('Y Position (mm)')
    ax.legend()
    ax.set_aspect('equal')

def plot_stacked_trajectories(ax: plt.Axes, dfs: list[pd.DataFrame],
                            time_spectrum: bool = True, source: tuple[float,float] = (0,0),
                            led_data: np.ndarray | None = None) -> None:
    """Plot trajectories for all input samples on the same plot.
    
    Args:
        ax: Figure axes
        dfs: List of DataFrames to retrieve trajectories from
        time_spectrum: If true, color-code time; otherwise, color-code samples
        source: The coordinates [x,y] of the odor source
        led_data: Array indicating LED activity
    """
    # Normalize column names for all DataFrames
    dfs = [normalize_column_names(df) for df in dfs]
    
    # Indicate LED activity
    if led_data is not None:
        for i, data in enumerate(dfs):
            cols = _get_colors_from_LED(data['stimulus'])
            ax.scatter(data['X-Center'], data['Y-Center'], c=cols, s=0.1)
    # Indicate time with color gradient
    elif time_spectrum:
        nframes = dfs[0].shape[0]
        cmap = plt.get_cmap('viridis')
        cols = cmap(np.linspace(0, 1, nframes))
        cols = np.flip(cols, 0)
        for i, data in enumerate(dfs):
            ax.scatter(data['X-Center'], data['Y-Center'], c=cols, s=0.1)
    # Indicate samples sequentially with color gradient
    else:
        cmap = plt.get_cmap('viridis')
        cols = cmap(np.linspace(0, 1, len(dfs)))
        for i, data in enumerate(dfs):
            ax.plot(data['X-Center'], data['Y-Center'], c=cols[i])
    
    ax.scatter(*source, c='b', marker='o', s=10)
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


def kernel_density_estimate(ax, data: np.ndarray, xlabel: str, label: str):
    """Plot kernel density estimate of data.
    
    Args:
        ax: Matplotlib axis
        data: Data to plot
        xlabel: X-axis label
        label: Legend label
    """
    # Remove NaN values
    data = data[~np.isnan(data)]
    
    # Calculate kernel density estimate
    kde = gaussian_kde(data)
    x_range = np.linspace(np.min(data), np.max(data), 100)
    density = kde(x_range)
    
    # Plot
    ax.plot(x_range, density, label=label)
    ax.set_xlabel(xlabel)
    ax.set_ylabel('Density')
    ax.legend()
    ax.grid(True)


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

def generate_trajectory_plot(data: pd.DataFrame, fps: float = 30.0) -> plt.Figure:
    """Generate a plot of the trajectory.
    
    Args:
        data: DataFrame with trajectory data
        fps: Frames per second
        
    Returns:
        Figure object
    """
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Plot head trajectory
    ax.plot(data['X-Head'], data['Y-Head'], 'b-', label='Head', alpha=0.7)
    ax.plot(data['X-Head'].iloc[0], data['Y-Head'].iloc[0], 'bo', label='Head Start')
    ax.plot(data['X-Head'].iloc[-1], data['Y-Head'].iloc[-1], 'b*', label='Head End')
    
    # Plot tail trajectory
    ax.plot(data['X-Tail'], data['Y-Tail'], 'r-', label='Tail', alpha=0.7)
    ax.plot(data['X-Tail'].iloc[0], data['Y-Tail'].iloc[0], 'ro', label='Tail Start')
    ax.plot(data['X-Tail'].iloc[-1], data['Y-Tail'].iloc[-1], 'r*', label='Tail End')
    
    # Plot midpoint trajectory if available
    if 'X-Mid' in data.columns and 'Y-Mid' in data.columns:
        ax.plot(data['X-Mid'], data['Y-Mid'], 'g--', label='Midpoint', alpha=0.5)
    
    ax.set_xlabel('X Position')
    ax.set_ylabel('Y Position')
    ax.set_title('Trajectory Plot')
    ax.legend()
    ax.grid(True)
    
    return fig

def generate_distribution_plot(data: Union[pd.DataFrame, np.ndarray], title: str = '', xlabel: str = '', ylabel: str = '', fps: float = 30.0) -> plt.Figure:
    """Generate a plot of trajectory distributions.
    
    Args:
        data: DataFrame with trajectory data or numpy array of values
        title: Plot title
        xlabel: X-axis label
        ylabel: Y-axis label
        fps: Frames per second
        
    Returns:
        Figure object
    """
    fig, ax = plt.subplots(figsize=(10, 6))
    
    if isinstance(data, pd.DataFrame):
        # Calculate speed and acceleration
        speeds = np.sqrt(
            np.diff(data['X-Head'])**2 + 
            np.diff(data['Y-Head'])**2
        ) * fps
        
        accels = np.diff(speeds) * fps
        
        # Plot speed distribution
        ax.hist(speeds, bins='auto', alpha=0.7, label='Speed')
        ax.set_xlabel('Speed (mm/s)' if not xlabel else xlabel)
        ax.set_ylabel('Count' if not ylabel else ylabel)
        ax.set_title('Speed Distribution' if not title else title)
    else:
        # Plot distribution of provided data
        ax.hist(data, bins='auto', alpha=0.7)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.set_title(title)
    
    ax.grid(True)
    plt.tight_layout()
    return fig

def generate_flag_plot(data: np.ndarray, flags: Dict[str, np.ndarray], 
                      title: str = '') -> plt.Figure:
    """Generate plot showing flags.
    
    Args:
        data: Original data
        flags: Dictionary of flag arrays
        title: Plot title
        
    Returns:
        Figure object
    """
    fig, ax = plt.subplots()
    
    # Plot data
    x = np.arange(len(data))
    ax.plot(x, data, 'k-', alpha=0.5, label='Data')
    
    # Plot flags
    colors = ['r', 'g', 'b', 'c', 'm', 'y']
    for (name, flag_array), color in zip(flags.items(), colors):
        # Ensure flag array matches data length
        if len(flag_array) != len(data):
            flag_array = np.pad(flag_array, (0, len(data) - len(flag_array)), mode='edge')
        ax.plot(x[flag_array], data[flag_array], color + 'o', label=name)
    
    # Add labels
    ax.set_title(title)
    ax.set_xlabel('Frame')
    ax.set_ylabel('Value')
    ax.legend()
    
    return fig

def generate_comparison_plot(raw_data: pd.DataFrame, processed_data: pd.DataFrame, fps: float = 30.0) -> plt.Figure:
    """Generate a comparison plot of raw and processed trajectories.
    
    Args:
        raw_data: DataFrame with raw trajectory data
        processed_data: DataFrame with processed trajectory data
        fps: Frames per second
        
    Returns:
        Figure object
    """
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # Plot raw trajectories
    ax.plot(raw_data['X-Head'], raw_data['Y-Head'], 'b--', label='Raw Head', alpha=0.5)
    ax.plot(raw_data['X-Head'].iloc[0], raw_data['Y-Head'].iloc[0], 'bo', label='Raw Head Start')
    ax.plot(raw_data['X-Head'].iloc[-1], raw_data['Y-Head'].iloc[-1], 'b*', label='Raw Head End')
    
    ax.plot(raw_data['X-Tail'], raw_data['Y-Tail'], 'r--', label='Raw Tail', alpha=0.5)
    ax.plot(raw_data['X-Tail'].iloc[0], raw_data['Y-Tail'].iloc[0], 'ro', label='Raw Tail Start')
    ax.plot(raw_data['X-Tail'].iloc[-1], raw_data['Y-Tail'].iloc[-1], 'r*', label='Raw Tail End')
    
    # Plot processed trajectories
    ax.plot(processed_data['X-Head'], processed_data['Y-Head'], 'b-', label='Processed Head', alpha=0.7)
    ax.plot(processed_data['X-Head'].iloc[0], processed_data['Y-Head'].iloc[0], 'bs', label='Processed Head Start')
    ax.plot(processed_data['X-Head'].iloc[-1], processed_data['Y-Head'].iloc[-1], 'b^', label='Processed Head End')
    
    ax.plot(processed_data['X-Tail'], processed_data['Y-Tail'], 'r-', label='Processed Tail', alpha=0.7)
    ax.plot(processed_data['X-Tail'].iloc[0], processed_data['Y-Tail'].iloc[0], 'rs', label='Processed Tail Start')
    ax.plot(processed_data['X-Tail'].iloc[-1], processed_data['Y-Tail'].iloc[-1], 'r^', label='Processed Tail End')
    
    ax.set_xlabel('X Position')
    ax.set_ylabel('Y Position')
    ax.set_title('Trajectory Comparison')
    ax.legend()
    ax.grid(True)
    
    return fig

def plot_trajectory(data: np.ndarray, flags: Optional[Dict[str, np.ndarray]] = None,
                   title: str = '') -> plt.Figure:
    """Plot trajectory with optional flags.
    
    Args:
        data: Trajectory data
        flags: Optional dictionary of flag arrays
        title: Plot title
        
    Returns:
        Figure object
    """
    fig, ax = plt.subplots()
    
    # Plot trajectory
    ax.plot(data[:, 0], data[:, 1], 'k-', alpha=0.5, label='Trajectory')
    
    # Plot flags if provided
    if flags is not None:
        colors = ['r', 'g', 'b', 'c', 'm', 'y']
        for (name, flag_array), color in zip(flags.items(), colors):
            # Ensure flag array matches data length
            if len(flag_array) != len(data):
                flag_array = np.pad(flag_array, (0, len(data) - len(flag_array)), mode='edge')
            ax.plot(data[flag_array, 0], data[flag_array, 1], color + 'o', label=name)
    
    # Add labels
    ax.set_title(title)
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.legend()
    
    return fig

def plot_metrics(metrics: Dict[str, np.ndarray], title: str = '') -> plt.Figure:
    """Plot metrics over time.
    
    Args:
        metrics: Dictionary of metric arrays
        title: Plot title
        
    Returns:
        Figure object
    """
    fig, axes = plt.subplots(len(metrics), 1, figsize=(10, 2*len(metrics)), sharex=True)
    if len(metrics) == 1:
        axes = [axes]
    
    # Plot each metric
    for (name, values), ax in zip(metrics.items(), axes):
        x = np.arange(len(values))
        ax.plot(x, values)
        ax.set_title(name)
        ax.set_ylabel('Value')
    
    # Add labels
    axes[-1].set_xlabel('Frame')
    fig.suptitle(title)
    plt.tight_layout()
    
    return fig