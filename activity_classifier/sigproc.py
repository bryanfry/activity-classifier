import numpy as np
from scipy import signal

"""
[ 20 features]
Highpass-filtered X / Y / Z / HP Magnitude 
- Min
- Max
- Std
- Skewness ??
- Kurtosis ??

[ 24 features]
Lowpass-filtered X / Y / Z / LP Magnitude 
- Mean
- Min
- Max
- Std
- Skewness ??
- Kurtosis ??

[20 Features]
1st Derivative of highpass-filtered (X / Y / Z / Magnitude) 
- Mean(Abs)
- Max
- Min
- Max(Abs)
- Std(Abs)


Periodicity Features [7 Features]
-DONE- [ 4 features ] dominant frequency: (Highpass-filtered X / Y / Z / Magnitude) 
-DONE- [ 1 feature ] autocorrelation: (Highpass-filtered Vector Magnitude)
-DONE- [ 1 feature ] spectral flatness: (Highpass-filtered Vector Magnitude)
-DONE-[ 1 feature ] Coefficient of variation of zero-crossing (Highpass-filtered vector magnitude)

SKIP FOR NOW -- Janky feature
-DONE-[ 1 feature ] spectral entropy: (Highpass-filtered Vector Magnitude)

[ 9 features ] Inter-axis features 
Ratios of Mean(Abs 1st Derivative) (HP-X / HP-Y)
Ratios of Mean(Abs 1st Derivative) (HP-Y / HP-Z)
Ratios of Mean(Abs 1st Derivative) (HP-Z / HP-X)
magnitude-normalized cross-correlation (HP-X / HP-Y)
magnitude-normalized cross-correlation (HP-Y / HP-Z)
magnitude-normalized cross-correlation (HP-Z / HP-X)
relative phase (HP-X / HP-Y) in degrees
relative phase (HP-Y / HP-Z) in degrees
relative phase (HP-Z / HP-X) in degrees
"""

class SmartIIRFilter:
    '''
    Class to apply an IIR filter to a time series, with optional handling of
    transients and time gaps in input data. 
    '''
    """
    def __init__BAK(
        self, filter_func, btype, order, cutoff, fs, transient_t=None,
        max_t_gap=None, shift_times=False
    ):    
        '''
        if max_t_gap is not None:
            assert max_t_gap > 0.
        Parameters
        ----------
        filter_func : scipy.signal filter design function (butter, bessel, etc.)
        btype : str, Filter type: 'lowpass', 'highpass', 'bandpass', 'bandstop'
        order : int, Filter order
        cutoff : float, Cutoff frequency (Hz)
        fs : float Sampling rate (Hz)
        transient_t : float. When time is provided to the apply method, the initial
            outputs up to this time will be returned as NaN, in effort to remove
            filter transients. Must be None if shift_times is True
        max_t_gap : float or None. If provided, time gaps in input data of this
            interval or larger will be treated as a 'filter reset', and an internal
            of NaNs defined by transient_t will returned following these gaps.
        shift_times : bool, optional. If True, the times for all outputs will
        be decreased by 2/cutoff freq. Aim is to make filter response more 
        symmetric in time about transient signal changes. This option requires
        that fs / cutoff is an integer -- an error will be raised in this
        condition is not met. Must be False if transient_t is not None.
        '''
        if transient_t is None:
            transient_t = 0.
        
        if shift_times != False: 
            if transient_t > 0.:
                raise ValueError("shift_times must be False if transient_t is not None.")

            n_shift_samples = (2. / cutoff * fs)
            if abs(np.round(n_shift_samples) - n_shift_samples) > 1e-6:
                raise ValueError("fs / cutoff must be an integer when shift_times is True.")    
        else:
            n_shift_samples = 0
        self.n_shift_samples = int(np.round(n_shift_samples))

        self.filter_func = filter_func
        self.btype = btype
        self.order = order
        self.cutoff = cutoff
        self.fs = fs
        self.transient_t = transient_t
        self.max_t_gap = max_t_gap
        self.b, self.a = filter_func(
            N=order, Wn=cutoff, btype=btype, analog=False, output='ba', fs=fs
        )
    """
    
    """
    def apply_BAK(self, x, t=None):
        '''
        Parameters
        ----------
        x : 1-d array of values
        t : 1-d array of times, optional. If provided, this will be used to 
            identify gaps per the transient_t and max_t_gap parameters.
        '''
        if self.transient_t > 0. and t is None:
            raise ValueError("Time array must be provided when transient_t > 0.")
        xf = signal.lfilter(self.b, self.a, x)

            return xf
        elif self.n_shift_samples > 0:
        else:
            assert len(t) == len(x)
            transient_n = int(self.transient_t * self.fs)  # Number of samples to return as NaN
            if self.transient_t <= 1./self.fs:
                return xf            
            
            gap_idx = [0]
            dt = np.diff(t)
            if self.max_t_gap is not None:
                gap_idx = gap_idx + np.where(dt > self.max_t_gap)[0].tolist()
            for i in gap_idx:
                xf[i:i+transient_n] = np.nan
            return xf
    """

    def __init__(
        self, filter_func, btype, order, cutoff, fs, perform_time_shift=False,
        transient_drop_t=None, max_t_gap=None
    ):    
        """
        Parameters
        ----------
        - filter_func : scipy.signal filter design function (butter, bessel, etc.)
        - btype : str, Filter type: 'lowpass', 'highpass', 'bandpass', 'bandstop'
        - order : int, Filter order
        - cutoff : float, Cutoff frequency (Hz)
        - fs : float Sampling rate (Hz)
        - perform_time_shift : bool, optional. If True, the times for all output
          samples will be shifted backward in time by 2/cutoff freq. This means
          that the initial interval of 2/fc will NOT be returned in the output,
          and the corresponding interval at the end will be populated with NaNs.
          If perform_time_shift is True, times must be provided when the apply()
          method is called.
        - transient_drop_t : float. If provided, the initial interval of this
          duration will be set to Nan. If any time gaps are found per max_t_gap
          parameter, the intitial interval after each gap will also be set to NaN.
        - max_t_gap : float or None. If provided, time gaps in input data of this
          interval or larger will be treated as a 'filter reset', and an internal
          of NaNs defined by transient_drop_t will returned following these gaps.
          The filter is applied chunkwise the sample intervals between the gaps.
          If max_t_gap is provided, times must be provided when the apply()
          method is called.
        """
        
        if max_t_gap is not None:
            if max_t_gap <= 1./fs:
                raise ValueError("max_t_gap must be greater than 1/(sample rate)")
        
        self.filter_func = filter_func
        self.btype = btype
        self.order = order
        self.cutoff = cutoff
        self.fs = fs
        self.perform_time_shift = perform_time_shift
        self.transient_drop_t = transient_drop_t
        if self.transient_drop_t is not None:
            self.n_transient_drop = int(np.round(transient_drop_t * fs))
        else:
            self.n_transient_drop = 0
        self.max_t_gap = max_t_gap
        if perform_time_shift:
            self.n_shift_samples = int(np.round(0.5 / self.cutoff * fs))
            #\#self.n_shift_samples = int(np.round(0.5 / self.cutoff * fs))
        else:
            self.n_shift_samples = 0
        self.b, self.a = filter_func(
            N=order, Wn=cutoff, btype=btype, analog=False, output='ba', fs=fs
        )
    
    def apply(self, x, t=None):
        '''
        Parameters
        ----------
        x : 1-d array of values
        t : 1-d array of times, optional. If provided, this will be used to 
            - identify gaps
            - shift times (if )
        '''
        n = len(x)
        if t is None:
            if self.n_shift_samples > 0:
                raise ValueError("Times must be provided when perform_time_shift is True.")
            if self.transient_drop_t is not None:
                raise ValueError("Times must be provided when transient_drop_t is not None.")
            if self.max_t_gap is not None:
                raise ValueError("Times must be provided when max_t_gap is not None.")
        else:
            assert len(t) == n

        # Populate idx with bounds of chunks to be filtered. 
        # These are contiguous time intervals with gaps < max_t_gap.
        idx = [0]
        
        if self.max_t_gap is not None:
            dt = np.diff(t)
            idx = idx + np.where(dt > self.max_t_gap)[0].tolist()
        idx = idx + [n]
        
        # Apply the filter chunkwise to each interval bounded by intervals in idx
        chks = []  # To be populated with filtered chunks
        for i in range(len(idx)-1):
            idx0 = idx[i]
            idxf = idx[i+1]
            chks.append(signal.lfilter(self.b, self.a, x[idx0:idxf]))
        
        # If transient drop is requested, drop the initial interval of each chunk
        if self.transient_drop_t is not None:
            for i in range(len(chks)):
                chk = chks[i]
                chk[:self.n_transient_drop] = np.nan
        
        # If time shift is requested, shift all times backward by 2/cutoff freq
        # and put NaNs at the trailing end of each chunk.
        if self.perform_time_shift:
            for i in range(len(chks)):
                nc = len(chks[i])
                if nc <= self.n_shift_samples:
                    chks[i] = np.full(nc, np.nan)
                else:
                    chks[i] = np.concatenate([
                        chks[i][self.n_shift_samples:],
                        np.full(self.n_shift_samples, np.nan)])
        
        return np.concatenate(chks)


def scalar_autocorrelation(x, fs, boxcar_width=5, skip_lags=20, nan_on_error=False):
    """
    Ingest 1d signal and return lags, autocorrelation, and maximum correlation 
    at first period. Works best if signal is band-limited (tested to 10 Hz
    lowpass at 100 Hz sample rate). Amplitude independent.
    """
    assert skip_lags > 1, "skip_lags must be greater than 1"
    
    try:
        ax = np.asarray(x)
        n = len(x)
        if len(x) < 2:
            return np.nan
        x = x - x.mean()
        cor = signal.correlate(x, x, mode='full')
        cor = cor / cor.max()
        
        if boxcar_width is not None and boxcar_width > 1:
            boxcar = np.ones(boxcar_width)/boxcar_width
            cor = np.convolve(cor, boxcar, mode='same')
        
        cor = cor[n+skip_lags-1:]  # limit to positive lags

        # Need to locate the first positive-to-negative zero crossing
        # in the first derivative of the autocorrelation curve. The location here
        # should correpond to the maximum periodic correction lag.

        dc = np.diff(cor)
        is_pos = np.sign(dc) > 0
        is_zc = np.logical_and(is_pos[:-1], np.logical_not(is_pos[1:]))
        first_zc_idx = np.where(is_zc)[0][0] + 1

        lags = signal.correlation_lags(n, n)[n+skip_lags-1:] / fs
        max_cor = cor[first_zc_idx]
        max_lag_time = lags[first_zc_idx]
    
    except Exception as e:
        if nan_on_error:
            return np.nan, np.nan, np.nan, np.nan
        else:
            raise e
    return cor, lags, max_cor, max_lag_time



def scalar_lagged_cross_correlation(x, y, fs, boxcar_width=5, nan_on_error=False):
    try:
        if x.shape != y.shape:
            raise ValueError("x and y must have same shape")
        n = len(x)
        x = x - x.mean()
        y = y - y.mean()
        cc = signal.correlate(x, y, mode='full')
        if boxcar_width is not None and boxcar_width > 1:
            boxcar = np.ones(boxcar_width)/boxcar_width
            cc = np.convolve(cc, boxcar, mode='same')
        
        
        # All-else equal, the cross correlation is largest (in abs value) near
        # zero lag where there is a maximum in the product of the two signals. We 
        # look for the maximum absolute value of the cross correlation, and 
        # then normalize at that lag.
        
        i = np.nanargmax(np.abs(cc))  # index of max abs cross correlation
        denom = np.sqrt(
            signal.correlate(x*x, np.ones_like(y), mode='full')*
            signal.correlate(np.ones_like(x), y*y, mode='full')
        )
        lags = signal.correlation_lags(n, n)
        ncc = cc[i] / denom[i]
        lag_time = lags[i] / fs
        return ncc, lag_time
    
    except Exception as e:
        if nan_on_error:
            return np.nan, np.nan
        else:
            raise e


def mag_phase_spectrum(x, fs, demean=True, window_func=np.hanning):
    if demean:
        x = x - x.mean()
    
    if window_func is not None:
        x = x * window_func(len(x))
    
    X = np.fft.fft(x)
    N = len(x)

    # Frequency axis, limit to positive frequencies
    freqs = np.fft.fftfreq(N, d=1/fs)
    pos_mask = freqs >= 0
    freqs = freqs[pos_mask]
    X = X[pos_mask]

    mag = np.abs(X)
    phase = np.angle(X)
    return mag, phase, freqs


def dominant_frequency(freqs, mag, phase):
    i = np.argmax(mag)
    return freqs[i], mag[i], phase[i], i


def spectral_entropy(mag):
    P = mag**2  # power spectrum
    P_norm = P / np.sum(P) # Normalize to probability distribution
    #entropy = -np.sum(P_norm * np.log(P_norm + 1e-12)) # Shannon entropy
    entropy = -np.sum(P_norm * np.log(P_norm + 0.)) # Shannon entropy
    entropy = entropy / np.log(len(P_norm)) # Normalize
    return entropy


def spectral_flatness(mag, nan_on_error=False):
    try:
        P = mag**2
        eps = 1e-12
        G = np.exp(np.mean(np.log(P + eps)))
        A = np.mean(P)
        return G / A
    except Exception as e:
        if nan_on_error:
            return np.nan
        else:
            raise e


def coefficient_of_variation_of_zero_crossing(x, nan_on_error=False):
    try:
        x = np.asarray(x)
        x = x - x.mean()
        zc = np.where(np.diff(np.sign(x)))[0]
        if len(zc) < 3:
            return np.nan
        zci = np.diff(zc)  # zero-crossing intervals
        return np.std(zci) / np.mean(zci)
    except Exception as e:
        if nan_on_error:
            return np.nan
        else:
            raise e
