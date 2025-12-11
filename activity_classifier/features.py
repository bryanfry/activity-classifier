from activity_classifier import sigproc
import activity_classifier as ac
import numpy as np
import pandas as pd
from scipy import stats, signal


FIRST_PASS_FILTER = sigproc.SmartIIRFilter(
    signal.butter, 'lowpass', order=4, cutoff=12, fs=100, perform_time_shift=True)
DECIMATION_FACTOR = 2
LP_FILTER = sigproc.SmartIIRFilter(
    signal.butter, 'lowpass', order=4, cutoff=0.25, fs=100, perform_time_shift=True)
HP_FILTER = sigproc.SmartIIRFilter(
    signal.butter, 'highpass', order=4, cutoff=0.25, fs=100, perform_time_shift=False)

AUTOCORRELATION_BOXCAR_WIDTH = 5
AUTOCORRELATION_SKIP_LAGS = 20
"""
For all data in given subject:
- Read data from csv file
- Ingest t, x, y, z in ChunkableTimeSeries
- Compute magnitude
- Apply LP and HP filters
- Get chunks
- Extract all features
"""


class CAP24TimeSeries(ac.ChunkableTimeSeries):

    def __init__(self, df, **kwargs):
        super().__init__(df, **kwargs)
        self.is_preprocessed = False
    
    def preprocess_time_series(
        self, first_pass_filter=FIRST_PASS_FILTER,
        decimation_factor=DECIMATION_FACTOR, lp_filter=LP_FILTER,
        hp_filter=HP_FILTER, inplace=True, verbose=False):    
        """
        Preprocess the time series by applying filters and decimation, using 
        following steps:
        
        - Apply first pass filter to prevent aliasing.
        - Decimate time and values columns by a decimation factor.
        - Compute vector magnitude for all filtered values columns.
        - Apply highpass filter to original values columns.
        - Apply lowpass filter to original values columns.
        - Return new CAP24TimeSeries instance with 8 values:
            - x_lp
            - y_lp
            - z_lp
            - mag_lp
            - x_hp
            - y_hp
            - z_hp
            - mag_hp

        Parameters
        ----------
        lp_filter : SmartIIRFilter, optional
        hp_filter : SmartIIRFilter, optional
        first_pass_filter : SmartIIRFilter, optional
        decimation_factor : int, optional
        inplace : bool, optional
        
        Returns
        -------
        ctsr : CAP24TimeSeries following processing.

        """
        cts = self._self_or_copy(inplace)

        if first_pass_filter is not None:
            cts.apply_smart_filter(first_pass_filter, inplace=True)

        if decimation_factor is not None:
            cts.decimate(decimation_factor, inplace=True)
        
        cts.calc_vector_magnitude(inplace=True)

        cts_lp = cts.apply_smart_filter(
            lp_filter, col_sfx='_lp', inplace=False, verbose=verbose)
        
        cts_hp = cts.apply_smart_filter(
            hp_filter, col_sfx='_hp', inplace=False, verbose=verbose)
                
        df = pd.DataFrame(data={
            'timestamp': cts_lp.df['timestamp'].values,
            'x_lp': cts_lp.df['x_lp'].values,
            'y_lp': cts_lp.df['y_lp'].values,
            'z_lp': cts_lp.df['z_lp'].values,
            'mag_lp': cts_lp.df['mag_lp'].values,
            'x_hp': cts_hp.df['x_hp'].values,
            'y_hp': cts_hp.df['y_hp'].values,
            'z_hp': cts_hp.df['z_hp'].values,
            'mag_hp': cts_hp.df['mag_hp'].values,
        })
        cts = CAP24TimeSeries(
            df=df, time_col='timestamp',
            value_cols=['x_lp', 'y_lp', 'z_lp', 'mag_lp', 'x_hp', 'y_hp', 'z_hp', 'mag_hp'],
            annot_cols=['annotation'],

        )
        cts.is_preprocessed = True
        return cts
        
    def get_feature_vector(self, fs):
        
        dt = 1. / fs
        vec = pd.Series()
        vec['t0'] = self.t0
        vec['duration'] = self.get_duration()
        
        # Highpass-filtered X / Y / Z / mag stats [ 20 features]
        for c in ['x', 'y', 'z', 'mag']:
            vec[f'FTR_{c}_hp_min'] = self.df[f'{c}_hp'].min()
            vec[f'FTR_{c}_hp_max'] = self.df[f'{c}_hp'].max()
            vec[f'FTR_{c}_hp_std'] = self.df[f'{c}_hp'].std()
            vec[f'FTR_{c}_hp_skewness'] = stats.skew(self.df[f'{c}_hp'])
            vec[f'FTR_{c}_hp_kurtosis'] = stats.kurtosis(self.df[f'{c}_hp'])

        # Lowpass-filtered X / Y / Z / mag stats [ 24 features]
        for c in ['x', 'y', 'z', 'mag']:
            vec[f'FTR_{c}_lp_mean'] = self.df[f'{c}_lp'].mean()
            vec[f'FTR_{c}_lp_min'] = self.df[f'{c}_lp'].min()
            vec[f'FTR_{c}_lp_max'] = self.df[f'{c}_lp'].max()
            vec[f'FTR_{c}_lp_std'] = self.df[f'{c}_lp'].std()
            vec[f'FTR_{c}_lp_skewness'] = stats.skew(self.df[f'{c}_lp'])
            vec[f'FTR_{c}_lp_kurtosis'] = stats.kurtosis(self.df[f'{c}_lp'])
            
        
        # Compute 1st Derivatives of abs highpass-filtered axes, save in dict
        dd = {
            'x' : np.abs(np.diff(self.df['x_hp'].values)) * fs,
            'y' : np.abs(np.diff(self.df['y_hp'].values)) * fs,
            'z' : np.abs(np.diff(self.df['z_hp'].values))* fs,
            'mag' : np.abs(np.diff(self.df['mag_hp'].values)) * fs
        }
        
        # 1st Derivative of highpass-filtered stats (X / Y / Z / Magnitude) [ 20 features]
        for c in ['x', 'y', 'z', 'mag']:
            vec[f'FTR_{c}_hp_deriv_max'] = np.max(np.abs(np.diff(self.df[f'{c}_hp'].values))) * fs
            vec[f'FTR_{c}_hp_deriv_min'] = np.min(np.abs(np.diff(self.df[f'{c}_hp'].values))) * fs  
            vec[f'FTR_{c}_hp_deriv_mean_abs'] = np.mean(dd[c])
            vec[f'FTR_{c}_hp_deriv_max_abs'] = np.max(dd[c])
            vec[f'FTR_{c}_hp_deriv_std_abs'] = np.std(dd[c])
        
        # Compute FFTs (highpass-filtered X / Y / Z / Magnitude)
        mags_x, phases_x, freqs_x = sigproc.mag_phase_spectrum(self.df['x_hp'], fs)
        mags_y, phases_y, freqs_y = sigproc.mag_phase_spectrum(self.df['y_hp'], fs)
        mags_z, phases_z, freqs_z = sigproc.mag_phase_spectrum(self.df['z_hp'], fs)
        mags_mag, phases_mag, freqs_mag = sigproc.mag_phase_spectrum(self.df['mag_hp'], fs)
        
        # Get dominant frequency /mag / phase for each axis
        fx, mx, px, ix = sigproc.dominant_frequency(freqs_x, mags_x, phases_x)
        fy, my, py, iy = sigproc.dominant_frequency(freqs_y, mags_y, phases_y)
        fz, mz, pz, iz = sigproc.dominant_frequency(freqs_z, mags_z, phases_z)
        fmag, mmag, pmag, imag = sigproc.dominant_frequency(freqs_mag, mags_mag, phases_mag)
        
        # Assign features for dominants frequencies / magnitudes (8 features)
        vec[f'FTR_dominant_freq_x'] = fx
        vec[f'FTR_dominant_freq_y'] = fy
        vec[f'FTR_dominant_freq_z'] = fz
        vec[f'FTR_dominant_freq_mag'] = fmag
        vec[f'FTR_dominant_mag_x'] = mx
        vec[f'FTR_dominant_mag_y'] = my
        vec[f'FTR_dominant_mag_z'] = mz
        vec[f'FTR_dominant_mag_mag'] = mmag
        
        # Assign features for vector magnitude periodicity (4 features)
        
        _, _, ac_mag, lag_time_mag = sigproc.scalar_autocorrelation(
            self.df['mag_hp'].values, fs, AUTOCORRELATION_BOXCAR_WIDTH,
            AUTOCORRELATION_SKIP_LAGS, nan_on_error=True)
        vec[f'FTR_mag_autocorrelation_max'] = ac_mag
        vec[f'FTR_mag_autocorrelation_lag_time'] = lag_time_mag
        
        vec[f'FTR_mag_spectral_flatness'] = sigproc.spectral_flatness(mags_mag, nan_on_error=True)
        vec[f'FTR_CoV_zero_crossing_mag'] = sigproc.coefficient_of_variation_of_zero_crossing(
            self.df['mag_hp'].values, nan_on_error=True)
        
        # Assign features for inter-axis phase differences (3 features)
        vec['FTR_xy_phase_diff'] = phases_x[ix] - phases_y[ix]
        vec['FTR_yz_phase_diff'] = phases_y[iy] - phases_z[iy]
        vec['FTR_zx_phase_diff'] = phases_z[iz] - phases_x[iz]

        # Assign Inter-axis ratios of mean absolute 1st derivative (3 features)
        vec[f'FTR_ratio_mean_abs_deriv_xy'] = np.mean(dd['x']) / np.mean(dd['y'])
        vec[f'FTR_ratio_mean_abs_deriv_yz'] = np.mean(dd['y']) / np.mean(dd['z'])
        vec[f'FTR_ratio_mean_abs_deriv_zx'] = np.mean(dd['z']) / np.mean(dd['x'])
        
        
        # Inter-axis magnitude normalized cross-correlation
        ncc_xy, lag_time_xy = sigproc.scalar_lagged_cross_correlation(
            self.df['x_hp'].values, self.df['y_hp'].values, fs, nan_on_error=True)
        ncc_yz, lag_time_yz = sigproc.scalar_lagged_cross_correlation(
            self.df['y_hp'].values, self.df['z_hp'].values, fs, nan_on_error=True)        
        ncc_zx, lag_time_zx = sigproc.scalar_lagged_cross_correlation(
            self.df['z_hp'].values, self.df['x_hp'].values, fs, nan_on_error=True)
        vec[f'FTR_mag_norm_cross_corr_xy'] = ncc_xy
        vec[f'FTR_mag_norm_cross_corr_yz'] = ncc_yz
        vec[f'FTR_mag_norm_cross_corr_zx'] = ncc_zx
        vec[f'FTR_lag_time_xy'] = lag_time_xy
        vec[f'FTR_lag_time_yz'] = lag_time_yz
        vec[f'FTR_lag_time_zx'] = lag_time_zx
        return vec

    def extract_features_dataframe(
        self, chunk_duration=10., fs=100./DECIMATION_FACTOR
        ):
        """
        Parameters
        ----------
        cts : ChunkableTimeSeries with t, x, y, z and annotation for one subject
        chunk_duration : int, optional
        fs : int, optional
        """
        if not self.is_preprocessed:
            raise ValueError("Time series must be preprocessed before feature extraction. Call preprocess_time_series() first.")
        chunks = self.get_chunks_by_time(chunk_duration)
        
        vs = [pd.Series() for _ in range(len(chunks))]
        for i, chunk in enumerate(chunks):
            if i % 100 == 0:
                print(f'extracting features for chunk {i} of {len(chunks)}')
            vs[i] = chunk.get_feature_vector(fs)
            # Add columns in the feature vector with nan counts for each filtered axis:
            for c in ['x_lp', 'y_lp', 'z_lp', 'x_hp', 'y_hp', 'z_hp']:
                vs[i][f'QC_nan_count_{c}'] = chunk.df[c].isnull().sum()
    
        dff = pd.DataFrame(vs)
        return dff


def main():
    files = sorted(
        [f'{CAP24_DIR}/{f}' for f in os.listdir(CAP24_DIR)
        if f.endswith('.csv.gz')]
    )
    for fp in files:
        df = ac.load_acc_data_file(fp, compression='gzip')
        cts = CAP24TimeSeries(df)
        cts.preprocess_time_series(inplace=True)
        dff = cts.extract_features_dataframe(chunk_duration=10, fs=100)
        dff.to_csv(f'{CAP24_DIR}/features/{fn.replace('.csv.gz', '.csv')}', index=False)

if __name__ == '__main__':
    main()