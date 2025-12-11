import os
import numpy as np
from copy import deepcopy
import pandas as pd
import matplotlib.pyplot as plt


CAP24_DIR = '/Users/bryanfry/data/capture24'
            
            
class ChunkableTimeSeries:
    '''
    Class to represent a multi-dimensional time series that can be chunked by samples or time.
    
    Parameters
    ----------
    t : 1-d array of times
    X : 2-d array of values
    annot : 1-d array of annotations, optional
    metadata : dict, optional
        Metadata dictionary containing information about the time series.
        Keys:
        - 'value_cols' : list of column names for the value columns
        - 'time_col' : name of the time column
        - 'annot_cols' : list of column names for the annotation columns
    '''
    def __init__(
        self, df, time_col='timestamp', value_cols=['x', 'y', 'z'],
        annot_cols=['annotation'], sort_by_time=False, metadata=dict()
        ):
        self.df = df
        self.time_col = time_col
        self.value_cols = value_cols.copy()    
        self.annot_cols = annot_cols.copy()
        self.other_cols = [
            col for col in df.columns 
            if col not in [time_col, *value_cols, *annot_cols]
        ]
        if sort_by_time:
            self.df = self.df.sort_values(by=time_col)
        self.t0 = self.get_time()[0]
        self.tf = self.get_time()[-1]
        self.metadata = metadata.copy()
        
    
    def __len__(self):
        return len(self.df)
    
    
    def copy(self):
        return deepcopy(self)


    def refresh(self):
        '''
        - Reorder columns in DataFrame
        - Reset index
        - Reset t0, tf
        '''
        self.df = self.df[
            [self.time_col, *self.value_cols, *self.annot_cols, *self.other_cols]
        ].reset_index(drop=True)
        self.t0 = self.get_time()[0]
        self.tf = self.get_time()[-1]
        
    def _self_or_copy(self, inplace):
        if not inplace:
            return self.copy()
        else:
            return self

    def _add_value(self, col, x):
        self.df[col] = x
        self.value_cols.append(col)
        self.refresh()
        return self

    def _add_annot(self, col, a):
        # TOOD - test
        self.df[col] = a
        self.annot_cols.append(col)
        self.refresh()
        return self

    def _add_other(self, col, x):
        # TOOD - test
        self.df[col] = x
        self.other_cols.append(col)
        self.refresh()
        return self

    def _remove_value(self, col):
        self.df.drop(columns=[col], inplace=True)
        self.value_cols.remove(col)
        self.refresh()
        return self

    def get_chunk_by_idx(self, i0, i_end, sort_by_time=False):
        #return ChunkableTimeSeries(
        return self.__class__(
            df=self.df.iloc[i0:i_end].copy(),
            time_col=self.time_col,
            value_cols=self.value_cols,
            annot_cols=self.annot_cols,
            sort_by_time=sort_by_time,
            metadata=self.metadata
        )
    
    def get_time(self):
        return self.df[self.time_col].values
        
    def get_values(self):
        return self.df[self.value_cols].values
    
    def get_annot(self):
        return self.df[self.annot_cols].values

    def get_duration(self):
        return self.tf - self.t0

    def get_chunks_by_samples(self, chunk_size, min_n=None):
        
        chunks = []
        for i in range(0, len(self.df), chunk_size):
            chunk = self.get_chunk_by_idx(i, i+chunk_size)
            chunks.append(chunk)
        if min_n is not None:
            chunks = [c for c in chunks if len(c) >= min_n]
        if len(chunks) == 0:
            raise ValueError("No chunks returned")
        return chunks


    def get_chunks_by_time(self, chunk_duration, min_dur=None):
        chunks = []
        
        chunk_t0 = self.t0
        chunk_tf = chunk_t0 + chunk_duration
        start_idx = 0
        end_idx = 0
        t = self.get_time()

        # Loop though data on single pass.

        while True:

            # Locate end of upcoming chunk
            while end_idx < len(self) and t[end_idx] < chunk_tf:
                end_idx += 1

            # Create chunk, if greater then len(0)
            if end_idx > start_idx:
                chunk = self.get_chunk_by_idx(start_idx, end_idx, sort_by_time=False)
                chunks.append(chunk)
            if chunk_tf >= self.tf:
                break

            # Define time bounds for next chunk
            chunk_t0 = chunk_tf
            chunk_tf = chunk_t0 + chunk_duration
            start_idx = end_idx  # Start next chunk just past end of previous chunk

        # Filter by duration, if requested
        if min_dur is not None:
            chunks = [c for c in chunks if c.get_duration() >= min_dur]
        if len(chunks) == 0:
            raise ValueError("No chunks returned")
        return chunks


    def decimate(self, factor, inplace=True):
        '''
        Decimate time and values columns by a factor. Decimation is purely
        indexed-based and does not account for actual sample intervals.
        '''
        cts = self._self_or_copy(inplace)
        cts.df = cts.df.iloc[::factor]
        cts.refresh()
        return cts

    
    def calc_vector_magnitude(self, mag_col='mag', inplace=True):
        '''
        Calculate the vector magnitude from all values columns, and add as new
        column in X called 'mag'. If the mag column is already present, make no
        change.
        '''
        cts = self._self_or_copy(inplace)
        if 'mag' in cts.value_cols:
            return cts
        mag = np.sqrt(np.sum(cts.get_values()**2, axis=1))
        cts._add_value(mag_col, mag)
        return cts


    def limit_time_interval(self, t0=None, tf=None, inplace=True):
        '''
        Limit the time interval of the time series to the given time interval.
        If t0 is None, use the start time of the time series.
        If tf is None, use the end time of the time series.
        '''
        if t0 is None:
            t0 = self.t0
        if tf is None:
            tf = self.tf
        cts = self._self_or_copy(inplace)
        cts.df = cts.df[
            (cts.df[self.time_col] >= t0) and (cts.df[self.time_col] <= tf)]
        cts.refresh()
        return cts
    
    def apply_smart_filter(
        self, filter_obj, col_sfx=None, inplace=True, verbose=False):
        '''
        Apply a SmartIIRFilter object to all value columns in the dataframe.
        If col_sfx is provided, place outputs in new columns with sfx applied
        to existing column names. If col_sfx is None or empty, overwrite existing
        columns.
        '''
        if col_sfx is None:
            col_sfx = ''
        cts = self._self_or_copy(inplace)
        for c in cts.value_cols.copy():  # Copy to avoid infinite loop when adding new columns
            if col_sfx == '':  # Overwrite existing columns
                cts.df[c] = filter_obj.apply(
                    x=cts.df[c].values, t=cts.get_time())
            else:  # Add new columns with sfx applied to existing columns
                cts._add_value(f'{c}{col_sfx}', filter_obj.apply(
                    x=cts.df[c].values, t=cts.get_time()))
            if verbose:
                print(f'applying {filter_obj.btype} filter to column... {c}')
        cts.refresh()
        return cts  


    def plot(self, ax=None, show_plot=True,**kwargs):
        if ax is None:
            _, ax = plt.subplots(figsize=(14, 3))
        fig = ax.figure
        
        for i in range(self.get_values().shape[1]):
            #ax.plot(self.t, self.X[:, i], label=labels[i], **kwargs)
            ax.plot(
                self.get_time(), self.get_values()[:, i],
                label=self.value_cols[i], **kwargs)
        ax.legend()
        
        if show_plot:
            plt.show()
        return fig, ax


def load_acc_data_file(
    fp=None, compression='gzip', convert_datetime=True, add_timestamp_col=True,
    calc_ENMO=False
):
    '''
    Load an accelerometer data file and preprocess times. If FP is None, the
    alphabetically first gzip file in the CAP24_DIR is loaded.
    '''
    if fp is None:
        fn = sorted([f for f in os.listdir(CAP24_DIR) if f.endswith('.csv.gz')])[0]
        fp = f'{CAP24_DIR}/{fn}'
    df = pd.read_csv(fp, compression=compression)
    if convert_datetime:
        df['time'] = pd.to_datetime(df['time'])
    if add_timestamp_col:
        df['timestamp'] = df['time'].apply(lambda x: x.timestamp())
        df['timestamp'] = df['timestamp'] - df['timestamp'].iloc[0]
    if calc_ENMO:
        df['ENMO'] = np.sqrt(df['x']**2 + df['y']**2 + df['z']**2) - 1
        df['ENMO_diff'] = df['ENMO'].diff()
    return df    
