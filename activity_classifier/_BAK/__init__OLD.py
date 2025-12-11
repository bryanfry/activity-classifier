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
    def __init__(self, t, X, annot=None, metadata=None):
        self.n = len(t)
        assert len(X) == self.n
        self.t = t
        self.X = X
        self.t0 = t[0]
        self.tf = t[-1]
        if metadata is not None:
            self.metadata = metadata
        else:
            metadata = {
                'value_cols': [f'x_{i+1}' for i in range(X.shape[1])],
                'time_col': 'time',
                'annot_cols': [f'annot_{i+1}' for i in range(annot.shape[1])]
            }
        if annot is not None:
            assert len(annot) == self.n
        self.annot = annot

    
    def _self_or_copy(self, inplace):
        if not inplace:
            return deepcopy(self)
        else:
            return self

    
    def get_chunks_by_samples(self, chunk_size):
        '''
        Return a list of ChunkableTimeSeries objects, each of size chunk_size.
        All chunks will have uniform number of samples. Time gaps are permissible - thus
        the time interval between first and last sample in the chunks may vary.
        If the last chunk is smaller than chunk_size, it will be discarded.
        '''
        chunks = []
        for i in range(0, self.n, chunk_size):
            t_chunk = self.t[i:i+chunk_size]
            X_chunk = self.X[i:i+chunk_size]
            annot_chunk = self.annot[i:i+chunk_size] if self.annot is not None else None
            chunks.append(ChunkableTimeSeries(t_chunk, X_chunk, annot_chunk, metadata=self.metadata))
        # Check size of last chunk
        if len(chunks[-1].X) < chunk_size:
            chunks.pop()
        # Raise error if no chunks are returned
        if len(chunks) == 0:
            raise ValueError("No chunks returned")
        return chunks


    def get_chunks_by_time(self, chunk_duration):
        '''
        Return a list of ChunkableTimeSeries objects, each of duration chunk_duration.
        All chunks will have uniform duration. Time gaps are permissible - but will
        always be of duration less than chunk_duration. The time interval between
        first and last sample in the chunks may but will not exceed chunk_duration.
        the time interval between first and last sample in the chunks may vary.
        Empty chunks will be discarded. 
        '''
        chunks = []
        chunk_t0 = self.t0
        chunk_tf = chunk_t0 + chunk_duration
        
        # Use a single pass through the data for O(n) complexity
        start_idx = 0
        end_idx = 0
        
        while chunk_tf <= self.tf:
            # Find the start index for this chunk
            while start_idx < self.n and self.t[start_idx] < chunk_t0:
                start_idx += 1
            
            # Find the end index for this chunk
            end_idx = start_idx
            while end_idx < self.n and self.t[end_idx] < chunk_tf:
                end_idx += 1
            
            # Create chunk if we have data
            if end_idx > start_idx:
                chunk = ChunkableTimeSeries(
                    self.t[start_idx:end_idx],
                    self.X[start_idx:end_idx],
                    self.annot[start_idx:end_idx] if self.annot is not None else None,
                    metadata=self.metadata
                )
                chunks.append(chunk)
            
            # Move to next chunk
            chunk_t0 = chunk_tf
            chunk_tf = chunk_t0 + chunk_duration
            start_idx = end_idx  # Start next chunk where this one ended
            
        return chunks

    def calc_vector_magnitude(self, inplace=True):
        '''
        Calculate the vector magnitude of the time series, and add as new
        column in X.
        '''
        cts = self._self_or_copy(inplace)
        mag = np.sqrt(np.sum(cts.X**2, axis=1))
        cts.X = np.column_stack([cts.X, mag])
        cts.metadata['value_cols'].append('mag')
        return cts
    

    def apply_smart_filter(self, filter_obj, inplace=False):
        '''
        Apply a SmartIIRFilter object to the time series.
        '''
        #if not isinstance(filter_obj, SmartIIRFilter):
        #    raise ValueError("filter_obj must be a SmartIIRFilter object")
        cts = self._self_or_copy(inplace)
        for i in range(cts.X.shape[1]):
            cts.X[:, i] = filter_obj.apply(cts.X[:, i], t=cts.t)
        return cts

    

    def plot(self, ax=None, show_plot=True,**kwargs):
        if ax is None:
            _, ax = plt.subplots(figsize=(14, 3))
        fig = ax.figure
        
        labels = self.metadata['value_cols']
        for i in range(self.X.shape[1]):
            ax.plot(self.t, self.X[:, i], label=labels[i], **kwargs)
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


def df_to_chunkable_ts(
    df, 
    time_col='timestamp',
    value_cols=['x', 'y', 'z'],
    sort_by_time=False,
    annot_cols=['annotation']
):
    if sort_by_time:
        df = df.sort_values(by=time_col)
    t = df[time_col].values.copy()
    X = df[value_cols].values.copy()
    if annot_cols is not None:
        annot = df[annot_cols].values.copy()
    else:
        annot = None
    cts = ChunkableTimeSeries(
        t, X, annot,
        metadata={'value_cols': value_cols, 'time_col': time_col, 'annot_cols': annot_cols}
    )
    return cts


def chunkable_ts_to_df(cts, time_col=None, value_cols=None, annot_cols=None):
    #if time_col is None:
    time_col = cts.metadata['time_col']
    #if value_cols is None:
    value_cols = cts.metadata['value_cols']
    #if annot_cols is None:
    annot_cols = cts.metadata['annot_cols']
    df = pd.DataFrame({
        time_col: cts.t, **{col: cts.X[:, i] 
        for i, col in enumerate(value_cols)}
    })
    if cts.annot is not None:
        df[cts.metadata['annot_cols']] = cts.annot
    return df
