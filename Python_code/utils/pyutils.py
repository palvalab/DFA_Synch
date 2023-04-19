import statsmodels.api as sm
from scipy.optimize import curve_fit

import numpy as np
import scipy as sp
import scipy.signal

from typing import Tuple, Sequence, Optional

import cupy as cp
import numpy as np

def _normalize_signal(x: np.ndarray) -> np.ndarray:
    eps = 1e-10
    xp = cp.get_array_module(x)

    x_abs = xp.abs(x)

    x_norm = x.copy()
    x_norm /= (x_abs + eps)

    return x_norm


def cplv(x: np.ndarray, y: Optional[np.ndarray]=None, is_normed: bool=False) -> float:
    """
        computes cPLV either for a given pair of complex signals or for all possible pairs of channels if x is 2d matrix
    :param x: 1d or 2d array of complex values
    :param y: Optional, 1d or 2d array of complex values
    :return: complex phase locking value
    """
    xp = cp.get_array_module(x)

    n_ts = x.shape[1]
    
    if is_normed:
        x_norm = x
        y_norm = x_norm if y is None else y
    else:
        x_norm = _normalize_signal(x)
        y_norm = x_norm if y is None else _normalize_signal(y)

    avg_diff = xp.inner(x_norm, xp.conj(y_norm)) / n_ts

    return avg_diff


class CustomNorm(sm.robust.norms.TukeyBiweight):
    ''' 
    Custom Norm Class using Tukey's biweight (bisquare).
    '''
    
    def __init__(self, weights, c=4.685, **kwargs):
        super().__init__(**kwargs)
        self.weights_vector = np.array(weights)
        self.flag = 0
        self.c = c
        
    def weights(self, z):
        """
            Instead of weights equal to one return custom
        INPUT:
            z : 1D array or list
        OUTPUT:
            weights: ndarray
        """
        if self.flag == 0:
            self.flag = 1
            return self.weights_vector.copy()
        else:
            subset = self._subset(z)
            return (1 - (z / self.c)**2)**2 * subset

def _dfa_boxcar(data_orig, win_lengths, xp):
    '''            
    Computes DFA using FFT-based method. (Nolte 2019 Sci Rep)
    Input: 
        data_orig:   1D array of amplitude time series.
        win_lenghts: 1D array of window lengths in samples.
    Output:
        fluct: Fluctuation function.
        slope: Slopes.
    
    '''
    data = xp.array(data_orig, copy=True)
    win_arr = xp.array(win_lengths)
    
    data -= data.mean(axis=1, keepdims=True)
    data_fft = xp.fft.fft(data)

    n_chans, n_ts = data.shape
    is_odd = n_ts % 2 == 1

    nx = (n_ts + 1)//2 if is_odd else n_ts//2 + 1
    data_power = 2*xp.abs(data_fft[:, 1:nx])**2

    if is_odd == False:
        data_power[:,~0] /= 2
        
    ff = xp.arange(1, nx)
    g_sin = xp.sin(xp.pi*ff/n_ts)
    
    hsin = xp.sin(xp.pi*xp.outer(win_arr, ff)/n_ts)
    hcos = xp.cos(xp.pi*xp.outer(win_arr, ff)/n_ts)

    hx = 1 - hsin/xp.outer(win_arr, g_sin)
    h = (hx / (2*g_sin.reshape(1, -1)))**2

    f2 = xp.inner(data_power, h)

    fluct = xp.sqrt(f2)/n_ts

    hy = -hx*(hcos*xp.pi*ff/n_ts - hsin/win_arr.reshape(-1,1)) / xp.outer(win_arr, g_sin)
    h3 = hy/(4*g_sin**2)

    slope = xp.inner(data_power, h3) / f2*win_arr
    
    return fluct, slope

def _fit_tukey(x,y,weights):
    '''
    Fit using Tukey's biweight function.    
    '''
    
    N = len(y)
    X = np.array([[1, x[i]] for i in range(N)])
    
    rlm_model = sm.RLM(y,X,M=CustomNorm(weights=weights,c=4.685))
    rlm_results = rlm_model.fit()
    
    b,a = rlm_results.params    
    return a,b

def fit_tukey(weighting: str, fluct: np.ndarray, N_samp: int, window_lengths: Sequence) -> Tuple[np.ndarray, np.ndarray]:
    # I am not sure if this is the correct way to get weights for Tukey!
    N_CH = fluct.shape[0]
    if weighting == 'sq1ox':
        weights = [np.sqrt(1/x) for x in N_samp/window_lengths]
    elif weighting == '1ox':
        weights = [(1/x) for x in N_samp/window_lengths]  
        
    dfa_values = np.zeros(N_CH)
    residuals  = np.zeros(N_CH)  
    x = np.log2(window_lengths)         
    for i in range(N_CH):
        y = np.log2(fluct[i])
        dfa_values[i], residuals[i] = _fit_tukey(x,y,weights)

    return dfa_values, residuals

def fit_weighted(weighting: str, fluct: np.ndarray, N_samp: int, window_lengths: Sequence) -> Tuple[np.ndarray, np.ndarray]:
    
    N_CH = fluct.shape[0]
    if weighting == 'sq1ox':
        sigma = [np.sqrt(1/x) for x in N_samp/window_lengths]
    elif weighting == '1ox':
        sigma = [(1/x) for x in N_samp/window_lengths]
    
    dfa_values = np.zeros(N_CH)
    residuals  = np.zeros(N_CH)        
    p0 = 0.7,-8                    # might have to be parametrized? or not?     
    x=np.log2(window_lengths) 
    for i in range(N_CH):
        y=np.log2(fluct[i])
        popt, pcov = curve_fit(f, x , y, p0, sigma = sigma,absolute_sigma=True)
        dfa_values[i], residuals[i] = popt

    return dfa_values, residuals


def dfa(data: np.ndarray, window_lengths: Sequence, method: str='boxcar', 
            use_gpu: bool=False, fitting ='Tukey', weighting = 'sq1ox') -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:    
    """
    Compute DFA with conventional (windowed) or FFT-based 'boxcar' method.
    
    INPUT:
        data:           2D array of size [N_channels x N_samples]. Should be normalized data!!
        window_lengths: sequence of window sizes, should be in samples.
        method:         either 'conv' or 'boxcar' 
        use_gpu:        If True, input np.array is converted to cp in function
        fitting:        'linfit' for regular unweighted linear fit, 
                        'Tukey' for biweight/bisquare,
                        'weighted' for weighted linear fit.
        weighting:      'sq1ox' or '1ox' 
                    
    OUTPUT:        
        fluctuation: 2D array of size N_channels x N_windows), 
        slope:       2D array of size N_channels x N_windows), 
        DFA:         1D vector of size N_channels 
        residuals:   1D vector of size N_channels        
    """

    module = cp.get_array_module(data, use_gpu)
    
    N_samp = data.shape[1]
    
    allowed_methods = ('boxcar','conv' )
    if not(method in allowed_methods):
        raise RuntimeError('Method {} is not allowed! Only {} are available'.format(method, ','.join(allowed_methods)))

    allowed_weightings = ('sq1ox', '1ox')
    if not(weighting in allowed_weightings):
        raise RuntimeError('Weighting {} is not allowed! Only {} are available'.format(weighting, ','.join(allowed_weightings)))

    if method == 'conv':
        fluct, slope =  _dfa_conv(data, window_lengths, xp=module)
    elif method == 'boxcar':
        fluct, slope =  _dfa_boxcar(data, window_lengths, xp=module) 
        
    if not(module is np):
        fluct = module.asnumpy(fluct)
        slope = module.asnumpy(slope)    
    
    if fitting == 'weighted':
        dfa_values, residuals = fit_weighted(weighting, fluct, N_samp, window_lengths)

    elif fitting == 'Tukey':
        dfa_values, residuals = fit_tukey(weighting, fluct, N_samp, window_lengths)
        
    elif fitting == 'linfit':           
        dfa_values, residuals = np.polyfit(np.log2(window_lengths), np.log2(fluct.T), 1)
    
    return fluct, slope, dfa_values, residuals