"""

--- obsolete since June 2023 ---


Copyright (C) 2023 Vladislav Myrov & Felix Siebenhühner

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. 
See the GNU General Public License for more details.
<http://www.gnu.org/licenses/>.

"""

import sys
source_directory = ''             # insert path to code directory here
sys.path.append(source_directory+'utils')

import os
import numpy as np
import scipy as sp
import scipy.io
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import tqdm 
from pyutils import cplv, dfa
from kuramoto import KuramotoFast

def simulate_single_run(k, node_frequencies, time=10, n_oscillators=500, frequency_spread=5, normalize_external=False,
                       noise_sigma=5, sr=200, weight_matrix=None, use_tqdm=True, omega_seed=42, random_seed=42):
    np.random.seed(omega_seed)
    
    n_nodes = len(node_frequencies)
    
    if weight_matrix is None:
        w = np.zeros((n_nodes, n_nodes))
    else:
        w = weight_matrix.copy()
        
    
    model = KuramotoFast(n_nodes=n_nodes, n_oscillators=n_oscillators, sampling_rate=sr, 
                         k_list=[k]*n_nodes, weight_matrix=w, use_tqdm=use_tqdm,
                         node_frequencies=node_frequencies, normalize_external=normalize_external,
                         frequency_spread=frequency_spread, noise_scale=noise_sigma, use_cuda=True)

    return model.simulate(time=time, random_seed=random_seed)

def smooth(x,window_len=11,window='hanning'):
    """smooth the data using a window with requested size.
    
    This method is based on the convolution of a scaled window with the signal.
    The signal is prepared by introducing reflected copies of the signal 
    (with the window size) in both ends so that transient parts are minimized
    in the begining and end part of the output signal.
    
    input:
        x: the input signal 
        window_len: the dimension of the smoothing window; should be an odd integer
        window: the type of window from 'flat', 'hanning', 'hamming', 'bartlett', 'blackman'
            flat window will produce a moving average smoothing.
​
    output:
        the smoothed signal
        
    example:
​
    t=linspace(-2,2,0.1)
    x=sin(t)+randn(len(t))*0.1
    y=smooth(x)
    
    see also: 
    
    numpy.hanning, numpy.hamming, numpy.bartlett, numpy.blackman, numpy.convolve
    scipy.signal.lfilter
 
    TODO: the window parameter could be the window itself if an array instead of a string
    NOTE: length(output) != length(input), to correct this: return y[(window_len/2-1):-(window_len/2)] instead of just y.
    """

    if x.ndim != 1:
        raise ValueError("smooth only accepts 1 dimension arrays.")

    if x.size < window_len:
        raise ValueError("Input vector needs to be bigger than window size.")

    if window_len<3:
        return x

    if not window in ['flat', 'hanning', 'hamming', 'bartlett', 'blackman']:
        raise ValueError("Window is on of 'flat', 'hanning', 'hamming', 'bartlett', 'blackman'")

    s=np.r_[x[window_len-1:0:-1],x,x[-2:-window_len-1:-1]]
    #print(len(s))
    if window == 'flat': #moving average
        w=np.ones(window_len,'d')
    else:
        w=eval('np.'+window+'(window_len)')

    y=np.convolve(w/w.sum(),s,mode='valid')
    return y[(window_len//2-1):-(window_len//2)-1]



# create subfolder "simulations"

os.makedirs(source_directory+'simulations',exist_ok=True) 
os.chdir(source_directory)


# load structural connectome

connectome_orig = sp.io.loadmat('metadata/HCP_Schaefer100_number_alphb_interlvd_nosubctx.mat')['sc_data']
connectome_real = np.log(connectome_orig + 1).mean(axis=-1)
connectome_real = connectome_real / connectome_real.mean()

# settings

sr = 200
node_frequencies = np.array([10]*100)
window_lengths   = np.geomspace(sr, sr*60, num=50)
k_values         = np.linspace(8, 14, 100)

# run simulations

for i, k in enumerate(tqdm.tqdm(k_values)):
    fpath = f'simulations/sim_{i}_l_value.npy'
    
    if os.path.exists(fpath):
        continue

    simulated = simulate_single_run(k, node_frequencies, weight_matrix=connectome_real*1.5, time=30, noise_sigma=3, use_tqdm=False)
    
    sim_envelope = np.abs(simulated)
    sim_envelope_warmed = sim_envelope[:, 2000:]
    
    sim_path = os.path.join(fpath)
    np.save(sim_path, simulated)

# init arrays

k_amp_values = np.zeros((k_values.shape[0], 100))
k_std_values = k_amp_values.copy()
k_dfa_values = k_amp_values.copy()
k_cplv_values = np.zeros((k_values.shape[0], 100, 100), dtype=np.complex)


# compute amplitude, cPLV, and DFA values

for i in range(k_values.shape[0]):    
    sim_path = os.path.join(f'simulations/sim_{i}_l_value.npy')
    
    simulated = np.load(sim_path)
    sim_envelope = np.abs(simulated)
    sim_envelope_warmed = sim_envelope[:, 2000:]
    
    simulated_normed = simulated / np.abs(simulated)
    
    k_amp_values[i] = sim_envelope_warmed.mean(axis=-1)
    k_std_values[i] = sim_envelope_warmed.std(axis=-1)
    k_dfa_values[i] = dfa(sim_envelope_warmed, window_lengths, use_gpu=True)[2]
    
    k_cplv_values[i] = cplv(simulated_normed)


#get means and standard deviations

k_amp_mean = k_amp_values.mean(axis=-1)
k_amp_std = k_amp_values.std(axis=-1)

k_dfa_mean = k_dfa_values.mean(axis=-1)
k_dfa_std = k_dfa_values.std(axis=-1)

k_dfa_5, k_dfa_95 = np.percentile(k_dfa_values, (5,95), axis=-1)
k_std_mean = k_std_values.mean(axis=-1)
k_std_std = k_std_values.std(axis=-1)

k_plv_mean = np.mean([np.abs(cplv_arr[np.triu_indices(100, 1)]) for cplv_arr in k_cplv_values], axis=-1)
k_plv_std = np.std([np.abs(cplv_arr[np.triu_indices(100, 1)]) for cplv_arr in k_cplv_values], axis=-1)
k_amp_mean_smooth = smooth(k_amp_mean, window_len=5, window='flat')
k_dfa_mean_smooth = smooth(k_dfa_mean, window_len=5, window='flat')
k_plv_mean_smooth = smooth(k_plv_mean, window_len=5, window='flat')


# plot connectivity and DFA

fig, ax = plt.subplots(figsize=(10,6))
ax_twin = ax.twinx()

lines = []
lines += ax.plot(k_values, k_plv_mean_smooth, label='Connectivity', color='green', lw=3)
lines += ax_twin.plot(k_values, k_dfa_mean_smooth, label='LRTCs', color='orange', lw=3)
ax.set_yticks(np.round(np.geomspace(0.05, 1, 5), 2))

ax.yaxis.set_minor_formatter(mticker.NullFormatter())
ax.yaxis.set_major_formatter(mticker.ScalarFormatter())
ax.minorticks_off()

plt.legend(lines, [l.get_label() for l in lines], fontsize=16, loc=2, frameon=False)

ax.set_xlabel('Control parameter [K]', fontsize=16)
ax.set_ylabel('Connectivity [PLV]', fontsize=16)
ax_twin.set_ylabel('LRTCs [DFA]', fontsize=16)

ax.tick_params(labelsize=14)
ax_twin.tick_params(labelsize=14)

ax.grid(False)
ax_twin.grid(False)

fig.savefig('kuramoto_l.svg', dpi=300)










