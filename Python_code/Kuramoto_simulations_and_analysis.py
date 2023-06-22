"""

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



import os
import json
import pickle

import numpy as np
import pandas as pd

import statsmodels.api as sm

import matplotlib.pyplot as plt
from matplotlib import gridspec

import tqdm as tqdm

from utils.pyutils import cplv, dfa

from sklearn.preprocessing import PolynomialFeatures


import sys
source_directory = ''             # insert path to code directory here
data_directory   = ''             # insert path to data directory here
sys.path.append(source_directory+'utils')

n_parcels = 400

sr = 200
central_frequency = 10.0

node_frequencies = np.array([central_frequency]*n_parcels)

window_lengths = np.geomspace(sr, sr*30, num=30)


resolution = 20
l_values = np.linspace(3, 9, resolution)
k_values = np.linspace(3, 9, resolution)


uniform_data_path   = data_directory + '/simulations_2d_uniform/'
realistic_data_path = data_directory + '/simulations_2d_non_log/'

config = json.load(open(os.path.join(uniform_data_path, 'config.json'), 'r'))


k_values = np.array(config['k'])
l_values = np.array(config['l'])

resolution = len(k_values)


individuals_uniform_data_path   = data_directory + '/simulations_individuals_uniform_rs/'
individuals_realistic_data_path = data_directory + '/simulations_individuals_non_log_subj_connectomes/'

n_subjects = 20
resolution = 25
k_values = np.linspace(3, 11, resolution)


if os.path.exists('uniform_individuals_stats.pickle'):
    uniform_data = pickle.load(open('uniform_individuals_stats.pickle', 'rb'))
    kl_amp_values_individuals_uniform = uniform_data['amplitude']
    kl_plv_values_individuals_uniform = uniform_data['plv'] 
    kl_dfa_values_individuals_uniform = uniform_data['dfa']
    kl_std_values_individuals_uniform = uniform_data['std']
    
else:
    kl_amp_values_individuals_uniform = np.zeros((resolution, n_subjects, n_parcels))
    kl_std_values_individuals_uniform = np.zeros((resolution, n_subjects, n_parcels))
    kl_dfa_values_individuals_uniform = np.zeros((resolution, n_subjects, n_parcels))
    kl_plv_values_individuals_uniform = np.zeros((resolution, n_subjects, n_parcels, n_parcels))

    bar = tqdm.tqdm(total=resolution*n_subjects)
    for i, k in enumerate(k_values):
        for j in range(n_subjects):
            fname = f'simulate_single_run_k_{i}_omega_seed_{j}.pickle'
            fpath = os.path.join(individuals_uniform_data_path, fname)

            if not(os.path.exists(fpath)):
                print(f'Passing {fname}...')
                continue

            simulated = pickle.load(open(fpath, 'rb'))
            simulated_crop = simulated['func_result'][..., 8000:]

            sim_envelope = np.abs(simulated_crop)

            kl_dfa_values_individuals_uniform[i,j] = dfa(sim_envelope, window_lengths, use_gpu=False)[2]
            kl_plv_values_individuals_uniform[i,j] = np.abs(cplv(simulated_crop))

            bar.update(1)
    uniform_data = {'amplitude': kl_amp_values_individuals_uniform, 'plv': kl_plv_values_individuals_uniform, 'dfa': kl_dfa_values_individuals_uniform, 'std': kl_std_values_individuals_uniform}
    pickle.dump(uniform_data, open('uniform_individuals_stats.pickle', 'wb'))


if os.path.exists('realistic_individuals_stats.pickle'):
    uniform_data = pickle.load(open('realistic_individuals_stats.pickle', 'rb'))
    kl_amp_values_individuals_realistic = uniform_data['amplitude']
    kl_plv_values_individuals_realistic = uniform_data['plv'] 
    kl_dfa_values_individuals_realistic = uniform_data['dfa']
    kl_std_values_individuals_realistic = uniform_data['std']
    
else:
    kl_amp_values_individuals_realistic = np.zeros((resolution, n_subjects, n_parcels))
    kl_std_values_individuals_realistic = np.zeros((resolution, n_subjects, n_parcels))
    kl_dfa_values_individuals_realistic = np.zeros((resolution, n_subjects, n_parcels))
    kl_plv_values_individuals_realistic = np.zeros((resolution, n_subjects, n_parcels, n_parcels))
    
    bar = tqdm.tqdm(total=resolution*n_subjects)
    for i, k in enumerate(k_values):    
        for j in range(n_subjects):
            fname = f'simulate_single_run_k_{i}_weight_matrix_{j}.pickle'
            fpath = os.path.join(individuals_realistic_data_path, fname)

            if not(os.path.exists(fpath)):
                print(f'Passing {fname}...')
                continue

            simulated = pickle.load(open(fpath, 'rb'))
            simulated_crop = simulated['func_result'][..., 8000:]

            sim_envelope = np.abs(simulated_crop)

            kl_amp_values_individuals_realistic[i,j] = sim_envelope.mean(axis=-1)
            kl_std_values_individuals_realistic[i,j] = sim_envelope.std(axis=-1)

            kl_dfa_values_individuals_realistic[i,j] = dfa(sim_envelope, window_lengths, use_gpu=False)[2]
            kl_plv_values_individuals_realistic[i,j] = np.abs(cplv(simulated_crop))

            bar.update(1)
    uniform_data = {'amplitude': kl_amp_values_individuals_realistic, 'plv': kl_plv_values_individuals_realistic, 'dfa': kl_dfa_values_individuals_realistic, 'std': kl_std_values_individuals_realistic}
    pickle.dump(uniform_data, open('realistic_individuals_stats.pickle', 'wb'))


plv_mean_realistic = kl_plv_values_individuals_realistic.mean(axis=(1,-1,-2))
dfa_mean_realistic = kl_dfa_values_individuals_realistic.mean(axis=(1,-1))

plv_mean_realistic = np.interp(k_values, np.linspace(3,11, 20), plv_mean_realistic)
dfa_mean_realistic = np.interp(k_values, np.linspace(3,11, 20), dfa_mean_realistic)

plv_mean_uniform = kl_plv_values_individuals_uniform.mean(axis=(1,-1,-2))
dfa_mean_uniform = kl_dfa_values_individuals_uniform.mean(axis=(1,-1))

plv_std_realistic = kl_plv_values_individuals_realistic.mean(axis=(-1,-2)).std(axis=1)
dfa_std_realistic = kl_dfa_values_individuals_realistic.mean(axis=-1).std(axis=1)

plv_std_realistic = np.interp(k_values, np.linspace(3,11, 20), plv_std_realistic)
dfa_std_realistic = np.interp(k_values, np.linspace(3,11, 20), dfa_std_realistic)

plv_std_uniform = kl_plv_values_individuals_uniform.mean(axis=(-1,-2)).std(axis=1)
dfa_std_uniform = kl_dfa_values_individuals_uniform.mean(axis=-1).std(axis=1)

k_crit_realistic = k_values / k_values[dfa_mean_realistic.argmax()]
k_crit_uniform = k_values / k_values[dfa_mean_uniform.argmax()]


fig, axes = plt.subplots(figsize=(5,10), nrows=4, gridspec_kw={'height_ratios': [2,1,2,1]})

# std_twinx = [axes[i].twinx() for i in [1,3]]

axes[0].plot(k_crit_realistic, plv_mean_realistic, color='blue')
axes[0].plot(k_crit_uniform, plv_mean_uniform, color='blue', ls='--')

axes[1].plot(k_crit_realistic, plv_std_realistic, color='blue')
axes[1].plot(k_crit_uniform, plv_std_uniform, ls='--', color='blue')

axes[2].plot(k_crit_realistic, dfa_mean_realistic, color='blue')
axes[2].plot(k_crit_uniform, dfa_mean_uniform, color='blue', ls='--')

axes[3].plot(k_crit_realistic, dfa_std_realistic, color='blue')
axes[3].plot(k_crit_uniform, dfa_std_uniform, ls='--', color='blue')

for ax, ylabel in zip(axes.tolist(), ['PLV', 'std', 'DFA', 'std']):
    # ax.set_xlim(k_values[[0, ~0]])
    ax.set_ylabel(ylabel)
    ax.set_xlim([0.6, 1.4])


for ax in axes:
    ax.set_xlabel('Local control parameter [K]')

fig.tight_layout()

fig.savefig('dfa_sync_model_spectra.svg', dpi=300)


k_values_dense = np.linspace(2.5, 10, 100)

kl_dfa_values_individuals = np.zeros((100, 10, n_parcels))
kl_plv_values_individuals = np.zeros((100, 10, n_parcels, n_parcels))

if os.path.exists('realistic_individuals_dense_stats.pickle'):
    dense_data = pickle.load(open('realistic_individuals_dense_stats.pickle', 'rb'))
    kl_plv_values_individuals = dense_data['plv'] 
    kl_dfa_values_individuals = dense_data['dfa']
else:
    bar = tqdm.tqdm(total=1000)
    for i, k in enumerate(k_values_dense):
        for jidx, j in enumerate(list(range(5)) + list(range(10,15))):
            fname = f'simulate_single_run_k_{i}_weight_matrix_{j}.pickle'
            fpath = os.path.join('/m/nbe/scratch/digital-twin/kuramoto_basic_sims/simulations_individuals_non_log_very_dense', fname)

            if not(os.path.exists(fpath)):
                print(f'Passing {fname}...')
                continue

            simulated = pickle.load(open(fpath, 'rb'))
            simulated_crop = simulated['func_result'][..., 8000:]

            sim_envelope = np.abs(simulated_crop)

            kl_dfa_values_individuals[i, jidx] = dfa(sim_envelope, window_lengths, use_gpu=False)[2]
            kl_plv_values_individuals[i, jidx] = np.abs(cplv(simulated_crop))

            bar.update(1)


dfa_mean = kl_dfa_values_individuals.mean(axis=(-1,-2))
dfa_std = kl_dfa_values_individuals.mean(axis=-1).std(axis=-1)

plv_mean = kl_plv_values_individuals.mean(axis=(-1,-2,-3))
plv_std = kl_plv_values_individuals.mean(axis=(-1,-2)).std(axis=-1)


k_values_dense = np.linspace(2.5, 10, 100)

if os.path.exists('realistic_individuals_dense_noisy_stats.pickle'):
    dense_data = pickle.load(open('realistic_individuals_dense_noisy_stats.pickle', 'rb'))
    kl_dfa_values_noisy = dense_data['dfa'] 
    kl_plv_values_noisy = dense_data['plv']
else:
    kl_dfa_values_noisy = np.zeros((100, 10, n_parcels))
    kl_plv_values_noisy = np.zeros((100, 10, n_parcels, n_parcels))

    bar = tqdm.tqdm(total=1000)
    for i, k in enumerate(k_values_dense):
        for jidx, j in enumerate(list(range(5)) + list(range(10,15))):
            fname = f'simulate_single_run_k_{i}_weight_matrix_{j}.pickle'
            fpath = os.path.join('/m/nbe/scratch/digital-twin/kuramoto_basic_sims/simulations_individuals_non_log_very_dense', fname)

            if not(os.path.exists(fpath)):
                print(f'Passing {fname}...')
                continue

            simulated = pickle.load(open(fpath, 'rb'))
            simulated_crop = simulated['func_result'][..., 8000:].copy()
            simulated_crop += np.random.normal(size=simulated_crop.shape, scale=0.1)*1j + np.random.normal(size=simulated_crop.shape, scale=0.1)

            sim_envelope = np.abs(simulated_crop)

            kl_dfa_values_noisy[i, jidx] = dfa(sim_envelope, window_lengths, use_gpu=False)[2]
            kl_plv_values_noisy[i, jidx] = np.abs(cplv(simulated_crop))

            bar.update(1)
    dense_data = {'plv': kl_plv_values_noisy, 'dfa': kl_dfa_values_noisy}
    pickle.dump( dense_data, open('realistic_individuals_dense_noisy_stats.pickle', 'wb'))


k_values_dense = np.linspace(2.5, 10, 100)

kl_dfa_values_uniform_noisy = np.zeros((100, 10, n_parcels))
kl_plv_values_uniform_noisy = np.zeros((100, 10, n_parcels, n_parcels))

if os.path.exists('realistic_individuals_dense_uniform_noisy_stats.pickle'):
    dense_data = pickle.load(open('realistic_individuals_dense_uniform_noisy_stats.pickle', 'rb'))
    kl_dfa_values_uniform_noisy = dense_data['dfa'] 
    kl_plv_values_uniform_noisy = dense_data['plv']
else:
    bar = tqdm.tqdm(total=1000)
    for i, k in enumerate(k_values_dense):
        for jidx, j in enumerate(range(10)):
            fname = f'simulate_single_run_k_{i}_weight_matrix_{j}.pickle'
            fpath = os.path.join('/m/nbe/scratch/digital-twin/kuramoto_basic_sims/simulations_individuals_uniform_very_dense', fname)

            if not(os.path.exists(fpath)):
                print(f'Passing {fname}...')
                continue

            simulated = pickle.load(open(fpath, 'rb'))
            simulated_crop = simulated['func_result'][..., 8000:].copy()
            simulated_crop += np.random.normal(size=simulated_crop.shape, scale=0.1)*1j + np.random.normal(size=simulated_crop.shape, scale=0.1)

            sim_envelope = np.abs(simulated_crop)

            kl_dfa_values_uniform_noisy[i, jidx] = dfa(sim_envelope, window_lengths, use_gpu=False)[2]
            kl_plv_values_uniform_noisy[i, jidx] = np.abs(cplv(simulated_crop))

            bar.update(1)
    dense_data = {'plv': kl_plv_values_uniform_noisy, 'dfa': kl_dfa_values_uniform_noisy}
    pickle.dump( dense_data, open('realistic_individuals_dense_uniform_noisy_stats.pickle', 'wb'))





dfa_mean = kl_dfa_values_noisy.mean(axis=(-1,-2))
dfa_std = kl_dfa_values_noisy.mean(axis=-1).std(axis=-1)
dfa_std_node = kl_dfa_values_noisy.std(axis=-1).mean(axis=-1)

plv_mean = kl_plv_values_noisy.mean(axis=(-1,-2,-3))
plv_std = kl_plv_values_noisy.mean(axis=(-1,-2)).std(axis=-1)
plv_std_node = kl_plv_values_noisy.std(axis=(-1,-2)).mean(axis=-1)


dfa_uniform_mean = kl_dfa_values_uniform_noisy.mean(axis=(-1,-2))
dfa_uniform_std = kl_dfa_values_uniform_noisy.mean(axis=-1).std(axis=-1)
dfa_uniform_std_node = kl_dfa_values_uniform_noisy.std(axis=-1).mean(axis=-1)

plv_uniform_mean = kl_plv_values_uniform_noisy.mean(axis=(-1,-2,-3))
plv_uniform_std = kl_plv_values_uniform_noisy.mean(axis=(-1,-2)).std(axis=-1)
plv_uniform_std_node = kl_plv_values_uniform_noisy.std(axis=(-1,-2)).mean(axis=-1)


k_crit_realistic = k_values_dense/k_values_dense[dfa_mean.argmax()]
k_crit_uniform = k_values_dense/k_values_dense[dfa_uniform_mean.argmax()]



# fig, axes = plt.subplots(figsize=(10,10), ncols=2, nrows=3)
fig = plt.figure(figsize=(7,7))
gs = gridspec.GridSpec(2, 3, height_ratios=[1,2])

axes = np.zeros((2,3), dtype=object)

axes[0,0] = fig.add_subplot(gs[0, :])

for idx in range(3):
    axes[1, idx] = fig.add_subplot(gs[1, idx])

ax_twin = axes[0,0].twinx()

lines = list()

lines += ax_twin.plot(k_crit_realistic, plv_mean, label='PLV', color='blue')
ax_twin.fill_between(k_crit_realistic, plv_mean - plv_std, plv_mean + plv_std, color='blue', alpha=0.2)

lines += axes[0,0].plot(k_crit_realistic, dfa_mean, color='orange', label='DFA')
axes[0,0].fill_between(k_crit_realistic, dfa_mean - dfa_std, dfa_mean + dfa_std, color='orange', alpha=0.2)

axes[0,0].legend(handles=lines, frameon=False)

axes[0,0].set_xlabel('K')
axes[0,0].set_ylabel('DFA')
ax_twin.set_ylabel('GS')

for regime, (start, ws, ax, xlim) in enumerate(zip([40,45,60], [10,20,10], axes[1], [(0.04, 0.08), [0.0, 1.0], [0.85, 1.0]])):
    end = start + ws
    start_k = k_crit_realistic[start]
    end_k = k_crit_realistic[start + ws]

    y_dfa_start = dfa_mean[start:start + ws].min()
    y_dfa_end = dfa_mean[start:start + ws].max()

    rect = plt.Rectangle([start_k, y_dfa_start], end_k - start_k, y_dfa_end - y_dfa_start, facecolor='none', edgecolor='blue')
    axes[0,0].add_patch(rect)

    x, y = kl_plv_values_noisy[start:end].mean(axis=(-2,-1)).flatten(), kl_dfa_values_noisy[start:end].mean(axis=-1).flatten()

    idx_sorted = np.argsort(x)
    x = x[idx_sorted]
    y = y[idx_sorted]

    x_lin = PolynomialFeatures(degree=1).fit_transform(x.reshape(-1,1))

    mdl_lin = sm.OLS(y, x_lin).fit()
    ax.plot(x, mdl_lin.predict(x_lin), color='orange')

    if regime == 1:
        x_sq = PolynomialFeatures(degree=2).fit_transform(x.reshape(-1,1))

        mdl_sq = sm.OLS(y, x_sq).fit()
        ax.plot(x, mdl_sq.predict(x_sq), color='red')

    ax.scatter(x, y, s=25)

for ax in axes[1]:
    ax.set_xlabel('GS')
    ax.set_ylim([0.6, 0.8])
    
axes[1,0].set_ylabel('DFA')

for ax in axes[1, 1:]:
    ax.set_yticks([])

# fig.tight_layout()

fig.savefig('dfa_sync_model_scatter.svg', dpi=300)


for regime, start, ws, xlim in zip(['subcritical', 'critical', 'supercritical'], [40,45,60], [10,20,10], [(0.04, 0.08), [0.0, 1.0], [0.85, 1.0]]):
    end = start + ws
    start_k = k_crit_realistic[start]
    end_k = k_crit_realistic[start + ws]

    k_regime = k_crit_realistic[start:start+ws]*k_values_dense[dfa_mean.argmax()]

    x, y = kl_plv_values_noisy[start:end].mean(axis=(-2,-1)).flatten(), kl_dfa_values_noisy[start:end].mean(axis=-1).flatten()

    df_dict = {'GS': kl_plv_values_noisy[start:end].mean(axis=(-2,-1)).flatten(),
                                      'DFA': kl_dfa_values_noisy[start:end].mean(axis=-1).flatten(), 
                                      'K': np.tile(k_crit_realistic[start:start+ws], (10,1)).T.flatten(),
                                      'Subject': list(range(1, ws+1))*10}
    df_regime_scatter = pd.DataFrame(df_dict)
    
    df_regime_scatter.to_csv(regime + '_scatter.csv', index=False)

df_plv_dfa_realistic = pd.DataFrame({'K': k_crit_realistic, 'PLV': plv_mean, 'DFA': dfa_mean})
df_plv_dfa_realistic.to_csv('gs_dfa_realistic.csv', index=False)

df_plv_dfa_uniform = pd.DataFrame({'K': k_crit_uniform, 'PLV': plv_uniform_mean, 'DFA': dfa_uniform_mean})
df_plv_dfa_uniform.to_csv('gs_dfa_uniform.csv', index=False)


fig, axes = plt.subplots(figsize=(10,10), nrows=2)

axes[0].plot(k_crit_realistic, plv_mean, color='orange', label='Realistic')
axes[0].fill_between(k_crit_realistic, plv_mean - plv_std, plv_mean + plv_std, color='orange', alpha=0.2)

axes[0].plot(k_crit_uniform, plv_uniform_mean, color='blue', label='Uniform')
axes[0].fill_between(k_crit_uniform, plv_uniform_mean - plv_uniform_std, plv_uniform_mean + plv_uniform_std, color='blue', alpha=0.2)

axes[0].legend(frameon=False)

axes[1].plot(k_crit_realistic, dfa_mean, color='orange', label='DFA')
axes[1].fill_between(k_crit_realistic, dfa_mean - dfa_std, dfa_mean + dfa_std, color='orange', alpha=0.2)

axes[1].plot(k_crit_uniform, dfa_uniform_mean, color='blue', label='DFA')
axes[1].fill_between(k_crit_uniform, dfa_uniform_mean - dfa_uniform_std, dfa_uniform_mean + dfa_uniform_std, color='blue', alpha=0.2)


for ax, ylabel in zip(axes, ['PLV', 'DFA']):
    ax.set_xlim([0.6, 1.3])
    ax.set_xlabel('K')
    ax.set_ylabel(ylabel)

# fig.tight_layout()

# fig.savefig('dfa_sync_model_scatter.svg', dpi=300)


mask_real = (k_crit_realistic >= 0.7) & (k_crit_realistic <= 1.3)
mask_uniform = (k_crit_uniform >= 0.7) & (k_crit_uniform <= 1.3)


fig, axes = plt.subplots(figsize=(10,10), nrows=4, gridspec_kw={'height_ratios': [2,1,2,1]})

axes[0].plot(k_crit_realistic[mask_real], plv_mean[mask_real], color='orange', label='Realistic')
axes[0].fill_between(k_crit_realistic[mask_real], plv_mean[mask_real] - plv_std[mask_real], plv_mean[mask_real] + plv_std[mask_real], color='orange', alpha=0.2)

axes[0].plot(k_crit_uniform[mask_uniform], plv_uniform_mean[mask_uniform], color='blue', label='Uniform')
axes[0].fill_between(k_crit_uniform[mask_uniform], plv_uniform_mean[mask_uniform] - plv_uniform_std[mask_uniform], plv_uniform_mean[mask_uniform] + plv_uniform_std[mask_uniform], color='blue', alpha=0.2)

axes[0].legend(frameon=False)

axes[1].plot(k_crit_realistic[mask_real], plv_std[mask_real], color='orange')
axes[1].plot(k_crit_uniform[mask_uniform], plv_uniform_std[mask_uniform], color='blue')

axes[2].plot(k_crit_realistic[mask_real], dfa_mean[mask_real], color='orange', label='DFA')
axes[2].fill_between(k_crit_realistic[mask_real], dfa_mean[mask_real] - dfa_std[mask_real], dfa_mean[mask_real] + dfa_std[mask_real], color='orange', alpha=0.2)

axes[2].plot(k_crit_uniform[mask_uniform], dfa_uniform_mean[mask_uniform], color='blue', label='DFA')
axes[2].fill_between(k_crit_uniform[mask_uniform], dfa_uniform_mean[mask_uniform] - dfa_uniform_std[mask_uniform], dfa_uniform_mean[mask_uniform] + dfa_uniform_std[mask_uniform], color='blue', alpha=0.2)

axes[3].plot(k_crit_realistic[mask_real], dfa_std[mask_real], color='orange')
axes[3].plot(k_crit_uniform[mask_uniform], dfa_uniform_std[mask_uniform], color='blue')

for ax, ylabel in zip(axes, ['PLV', 'std', 'DFA', 'std']):
    ax.set_xlim([0.7, 1.3])
    ax.set_xlabel('K')
    ax.set_ylabel(ylabel)

fig.tight_layout()

fig.savefig('dfa_sync_model_spectra.svg', dpi=300)





