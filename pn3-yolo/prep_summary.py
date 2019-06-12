from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import pandas as pd

# Settings
max_norm  = 16
results_dir = '../../exp_results/pn3-yolo'
max_query = 1000 # out of 5000
no_images = 1000 # out of 5000

for noise_f in ['gba', 'per', 'ran']:
    save_path = results_dir + '/%s_%iN%iQ' % (noise_f, no_images, max_norm)
    
    # Load results
    entries = 8
    if noise_f == 'ran': entries = 5
    res = np.empty((0, entries))
    
    for trial in np.arange(99, max_query, 100):
        temp = np.load(save_path + '%.3i-%.3i.npy' % (trial - 99, trial))
        for entry in temp:
            tp = np.concatenate([np.array([entry[0].mean(), entry[1].mean(), entry[2].mean(), entry[3].mean()]), entry[4:].astype(np.float)])
            res = np.concatenate([res, tp.reshape((1, entries))])

    # Create dataframe
    results = pd.DataFrame(res)

    if noise_f == 'gba':
        results.columns = ['mPrecision', 'mRecall', 'mAP', 'mF1', 'sigma', 'theta', 'lambd', 'sides']
        results['sides'] = results['sides'].astype(int)
    if noise_f == 'per':
        results.columns = ['mPrecision', 'mRecall', 'mAP', 'mF1', 'freq_x', 'freq_y', 'freq_sine', 'octave']
        results['octave'] = results['octave'].astype(int)
    if noise_f == 'ran': results.columns = ['mPrecision', 'mRecall', 'mAP', 'mF1', 'seed']

    # Save file
    results.to_csv('data_summary/summary-%s_%iN%iQ%i.csv' % (noise_f, no_images, max_norm, max_query))
    