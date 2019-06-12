from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from matplotlib.ticker import PercentFormatter


# Settings
max_norm  = 16
results_dir = '../../exp_results/pn3-yolo'
max_query = 1000 # out of 5000
no_images = 1000 # out of 5000

# Clean dataset statistics
clean_stats = np.load(results_dir + '/clean_preds.npy')
names = clean_stats[0]
targets = clean_stats[1].astype(int)
clean_stats = clean_stats[2:].astype(float)

for name_ind, chosen in enumerate(names):
    clean_class = [clean_stats[0][name_ind], clean_stats[1][name_ind], clean_stats[2][name_ind], clean_stats[3][name_ind]]
    
    # Load and process data
    for noise_f in ['gba', 'per']:#, 'ran']:
        save_path = results_dir + '/%s_%iN%iQ' % (noise_f, no_images, max_norm)

        # Load results
        entries = 8
        if noise_f == 'ran': entries = 5
        res = np.empty((0, entries))

        for trial in np.arange(99, max_query, 100):
            temp = np.load(save_path + '%.3i-%.3i.npy' % (trial - 99, trial))
            for entry in temp:
                tp = np.concatenate([np.array([entry[0][0], entry[1][0], entry[2][0], entry[3][0]]), entry[4:].astype(np.float)])
                res = np.concatenate([res, tp.reshape((1, 8))])

        # Create dataframe
        results = pd.DataFrame(res)

        if noise_f == 'gba':
            results.columns = ['Precision', 'Recall', 'mAP', 'F1', 'sigma', 'theta', 'lambd', 'sides']
            results['sides'] = results['sides'].astype(int)
            gba_class = results

        if noise_f == 'per':
            results.columns = ['Precision', 'Recall', 'mAP', 'F1', 'freq_x', 'freq_y', 'freq_sine', 'octave']
            results['octave'] = results['octave'].astype(int)
            per_class = results

        if noise_f == 'ran': results.columns = ['mPrecision', 'mRecall', 'mAP', 'mF1', 'seed']
            #ran_class = results
        
    #xlim = (0.2, 1.0)
    #ylim = (0, 0.145)

    fig = plt.figure(figsize = (20, 4))
    plt.subplots_adjust(wspace = 0.2)
    plt.title(chosen, fontsize = 24)
    plt.axis('off')
    
    num_bins = 50
    ax = fig.add_subplot(1, 4, 1)
    a = gba_class['Precision']
    b = per_class['Precision']
    #c = ran_class['ev-T_in3']
    data = np.hstack((a,b))
    #data = np.hstack((a,b,c))
    bins=np.histogram(data, bins = num_bins)[1]

    ax.set_title('Precision', size = 18)
    ax.hist(a, bins = bins, weights = np.ones(len(a)) / len(a), color = 'C0', alpha = 0.66, label = 'Gabor')
    ax.hist(b, bins = bins, weights = np.ones(len(b)) / len(b), color = 'C1', alpha = 0.66, label = 'Perlin')
    #ax.hist(c, bins = bins, weights = np.ones(len(c)) / len(c), color = 'C2', alpha = 0.66, label = 'Random')
    ax.set_ylabel('Perturbation Frequency', size = 17, labelpad = 8)
    ax.yaxis.set_major_formatter(PercentFormatter(1, decimals = 0))
    ax.set_xlabel('Precision', size = 14, labelpad = 8)
    #ax.axvline(x = 0.1, color = 'C2', alpha = 0.5, label = 'Median random noise', linestyle = '--')
    ax.axvline(x = clean_class[0], color = 'grey', alpha = 0.5, label = 'Clean dataset', linestyle = '--')
    ax.tick_params(labelsize = 10)
    #ax.set_xlim(xlim)
    #ax.set_ylim(ylim)

    num_bins = 50
    ax = fig.add_subplot(1, 4, 2)
    a = gba_class['Recall']
    b = per_class['Recall']
    #c = ran_class['ev-T_in3']
    data = np.hstack((a,b))
    #data = np.hstack((a,b,c))
    bins=np.histogram(data, bins = num_bins)[1]

    ax.set_title('Recall', size = 18)
    ax.hist(a, bins = bins, weights = np.ones(len(a)) / len(a), color = 'C0', alpha = 0.66, label = 'Gabor')
    ax.hist(b, bins = bins, weights = np.ones(len(b)) / len(b), color = 'C1', alpha = 0.66, label = 'Perlin')
    #ax.hist(c, bins = bins, weights = np.ones(len(c)) / len(c), color = 'C2', alpha = 0.66, label = 'Random')
    ax.yaxis.set_major_formatter(PercentFormatter(1, decimals = 0))
    ax.set_xlabel('Recall', size = 14, labelpad = 8)
    #ax.axvline(x = 0.1, color = 'C2', alpha = 0.5, label = 'Median random noise', linestyle = '--')
    ax.axvline(x = clean_class[1], color = 'grey', alpha = 0.5, label = 'Clean dataset', linestyle = '--')
    ax.tick_params(labelsize = 10)
    #ax.set_xlim(xlim)
    #ax.set_ylim(ylim)
    
    num_bins = 40
    ax = fig.add_subplot(1, 4, 3)
    a = gba_class['mAP']
    b = per_class['mAP']
    #c = ran_class['ev-T_in3']
    data = np.hstack((a,b))
    #data = np.hstack((a,b,c))
    bins=np.histogram(data, bins = num_bins)[1]

    ax.set_title('mAP', size = 18)
    ax.hist(a, bins = bins, weights = np.ones(len(a)) / len(a), color = 'C0', alpha = 0.66, label = 'Gabor')
    ax.hist(b, bins = bins, weights = np.ones(len(b)) / len(b), color = 'C1', alpha = 0.66, label = 'Perlin')
    #ax.hist(c, bins = bins, weights = np.ones(len(c)) / len(c), color = 'C2', alpha = 0.66, label = 'Random')
    ax.yaxis.set_major_formatter(PercentFormatter(1, decimals = 0))
    ax.set_xlabel('mean Average Precision', size = 14, labelpad = 8)
    #ax.axvline(x = 0.1, color = 'C2', alpha = 0.5, label = 'Median random noise', linestyle = '--')
    ax.axvline(x = clean_class[2], color = 'grey', alpha = 0.5, label = 'Clean dataset', linestyle = '--')
    ax.tick_params(labelsize = 10)
    #ax.set_xlim(xlim)
    #ax.set_ylim(ylim)
    
    num_bins = 50
    ax = fig.add_subplot(1, 4, 4)
    a = gba_class['F1']
    b = per_class['F1']
    #c = ran_class['ev-T_in3']
    data = np.hstack((a,b))
    #data = np.hstack((a,b,c))
    bins=np.histogram(data, bins = num_bins)[1]

    ax.set_title('F1', size = 18)
    ax.hist(a, bins = bins, weights = np.ones(len(a)) / len(a), color = 'C0', alpha = 0.66, label = 'Gabor')
    ax.hist(b, bins = bins, weights = np.ones(len(b)) / len(b), color = 'C1', alpha = 0.66, label = 'Perlin')
    #ax.hist(c, bins = bins, weights = np.ones(len(c)) / len(c), color = 'C2', alpha = 0.66, label = 'Random')
    ax.yaxis.set_major_formatter(PercentFormatter(1, decimals = 0))
    ax.set_xlabel('F1 score', size = 14, labelpad = 8)
    #ax.axvline(x = 0.1, color = 'C2', alpha = 0.5, label = 'Median random noise', linestyle = '--')
    ax.axvline(x = clean_class[3], color = 'grey', alpha = 0.5, label = 'Clean dataset', linestyle = '--')
    ax.tick_params(labelsize = 10)
    #ax.set_xlim(xlim)
    #ax.set_ylim(ylim)

    plt.legend(fontsize = 16, bbox_to_anchor=(-0.4, 1.35), ncol = 3)
    plt.savefig('figures/%i_%s.png' % (name_ind, chosen), dpi = 300, bbox_inches = 'tight')
    plt.close()
    