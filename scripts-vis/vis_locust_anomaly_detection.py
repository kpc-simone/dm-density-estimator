import sys,os

from scipy.stats import multivariate_normal
from sklearn.datasets import make_spd_matrix
import random
import math

import warnings
warnings.filterwarnings("ignore")

sys.path.insert(0,os.path.abspath(os.path.join(os.path.dirname("__file__"), '../dm-density-estimator')))
from sklearn.cluster import estimate_bandwidth
from sspod import SinglescaleSSPOD
import numpy as np
import cupy as cp

from sklearn.metrics import roc_curve,precision_recall_curve
from scipy.interpolate import interp1d
from random import shuffle

import matplotlib.pyplot as plt
from matplotlib import cm
import pandas as pd

# generate random multivariate normal distribution
def make_mvn_mixture(
        mus,
        cov_matrices,
        mode_props = [1.],
        generate_samples = True,
        n_samples = 1000,
        ):
        
    assert len( mus ) == len( mode_props )
    assert len( cov_matrices ) == len( mode_props )
    assert np.allclose( np.array( sum(mode_props) ), np.array([1.]) )

    rvs = [ multivariate_normal( mean = mu, cov = cov_matrix ) for mu, cov_matrix in zip(mus,cov_matrices) ]
    def pdf(xs):
        return np.sum([m * rvs[m_idx].pdf(xs) for m_idx, m in enumerate(mode_props)], axis=0)
    
    if generate_samples == True:
        all_mode_samples = [ rvs[m_idx].rvs( size = ( math.ceil(m*n_samples,) ) ).reshape(math.ceil(m*n_samples),-1) for m_idx, m in enumerate(mode_props) ]
        #for mode_samples in all_mode_samples:
        #    print(mode_samples.shape)
        samples = np.concatenate( all_mode_samples, axis=0)
        return samples, pdf
        
    else:
        return pdf

if __name__ == '__main__':
    pmax = 0.008

    seed = 42
    np.random.seed(seed)    

    n_samples_total = 1000
    n_features = 2
    n_modes = 3

    mode_props = np.random.uniform( low = 0.1, high = 1., size = (n_modes,) )
    mode_props /= mode_props.sum()

    contamination_level = 0.1
    n_inliers = int( (1. - contamination_level) * n_samples_total)

    mus = [ np.random.uniform( low = -4, high = 4, size = (n_features,) ) for m in range(n_modes) ]
    cov_matrices = [ make_spd_matrix(n_features, random_state = m*seed) for m in range(n_modes) ]

    inlier_xs, pdf = make_mvn_mixture(mus,cov_matrices,n_samples = n_inliers, mode_props = mode_props )
    
    n_outliers = n_samples_total - inlier_xs.shape[0]
    outlier_xs = np.random.uniform( low = np.tile( -6., n_features ), high = np.tile( 6., n_features ), size = (n_outliers,n_features) )
    labels_y = np.concatenate( [ np.zeros((inlier_xs.shape[0],1)), np.ones((outlier_xs.shape[0],1)) ], axis = 0 )
    data_xs = np.concatenate( [ inlier_xs, outlier_xs ], axis = 0 )

    # NETWORKS
    networks_dir = '../networks/locust'
    NETWORKS = [ os.path.join(networks_dir,f) for f in os.listdir(networks_dir) if 'npz' in f ]

    domain_ranges = np.array( [ (-6,6) for d in range(n_features) ] )
    
    cmap = cm.get_cmap('inferno')

    fig,(ax1,ax2,ax3) = plt.subplots( 1,3,figsize = (12.,4.) )
    for label,cval in zip([0,1],[0.1,0.5]):
        idxs = (labels_y == label).flatten()
        color = cmap( float(cval) )
        print(label,color)
        if label == 0:
            class_label = 'Inlier'
        else:
            class_label = 'Outlier'
        ax1.scatter(data_xs[idxs,0],data_xs[idxs,1],color=color,s=5,label=class_label)
    ax1.legend(loc='lower left',fancybox=False)
    ax1.set_xlabel(r'$x_1$')
    ax1.set_ylabel(r'$x_2$')
    ax1.set_xlim(-6,6)
    ax1.set_ylim(-6,6)
    
    roc_datas = {}
    pr_datas = {}
    for n,network in enumerate(NETWORKS):
        
        # SSPOD PDF
        # train network
        ssp_ls = np.zeros( (n_features,1) )
        for d in range( n_features ):
            ssp_ls[d,:] = estimate_bandwidth(data_xs[:,d].reshape(-1,1))        
        X_gpu = cp.array(data_xs)
        clf = SinglescaleSSPOD( network, norm = True )
        clf.fit(X_gpu,ssp_ls)
        decision_scores = clf.process_decision_scores()
        decision_scores = np.nan_to_num(decision_scores)

        # release memory -- CuPy
        mempool = cp.get_default_memory_pool()
        pinned_mempool = cp.get_default_pinned_memory_pool()
        mempool.free_all_blocks()
        pinned_mempool.free_all_blocks()
        
        fpr, tpr, thresholds = roc_curve(labels_y.flatten(), decision_scores, pos_label=1)
        roc_data = np.vstack([fpr,tpr,thresholds]).T
        roc_datas[n] = roc_data

        pre, rec, thresholds = precision_recall_curve( labels_y.flatten(), decision_scores )
        pr_data = np.vstack([rec[:-1],pre[:-1],thresholds]).T
        pr_datas[n] = pr_data

    fpr_common = np.linspace(0.,1.,100)
    tpr_rs = np.zeros((fpr_common.shape[0],len(NETWORKS)))
    for key,data in roc_datas.items():
        fpr = data[:,0]
        tpr = data[:,1]
        
        tpr_func = interp1d(fpr,tpr)
        tpr_rs[:,key] = tpr_func(fpr_common)

    q50 = np.quantile(tpr_rs,axis=1,q=0.5)
    q10 = np.quantile(tpr_rs,axis=1,q=0.1)
    q90 = np.quantile(tpr_rs,axis=1,q=0.9)

    ax2.plot(fpr_common,q50,color='dimgray')
    ax2.fill_between(fpr_common,q10,q90,color='dimgray',alpha=.2)
    ax2.plot([0,1],[0,1],color='k',linestyle='--')
    ax2.text(0.45,0.35, 'Chance level', rotation=45, rotation_mode='anchor')
    ax2.set_xlabel('False positive rate')
    ax2.set_ylabel('True positive rate')
    ax2.set_xlim(0.,1.05)
    ax2.set_ylim(0.,1.05)
    for spine in ['top','right']:
        ax2.spines[spine].set_visible(False)
        
    rec_common = np.linspace(0.,1.,100)
    pre_rs = np.zeros((rec_common.shape[0],len(NETWORKS)))
    for key,data in pr_datas.items():
        rec = data[:,0]
        pre = data[:,1]
        
        pre_func = interp1d(rec,pre,bounds_error=False)
        pre_rs[:,key] = pre_func(fpr_common)

    q50 = np.quantile(pre_rs,axis=1,q=0.5)
    q10 = np.quantile(pre_rs,axis=1,q=0.1)
    q90 = np.quantile(pre_rs,axis=1,q=0.9)

    print('fraction pos labels: ', labels_y.mean())
    no_skill = labels_y.mean()
    ax3.plot(rec_common,q50,color='dimgray')
    ax3.fill_between(rec_common,q10,q90,color='dimgray',alpha=.2)
    ax3.axhline(no_skill,color='k',linestyle='--')
    ax3.text(0.05,no_skill+0.02, 'Chance level')
    ax3.set_xlabel('Recall')
    ax3.set_ylabel('Precision')
    ax3.set_xlim(0.,1.05)
    ax3.set_ylim(0.,1.05)
    
    for ax in (ax2,ax3):
        for spine in ['top','right']:
            ax.spines[spine].set_visible(False)
    
    fig.tight_layout()
    plt.show()