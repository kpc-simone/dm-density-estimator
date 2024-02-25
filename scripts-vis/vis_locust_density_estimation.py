import sys,os

from scipy.stats import multivariate_normal
from sklearn.datasets import make_spd_matrix
import random
import math

import warnings
warnings.filterwarnings("ignore")

sys.path.insert(0,os.path.abspath(os.path.join(os.path.dirname("__file__"), '../dm-density-estimator')))
from sspod import SinglescaleSSPOD
import numpy as np
import cupy as cp

from sklearn.cluster import estimate_bandwidth
# from sklearn.metrics import roc_auc_score,average_precision_score
from random import shuffle

import matplotlib.pyplot as plt
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


from scipy.optimize import curve_fit
from scipy import stats
def line(x,m,b):
    return m*x + b

def fit_line(f1,f2,func=line,ax=None):
    xs = np.array(f1).flatten()
    ys = np.array(f2).flatten()
    
    popt,pcov = curve_fit(func,xs,ys)
    fs = func(xs,*popt)
    rs = ys - fs
    ss_res = np.sum(rs**2)
    ss_tot = np.sum( ( ys - np.mean(ys) )**2 )
    rsq = 1 - ss_res / ss_tot
    rval = stats.pearsonr(fs,ys)[0]

    if ax is not None:
        xmin = min(xs)
        xmax = max(xs)
        ymin = min(ys)
        ymax = max(ys)

        ts = np.linspace(xmin-0.1*(xmax-xmin),xmax+0.1*(xmax-xmin),len(xs))
        ax.plot(ts,func(ts,*popt),color='k',label='Fit')
        
        #m,b = popt

        ax.text(0.6,0.1,'$r$ = {:4.3f}'.format(rval),transform=ax.transAxes,ha='left',fontsize=10)
        # ax.text(0.1,0.10,'$r^2$ = {:4.3f}'.format(rsq),transform=ax.transAxes,ha='left',fontsize=10)
        # ax.text(0.1,0.05,'$m$ = {:4.3f}'.format(m),transform=ax.transAxes,ha='left',fontsize=10)
    return rval,rsq,popt


if __name__ == '__main__':
    pmax = 0.008

    seed = 42
    np.random.seed(seed)
    
    n_samples_total = 1000
    n_features = 2
    n_modes = 3

    mode_props = np.random.uniform( low = 0.1, high = 1., size = (n_modes,) )
    mode_props /= mode_props.sum()

    contamination_level = 0.
    n_inliers = int( (1. - contamination_level) * n_samples_total)

    mus = [ np.random.uniform( low = -4, high = 4, size = (n_features,) ) for m in range(n_modes) ]
    cov_matrices = [ make_spd_matrix(n_features, random_state = m*seed) for m in range(n_modes) ]

    inlier_xs, pdf = make_mvn_mixture(mus,cov_matrices,n_samples = n_inliers, mode_props = mode_props )

    n_outliers = n_samples_total - inlier_xs.shape[0]
    labels_y = np.zeros( (inlier_xs.shape[0],1) )
    data_xs = inlier_xs 

    # NETWORKS
    networks_dir = '../networks/locust'
    NETWORKS = [ os.path.join(networks_dir,f) for f in os.listdir(networks_dir) if 'npz' in f ]

    domain_ranges = np.array( [ (-6,6) for d in range(n_features) ] )
    print(domain_ranges)
    n_eval_points = 50
    meshes = np.meshgrid(*[np.linspace(b[0], b[1], n_eval_points) 
                        for b in domain_ranges])
    eval_xs = np.vstack([m.flatten() for m in meshes]).T
    #print(eval_xs.reshape((n_eval_points,n_eval_points)))

    # sample points
    # ax1.scatter( data_xs[:,0], data_xs[:,1], color='dimgray', s = 5 )
    # ax1.set_xlim( domain_ranges[0,:] )
    # ax1.set_ylim( domain_ranges[1,:] )

    # TRUE PDF
    pdf_eval_xs = pdf(eval_xs)
    pdf_eval_xs /= pdf_eval_xs.sum()

    for network in NETWORKS:
        print(os.path.basename(network))
        fig,axes = plt.subplots(1,5,figsize=(14.,3.5),squeeze=False,gridspec_kw={'width_ratios':[20,1,20,1,20]})
        ax1 = axes[0,0]
        cax1 = axes[0,1]
        
        ax2 = axes[0,2]
        cax2 = axes[0,3]
        
        ax3 = axes[0,4]
        
        im = ax1.imshow( pdf_eval_xs.reshape( (n_eval_points,n_eval_points) ), 
                        # norm = LogNorm(vmin = 0.001, vmax = 1.),
                        origin = 'lower', 
                        cmap = 'inferno', vmin = 0., vmax = pmax,
                        extent = domain_ranges.ravel(),
                        aspect = 'auto'
                        )
        ax1.set_title('Ground Truth')
        cbar = fig.colorbar(im, cax = cax1, orientation = 'vertical')
        cax1.set_title('P(x)')
        cbar.ax.tick_params( labelsize = 8 )
        cbar.ax.yaxis.set_ticks_position( 'right' )
                
        # SSPOD PDF
        # train network
        ssp_ls = np.zeros( (n_features,1) )
        for d in range( n_features ):
            ssp_ls[d,:] = estimate_bandwidth(data_xs[:,d].reshape(-1,1))
        X_gpu = cp.array(data_xs)
        clf = SinglescaleSSPOD( network, norm = True )
        clf.fit(X_gpu,ssp_ls)
        decision_scores = clf.get_scores()
        decision_scores = np.nan_to_num(decision_scores)

        # query the network
        X_qp = cp.array(eval_xs)
        clf.predict(X_qp)
        scores_qp = clf.get_scores()
        scores_qp = np.nan_to_num(scores_qp)
        scores_qp /= scores_qp.sum()
        
        #fig,ax = plt.subplots(1,1)
        #ax.hist(scores_qp,range=(-0.005,0.005),bins=21)
        #plt.show()

        # release memory -- CuPy
        mempool = cp.get_default_memory_pool()
        pinned_mempool = cp.get_default_pinned_memory_pool()
        mempool.free_all_blocks()
        pinned_mempool.free_all_blocks()

        im2 = ax2.imshow( scores_qp.reshape( (n_eval_points,n_eval_points) ), 
                        # norm = LogNorm(vmin = 0.001, vmax = 1.),
                        origin = 'lower', 
                        cmap = 'inferno', vmin = 0., vmax = pmax/2,
                        extent = domain_ranges.ravel(),
                        aspect = 'auto'
                        )
        ax2.set_title(r'Network Output')
        cbar = fig.colorbar(im2, cax = cax2, orientation = 'vertical')
        cax2.set_title('Output (a.u.)')
        cbar.ax.tick_params( labelsize = 8 )
        cbar.ax.yaxis.set_ticks_position( 'right' )

        for ax in (ax1,ax2):
            ax.set_xlabel(r'$x_1$')
            ax.set_ylabel(r'$x_2$')
        
        # linear correlation, spearman rank-
        ax3.scatter( pdf_eval_xs[::5], scores_qp[::5], color = 'dimgray', s = 5, label='Data' )
        fit_line(pdf_eval_xs, scores_qp,ax=ax3)
        
        ax3.set_xlim(0.,pmax)
        xticks = np.linspace(0.,pmax,5)
        ax3.set_xticks(xticks)
        
        ax3.set_ylim(0.,pmax/2)
        yticks = [t/2 for t in xticks]
        ax3.set_yticks(yticks)
        
        ax3.legend(loc='upper left',fancybox=False)
        ax3.set_xlabel('Probability Density')
        ax3.set_ylabel('Network Output')

        fig.tight_layout()
        plt.show()