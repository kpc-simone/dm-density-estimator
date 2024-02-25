import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import sys,os

sys.path.insert(0,'../utils')
from sspspace_gpu import RandomSSPSpace
from pop_utils import *

import cupy as cp

np.random.seed(0)

# load KC calcium imaging data
spdf = pd.read_csv('../srinivasan-2023/tseries_sigpvals.csv')
ecdf = pd.read_csv('../srinivasan-2023/tseries_head.csv')
spdf = pd.concat( [ecdf.loc[:,'odor'],spdf],axis=1)
spdf = spdf.groupby('odor').mean()
kc_data = spdf.to_numpy()
kc_data = np.where(kc_data < 0.05, 1., 0.)
shuffle_xs = np.random.permutation( range(kc_data.shape[0]))
shuffle_ys = np.random.permutation( range(kc_data.shape[1]))
kc_data = kc_data[shuffle_xs,:]
kc_data = kc_data[:,shuffle_ys]
kc_plot_data = kc_data[:,::2]

# load network
pop_filepath = '../networks/drosophila-bundle/bundle_encoders-seed_0-d_1024-xi_-0.04859357149201574-N_700-rho_0.06.pkl'
scaled_encoders,bias,ssp_encoder,xi_final,n_neurons = load_pop_props_encoder( pop_filepath )
print('ssp_encoder object in main script: ', ssp_encoder)

# generate test samples corresponding to olfactory stimuli
n_samples = 7
domain_bounds = np.tile( np.array([[-10],[10]]), ssp_encoder.domain_dim )
test_xs = np.random.uniform( low = domain_bounds[0,:], high = domain_bounds[1,:], size = (n_samples, domain_bounds.shape[1]) ) 
test_phis = ssp_encoder.encode( test_xs )

# query simulated KCs on test samples
A = ReLU(scaled_encoders,bias,test_phis)
A = cp.where(A>0,1.,0.).get()
kc_plot_model = A[:,:62]

fig,(ax1,ax2) = plt.subplots(1,2,figsize=(2.5,5.))
ax1.set_title(r'Odors $\to$',fontsize=10)
ax1.set_ylabel(r'Cells $\to$')
ax1.set_xlabel('Data')
ax2.set_xlabel('Model')
ax1.imshow(kc_plot_data.T,
        cmap = 'Greys_r',
        aspect = 'auto',
        interpolation = 'none'
        )        
ax2.imshow( kc_plot_model.T,
            cmap = 'Greys_r',
            aspect = 'auto',
            interpolation = 'none' 
            )        
for ax in (ax1,ax2):
    ax.set_xticks([])
    ax.set_yticks([])
fig.tight_layout()        
plt.show()