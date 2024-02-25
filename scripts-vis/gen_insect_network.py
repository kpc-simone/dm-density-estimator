from tqdm import tqdm
import numpy as np
import pickle
import nengo
import sys,os

sys.path.insert(0,'../utils')
from pop_utils import make_pop,get_pop_props,sample_bundle_encoders,sparsity_to_x_intercept,ReLU
from sspspace_gpu import RandomSSPSpace
import cupy as cp

def gen_sspod_network(out_dir, ssp_encoder, p = 0.001, n_neurons = 50000,encoders_type = 'bundle', seed = 0):
    x_intercept = sparsity_to_x_intercept(d = ssp_encoder.ssp_dim, p = p)
    bias_ = -x_intercept

    np.random.seed( seed )
    
    if encoders_type == 'bundle':
        domain_bounds = np.tile( np.array([[-10],[10]]), domain_dim )
        encoders = sample_bundle_encoders( domain_bounds = domain_bounds, 
                                            ssp_encoder = ssp_encoder, 
                                            n_neurons = n_neurons
                                            )
    elif encoders_type == 'random':
        encoders = nengo.dists.UniformHypersphere(surface=True).sample(n_neurons, ssp_encoder.ssp_dim)
    
    model,sim = make_pop(n_neurons = n_neurons, 
                                ssp_dim = ssp_encoder.ssp_dim, 
                                xi = bias_,
                                encoders = encoders,
                                )
    scaled_encoders,bias = get_pop_props(model,sim)

    # check actual sparsity
    domain_bounds = np.tile( np.array([[-10],[10]]), domain_dim )
    sample_xs = cp.random.uniform( low = -10, high = 10, size = (1000,50) )
    sample_phis = ssp_encoder.encode( sample_xs )
    A = ReLU(scaled_encoders,bias,sample_phis)
    rho_actual = ( A > 0 ).mean()
    print(rho_actual)
    
    out_data = {
        'scaled_encoders'   : scaled_encoders,
        'bias'              : bias_,
        'xi_final'          : bias_,
        'n_neurons'         : n_neurons,
        'ssp_encoder'       : ssp_encoder,
        'rho-actual'        : rho_actual
    }
    
       
    if encoders_type == 'bundle':
        out_filename = '{}_encoders-seed_{}-d_{}-xi_{}-N_{}-rho_{}.pkl'.format(encoders_type,seed,ssp_encoder.ssp_dim,bias_,n_neurons,p)
        out_filepath = os.path.join(out_dir,out_filename) 
        pickle.dump( out_data, file = open(out_filepath, 'wb') )
        
    elif encoders_type == 'random':
        out_filename = '{}_encoders-seed_{}-d_{}-xi_{}-N_{}-rho_{}.npz'.format(encoders_type,seed,ssp_encoder.ssp_dim,bias_,n_neurons,p)
        out_filepath = os.path.join(out_dir,out_filename) 
        np.savez( file = out_filepath, allow_pickle = True, **out_data )

if __name__ == '__main__':
    args = sys.argv[1:]
    
    seed = 0
    if '--drosophila' in args:
        out_dir = '../networks/drosophila-bundle'
        if not os.path.exists(out_dir):
            os.mkdir(out_dir)

        ssp_dim_ = 1024
        domain_dim = 50
        ssp_encoder = RandomSSPSpace( domain_dim = domain_dim, ssp_dim = ssp_dim_, length_scale = np.sqrt(domain_dim) )
        encoders_type = 'bundle'
        n_neurons = 700
        rho_desired = 0.06
        
        gen_sspod_network(out_dir, 
                            ssp_encoder, 
                            p = rho_desired, 
                            n_neurons = n_neurons,
                            encoders_type = encoders_type, 
                            seed = seed
                            )

    # locust
    #seed = 42
