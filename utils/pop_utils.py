import nengo
import cupy as cp
import numpy as np
import scipy
import os

import pickle

import matplotlib.pyplot as plt

def sparsity_to_x_intercept(d, p):
	sign = 1
	if p > 0.5:
		p = 1.0 - p
		sign = -1
	return sign * np.sqrt(1-scipy.special.betaincinv((d-1)/2.0, 0.5, 2*p))

def make_pop(n_neurons,ssp_dim,xi,encoders,neuron_type = nengo.RectifiedLinear()):
    
    model = nengo.Network()
    with model:
        
        if encoders is None:
            encoders = nengo.dists.UniformHypersphere(surface=True).sample(n_neurons, ssp_dim)
            
        model.ens = nengo.Ensemble(n_neurons = n_neurons, dimensions = ssp_dim,
                             encoders = encoders,
                             gain = np.ones(n_neurons),
                             bias = np.zeros(n_neurons) + xi,
                             neuron_type = neuron_type,
                             normalize_encoders = False,
                            )
                            
    # want to know activities of neurons for certain inputs
    sim = nengo.Simulator( model, progress_bar = False )
    return model,sim

def get_pop_props_cpu( model,sim ):
    scaled_encoders = np.array( sim.data[model.ens].scaled_encoders.T )
    bias = np.array( sim.data[model.ens].bias.reshape(1,-1) )
    return scaled_encoders,bias

def get_pop_props( model,sim ):
    scaled_encoders = cp.array( sim.data[model.ens].scaled_encoders.T )
    bias = cp.array( sim.data[model.ens].bias.reshape(1,-1) )
    return scaled_encoders,bias
    
def sample_bundle_encoders( domain_bounds, ssp_encoder, n_neurons, normalize = True ):

    # sample from the domain
    sample_xs = np.random.uniform( low = domain_bounds[0,:], high = domain_bounds[1,:], size = (1000, domain_bounds.shape[1]) )
    
    # project samples points to ssp_space
    sample_phis = ssp_encoder.encode( sample_xs )
    
    # for each cell, sample bundle size from discrete distribution
    es = [1, 2, 3, 4, 5, 6, 7]
    ps = [0.3, 0.25, 0.2, 0.1, 0.05, 0.05, 0.05]
    set_sizes = np.random.choice( es, n_neurons, p = ps ).tolist()

    encoders = np.zeros( (n_neurons,ssp_encoder.ssp_dim) )
    for i in range(n_neurons):
        set_size = set_sizes[i]
        idxs = np.random.randint( low = 0, high = sample_xs.shape[0], size = set_size )
        
        vs = sample_phis[idxs,:].get()

        e = np.mean( vs, axis = 0 )
        if normalize:
            e /= np.linalg.norm(e)
        
        encoders[i,:] = e

    return np.array(encoders)
    
def load_pop_props( pop_props_obj_filepath ):
    
    pop_props_obj = np.load( pop_props_obj_filepath )
    
    filename = os.path.basename(pop_props_obj_filepath)
    rho_specified = float( filename[ filename.find('rho') + 4: ].split('-')[0] )

    scaled_encoders = cp.array( pop_props_obj['scaled_encoders'] )
    bias = cp.array( pop_props_obj['bias'] )
    
    ssp_dim = pop_props_obj['ssp_dim'].astype(int)
    xi_final = pop_props_obj['xi_final'].astype(float)
    n_neurons = pop_props_obj['n_neurons'].astype(int)
    
    return scaled_encoders,bias,ssp_dim,xi_final,n_neurons,rho_specified

def load_pop_props_encoder( pop_props_obj_filepath ):
    
    pop_props_obj = pickle.load( file = open(pop_props_obj_filepath,'rb') )
    ssp_encoder = pop_props_obj['ssp_encoder']
    
    scaled_encoders = cp.array( pop_props_obj['scaled_encoders'] )
    bias = cp.array( pop_props_obj['bias'] )
    xi_final = pop_props_obj['xi_final'].astype(float)
    n_neurons = pop_props_obj['n_neurons']
    
    return scaled_encoders,bias,ssp_encoder,xi_final,n_neurons

def ReLU( scaled_encoders,bias,test_phis ):
    return cp.clip( cp.dot(test_phis, scaled_encoders) + bias, a_min = 0., a_max = None )