import cupy as cp
import sys,os
import time 

sys.path.append(os.path.join(os.path.dirname(__file__),'../utils'))
#from encoders_gpu import RandomSSPSpace
from sspspace_gpu import RandomSSPSpace
from pop_utils import *

class SinglescaleSSPOD(object):
    def __init__(self,pop_props_obj_filepath,norm = False):
        # load network here
        scaled_encoders,bias,ssp_dim,xi_final,n_neurons,rho_specified = load_pop_props( pop_props_obj_filepath )

        self.n_neurons = n_neurons
        self.rho_specified = rho_specified
        self.scaled_encoders = scaled_encoders
        self.bias = bias
        self.norm = norm
    
    def fit(self,X,length_scale):
        
        domain_dim = X.shape[1]
        self.ssp_encoder = RandomSSPSpace( domain_dim = domain_dim, ssp_dim = 1024, length_scale = length_scale )

        phis = self.ssp_encoder.encode(X)

        A = ReLU(self.scaled_encoders,self.bias,phis)
        self.rho_actual = ( A > 0 ).mean()
        
        self.w = cp.mean( A, axis = 0 )
        self.scores = A @ self.w
        
        return self

    
    def predict(self,X):
        
        phis = self.ssp_encoder.encode(X)
        A = ReLU(self.scaled_encoders,self.bias,phis)
        self.scores = A @ self.w
        
        return self

    def get_scores(self):
        return self.scores.get()
        
    def get_labels(self, contamination = 0.1, method = 'log'):

        # TODO: load these curve fitting params from a file and pass at build time
        if method == 'log':
            a = 1.48113569
            p = 0.3187553
            b = -3.58892504
            self.threshold = 10**( a * cp.power(contamination,p) + b )
            self.labels = self.scores < self.threshold
            return self.threshold,self.labels
        elif method == 'quantile-cpu':
            scores_cpu = self.scores.get()
            self.threshold = np.quantile(scores_cpu, q = contamination )
            self.labels = scores_cpu < self.threshold
            return self.threshold,self.labels

    def process_decision_scores(self):
        ys = 1. - self.get_scores()
        return ys