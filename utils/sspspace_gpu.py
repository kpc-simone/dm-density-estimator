import cupy as cp

def make_unitary_matrix_fourier( ssp_dim, domain_dim, eps=1e-3, rng = cp.random):
    a = rng.rand( (ssp_dim - 1)//2, domain_dim )
    sign = rng.choice((-1, +1), size=cp.shape(a) )
    phi = sign * cp.pi * (eps + a * (1 - 2 * eps))

    fv = cp.zeros( (ssp_dim,domain_dim), dtype='complex64')
    fv[0,:] = 1

    fv[1:(ssp_dim + 1) // 2,:] = phi
    fv[-1:ssp_dim // 2:-1,:] = -fv[1:(ssp_dim + 1) // 2,:]
    
    if ssp_dim % 2 == 0:
        fv[ssp_dim // 2,:] = 1

    return fv

class SSPEncoder:
    def __init__(self, phase_matrix, length_scale):
        '''
        Represents a domain using spatial semantic pointers.

        Parameters:
        -----------

        phase_matrix : cp.ndarray
            A ssp_dim x domain_dim ndarray representing the frequency 
            components of the SSP representation.

        length_scale : float or cp.ndarray
            Scales values before encoding.
        '''
        self.phase_matrix = phase_matrix
        self.domain_dim = self.phase_matrix.shape[1]
        self.ssp_dim = self.phase_matrix.shape[0]
        self.update_lengthscale(length_scale)

    def update_lengthscale(self, scale):
        '''
        Changes the lengthscale being used in the encoding.
        '''
        if not isinstance(scale, cp.ndarray) or scale.size == 1:
            self.length_scale = scale * cp.ones((self.domain_dim,))
        else:
            assert scale.size == self.domain_dim
            self.length_scale = scale
        assert self.length_scale.size == self.domain_dim
    
    def encode(self,x):
        '''
        Transforms input data into an SSP representation.

        Parameters:
        -----------
        x : cp.ndarray
            A (num_samples, domain_dim) array representing data to be encoded.

        Returns:
        --------
        data : cp.ndarray
            A (num_samples, ssp_dim) array of the ssp representation of the data
            
        '''
        
        x = cp.atleast_2d(x)
        ls_mat = cp.atleast_2d(cp.diag(1/self.length_scale.flatten()))
        
        assert ls_mat.shape == (self.domain_dim, self.domain_dim), f'Expected Len Scale mat with dimensions {(self.domain_dim, self.domain_dim)}, got {ls_mat.shape}'
        scaled_x = x @ ls_mat
        data = cp.fft.ifft( cp.exp( 1.j * self.phase_matrix @ scaled_x.T), axis=0 ).real
        
        return data.T   
    
def RandomSSPSpace(domain_dim, ssp_dim, length_scale = None, 
                   rng = cp.random.default_rng() ):
    
    phase_matrix = make_unitary_matrix_fourier(ssp_dim,domain_dim)

    length_scale = cp.array( length_scale )
    return SSPEncoder(phase_matrix, length_scale=length_scale)