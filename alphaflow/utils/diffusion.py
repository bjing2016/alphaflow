import torch
import numpy as np

#https://github.com/scipy/scipy/blob/main/scipy/spatial/transform/_rotation.pyx
def rmsdalign(a, b, weights=None): # alignes B to A  # [*, N, 3]
    B = a.shape[:-2]
    N = a.shape[-2]
    if weights == None:
        weights = a.new_ones(*B, N)
    weights = weights.unsqueeze(-1)
    a_mean = (a * weights).sum(-2, keepdims=True) / weights.sum(-2, keepdims=True)
    a = a - a_mean
    b_mean = (b * weights).sum(-2, keepdims=True) / weights.sum(-2, keepdims=True)
    b = b - b_mean
    B = torch.einsum('...ji,...jk->...ik', weights * a, b)
    u, s, vh = torch.linalg.svd(B)

    # Correct improper rotation if necessary (as in Kabsch algorithm)
    '''
    if torch.linalg.det(u @ vh) < 0:
        s[-1] = -s[-1]
        u[:, -1] = -u[:, -1]
    '''
    sgn = torch.sign(torch.linalg.det(u @ vh))
    s[...,-1] *= sgn
    u[...,:,-1] *= sgn.unsqueeze(-1)
    C = u @ vh # c rotates B to A
    return b @ C.mT + a_mean
    
def kabsch_rmsd(a, b, weights=None):
    B = a.shape[:-2]
    N = a.shape[-2]
    if weights == None:
        weights = a.new_ones(*B, N)
    b_aligned = rmsdalign(a, b, weights)
    out = torch.square(b_aligned - a).sum(-1)
    out = (out * weights).sum(-1) / weights.sum(-1)
    return torch.sqrt(out)

class HarmonicPrior:
    def __init__(self, N = 256, a =3/(3.8**2)):
        J = torch.zeros(N, N)
        for i, j in zip(np.arange(N-1), np.arange(1, N)):
            J[i,i] += a
            J[j,j] += a
            J[i,j] = J[j,i] = -a
        D, P = torch.linalg.eigh(J)
        D_inv = 1/D
        D_inv[0] = 0
        self.P, self.D_inv = P, D_inv
        self.N = N

    def to(self, device):
        self.P = self.P.to(device)
        self.D_inv = self.D_inv.to(device)
        
    def sample(self, batch_dims=()):
        return self.P @ (torch.sqrt(self.D_inv)[:,None] * torch.randn(*batch_dims, self.N, 3, device=self.P.device))
    
    
'''
def transition_matrix(N_bins=1000, X_max=5):
    bins = torch.linspace(0, X_max, N_bins+1, dtype=torch.float64)
    cbins = (bins[1:] + bins[:-1]) / 2
    bw = cbins[1] - cbins[0]
    mu = 2 / cbins - cbins
    idx = torch.arange(N_bins)
    mat = torch.zeros((N_bins, N_bins), dtype=torch.float64)
    mat[idx, idx] = -2 / bw**2
    
    mat[idx[1:], idx[:-1]] = mu[idx[:-1]] / 2 / bw + 1 / bw**2    #   M_{i+1,i} = -mu[i]/2
    mat[idx[:-1], idx[1:]] = -mu[idx[1:]] / 2 / bw + 1 / bw**2    #   M_{i+1,i} = mu[i]/2
    mat[idx, idx] -= mat.sum(0) # fix edges

    return mat, bins

_mat, _bins = transition_matrix()
_D, _Q = torch.linalg.eig(_mat)
_Q_inv = torch.linalg.inv(_Q)
_sigmas = torch.from_numpy(np.load('chain_stats.npy'))


def add_noise(dists, residue_index, mask, t, device='cpu'):

    sigmas, Q, D, Q_inv, bins = _sigmas.to(device), _Q.to(device), _D.to(device), _Q_inv.to(device), _bins.to(device)
    
    mask = mask.unsqueeze(-1) * mask.unsqueeze(-2)
    # dists = torch.sum((pseudo_beta[..., None, :] - pseudo_beta[..., None, :, :]) ** 2, dim=-1)**0.5
    offsets = torch.abs(residue_index.unsqueeze(-1) - residue_index.unsqueeze(-2))
    sigmas = sigmas[offsets]
    ndists = dists / sigmas * mask
    
    bindists = (ndists.unsqueeze(-1) > bins).sum(-1)
    bindists = torch.clamp(bindists, 0, 999)
    
    P = ((Q*torch.exp(D*t)) @ Q_inv).T # now we have a row stochatic matrix  P_ij = P(i -> j)
    probs = P.real[bindists]   # this is equivalent to left multiplication by basis e_i

    probs = torch.clamp(probs / probs.sum(-1, keepdims=True), 0, 1)
    newbindists = Categorical(probs, validate_args=False).sample()
    cbins = (bins[1:] + bins[:-1]) / 2
    newdists = cbins[newbindists] * mask * sigmas

    return newdists.float()

def sample_posterior(orig_dists, noisy_dists, residue_index, mask, s, t, device='cpu'):
    sigmas, Q, D, Q_inv, bins = _sigmas.to(device), _Q.to(device), _D.to(device), _Q_inv.to(device), _bins.to(device)
    mask = mask.unsqueeze(-1) * mask.unsqueeze(-2)

    P_0s = ((Q*torch.exp(D*s)) @ Q_inv).T
    P_st = ((Q*torch.exp(D*(t-s))) @ Q_inv).T

    offsets = torch.abs(residue_index.unsqueeze(-1) - residue_index.unsqueeze(-2))
    sigmas = sigmas[offsets]
    
    orig_ndists = orig_dists / sigmas * mask
    orig_bindists = (orig_ndists.unsqueeze(-1) > bins).sum(-1)
    orig_bindists = torch.clamp(orig_bindists, 0, 999)

    noisy_ndists = noisy_dists / sigmas * mask
    noisy_bindists = (noisy_ndists.unsqueeze(-1) > bins).sum(-1)
    noisy_bindists = torch.clamp(noisy_bindists, 0, 999)

    probs = P_0s.real[orig_bindists] * P_st.T.real[noisy_bindists]
    probs = torch.clamp(probs / probs.sum(-1, keepdims=True), 0, 1)
    newbindists = Categorical(probs, validate_args=False).sample()
    cbins = (bins[1:] + bins[:-1]) / 2
    newdists = cbins[newbindists] * mask * sigmas
    
    return newdists.float()

def sample_prior(residue_index, device='cpu'):

    sigmas, Q, D, Q_inv, bins = _sigmas.to(device), _Q.to(device), _D.to(device), _Q_inv.to(device), _bins.to(device)
    B, L = residue_index.shape
    probs = Q[:,D.real.argmax()].real
    probs = torch.clamp(probs / probs.sum(-1, keepdims=True), 0, 1).broadcast_to(B, L, L, 1000)
    
    offsets = torch.abs(residue_index.unsqueeze(-1) - residue_index.unsqueeze(-2))
    sigmas = sigmas[offsets]
    
    newbindists = Categorical(probs, validate_args=False).sample()
       
    cbins = (bins[1:] + bins[:-1]) / 2
    newdists = cbins[newbindists] * sigmas
    return newdists.float()
'''
