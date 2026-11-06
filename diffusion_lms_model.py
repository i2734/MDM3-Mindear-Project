"""ATC diffusion LMS with multiscale spatial basis for 2D sound level estimation.

The ATC update is the no-measurement-sharing (C=I) case of Cattivelli & Sayed
(2010), Eq. (24). Spatial RBF design and dB-domain regression are project
modelling choices, not formulas from the paper.
"""
import numpy as np
from scipy.spatial import distance_matrix
from scipy.sparse.csgraph import minimum_spanning_tree

class DiffusionAcousticLMS:
    def __init__(self, n_basis=100, basis_width_ratio=0.08,
                 learning_rate=0.15, n_iterations=1500,
                 communication_radius=None, warm_start=True,
                 readout='mean'):
        self.n_basis=int(n_basis)
        self.basis_width_ratio=float(basis_width_ratio)
        self.learning_rate=float(learning_rate)
        self.n_iterations=int(n_iterations)
        self.communication_radius=communication_radius
        self.warm_start=bool(warm_start)
        self.readout=readout
        if self.n_basis<1 or self.basis_width_ratio<=0 or self.learning_rate<=0 or self.n_iterations<1:
            raise ValueError('Invalid model hyperparameters')
        if readout not in ('mean','node0'):
            raise ValueError('readout must be mean or node0')

    def _features(self, positions, centers, width):
        sq=distance_matrix(positions,centers)**2
        # Constant background plus local Gaussian bases; unlike normalised RBF,
        # each local feature retains its own spatial decay.
        local=np.exp(-sq/(2*width**2))
        return np.column_stack((np.ones(len(positions)),local))

    def _weights(self, phones):
        n=len(phones)
        if n==1:return np.ones((1,1)),0.
        dist=distance_matrix(phones,phones)
        if self.communication_radius is None:
            radius=max(float(minimum_spanning_tree(dist).max())*(1+1e-8),1e-8)
        else:radius=float(self.communication_radius)
        edges=dist<=radius+1e-10
        np.fill_diagonal(edges,True)
        seen={0}; stack=[0]
        while stack:
            for other in np.flatnonzero(edges[stack.pop()]):
                if int(other) not in seen:
                    seen.add(int(other));stack.append(int(other))
        if len(seen)!=n:raise ValueError('Disconnected communication network')
        degree=edges.sum(axis=1)-1
        W=np.zeros((n,n))
        for k in range(n):
            for l in np.flatnonzero(edges[k]):
                if l!=k:W[k,l]=1/(1+max(degree[k],degree[l]))
            W[k,k]=1-W[k].sum()
        return W,radius

    def fit_predict(self,phone_positions,phone_levels,query_positions,x=None,y=None):
        phones=np.asarray(phone_positions,dtype=float)
        levels=np.asarray(phone_levels,dtype=float)
        queries=np.asarray(query_positions,dtype=float)
        if levels.ndim==1:levels=levels[None,:]
        if phones.ndim!=2 or phones.shape[1]!=2 or levels.ndim!=2 or levels.shape[1]!=len(phones):
            raise ValueError('Expected phone positions (N,2) and phone levels (T,N)')
        if queries.ndim!=2 or queries.shape[1]!=2:raise ValueError('Expected query positions (M,2)')
        positions=phones.copy()
        if x is not None and y is not None:
            x=np.asarray(x);y=np.asarray(y)
            positions[:,0]=x[np.abs(phones[:,0,None]-x[None,:]).argmin(axis=1)]
            positions[:,1]=y[np.abs(phones[:,1,None]-y[None,:]).argmin(axis=1)]
        domain=np.vstack((phones,queries))
        lo=domain.min(axis=0);hi=domain.max(axis=0)
        side=int(np.ceil(np.sqrt(self.n_basis)))
        gx,gy=np.meshgrid(np.linspace(lo[0],hi[0],side),np.linspace(lo[1],hi[1],side))
        centers=np.column_stack((gx.ravel(),gy.ravel()))[:self.n_basis]
        width=max(self.basis_width_ratio*np.linalg.norm(hi-lo),1e-6)
        F=self._features(positions,centers,width)
        F_query=self._features(queries,centers,width)
        # Regression in dB is an explicit statistical modelling assumption:
        # it is NOT physically summing independent source energies in dB.
        offset=80.
        scale=20.
        D=(levels-offset)/scale
        W,radius=self._weights(phones)
        N,K=F.shape
        state=np.zeros((N,K))
        predictions=np.empty((len(levels),len(queries)))
        for t in range(len(levels)):
            if t==0 or not self.warm_start:
                state[:]=0.
                state[:,0]=np.mean(D[t])
            d=D[t]
            for _ in range(self.n_iterations):
                # Local LMS adapt: Cattivelli & Sayed (2010), Eq. (24), C=I.
                residual=d-np.einsum('nk,nk->n',F,state)
                psi=state+self.learning_rate*residual[:,None]*F
                # Exchange and combine intermediate coefficient estimates.
                state=W@psi
            coefficients=state.mean(axis=0) if self.readout=='mean' else state[0]
            predictions[t]=offset+scale*(F_query@coefficients)
        self.node_parameters_=state.copy()
        self.communication_weights_=W
        self.effective_communication_radius_=radius
        self.basis_centers_=centers
        self.basis_width_=width
        return predictions
