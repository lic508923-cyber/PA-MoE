"""PCA-projected diagonal GMM with conservative few-shot selection."""
from __future__ import annotations
import math
import torch
from torch import nn

class GMMEnergy(nn.Module):
    def __init__(self, hidden_dim: int, num_components: int = 4, projection_dim: int = 32,
                 min_var: float = 1e-3, min_samples_per_component: int = 20,
                 covariance_shrinkage: float = .1) -> None:
        super().__init__()
        self.hidden_dim=hidden_dim; self.num_components=num_components
        self.projection_dim=min(projection_dim,hidden_dim); self.min_var=min_var
        self.min_samples_per_component=min_samples_per_component
        self.covariance_shrinkage=covariance_shrinkage
        self.register_buffer("projection_mean",torch.zeros(hidden_dim))
        self.register_buffer("projection",torch.zeros(self.projection_dim,hidden_dim))
        self.register_buffer("active_projection_dim",torch.tensor(1,dtype=torch.long))
        self.register_buffer("mixture_weights",torch.full((num_components,),1.0/num_components))
        self.register_buffer("means",torch.zeros(num_components,self.projection_dim))
        self.register_buffer("variances",torch.ones(num_components,self.projection_dim))
        self.register_buffer("active_components",torch.tensor(1,dtype=torch.long))
        self.register_buffer("is_fitted",torch.tensor(False))

    def project(self,hidden):
        d=int(self.active_projection_dim.item())
        return (hidden-self.projection_mean) @ self.projection[:d].T

    def compute_energy(self,hidden):
        projected=self.project(hidden); k=int(self.active_components.item()); d=projected.size(1)
        weights=self.mixture_weights[:k].clamp_min(1e-12); means=self.means[:k,:d]
        variances=self.variances[:k,:d].clamp_min(self.min_var)
        difference=projected[:,None,:]-means[None,:,:]
        log_prob=-0.5*((difference.square()/variances[None,:,:]).sum(-1)+torch.log(variances).sum(-1)[None,:]+d*math.log(2*math.pi))
        component=log_prob+torch.log(weights)[None,:]
        return -torch.logsumexp(component,dim=-1),torch.softmax(component,dim=-1)

    def forward(self,hidden):
        energy,responsibility=self.compute_energy(hidden)
        return {"energy":energy,"responsibility":responsibility,"projected_hidden":self.project(hidden)}

    @torch.no_grad()
    def fit_normal(self,hidden,max_iter=50,tolerance=1e-4):
        if hidden.ndim!=2 or hidden.size(1)!=self.hidden_dim or hidden.size(0)<2:
            raise ValueError("normal hidden states must have shape [N, hidden_dim] with N >= 2")
        centered=hidden-hidden.mean(0); d=min(self.projection_dim,hidden.size(0)-1,self.hidden_dim)
        # SVD produces a deterministic orthonormal low-dimensional density space.
        _,_,vh=torch.linalg.svd(centered,full_matrices=False)
        self.projection_mean.copy_(hidden.mean(0)); self.projection.zero_(); self.projection[:d].copy_(vh[:d])
        self.active_projection_dim.fill_(d); data=centered @ vh[:d].T
        # Few-shot supports stay single-Gaussian until each component has ample evidence.
        k=min(self.num_components,max(1,hidden.size(0)//self.min_samples_per_component))
        # Deterministic farthest-point variant of k-means++ initialization.
        chosen=[int(torch.argmax(data.square().sum(1)).item())]
        while len(chosen)<k:
            distance=torch.cdist(data,data[chosen]).square().min(1).values
            chosen.append(int(torch.argmax(distance).item()))
        means=data[torch.tensor(chosen,device=data.device)].clone()
        variances=data.var(0,unbiased=False).clamp_min(self.min_var).expand(k,-1).clone(); weights=data.new_full((k,),1/k)
        previous=None
        for _ in range(max_iter):
            diff=data[:,None,:]-means[None,:,:]
            log_prob=-0.5*((diff.square()/variances[None,:,:]).sum(-1)+torch.log(variances).sum(-1)[None,:]+d*math.log(2*math.pi))+torch.log(weights.clamp_min(1e-12))[None,:]
            responsibility=torch.softmax(log_prob,dim=1); counts=responsibility.sum(0).clamp_min(1e-6)
            weights=counts/counts.sum(); means=(responsibility.T@data)/counts[:,None]
            diff=data[:,None,:]-means[None,:,:]; variances=(responsibility[:,:,None]*diff.square()).sum(0)/counts[:,None]
            global_variance=data.var(0,unbiased=False).clamp_min(self.min_var)
            variances=(1-self.covariance_shrinkage)*variances+self.covariance_shrinkage*global_variance[None,:]
            variances.clamp_(min=self.min_var); likelihood=torch.logsumexp(log_prob,dim=1).mean()
            if previous is not None and abs(float(likelihood-previous))<tolerance: break
            previous=likelihood
        self.mixture_weights.zero_(); self.mixture_weights[:k].copy_(weights)
        self.means.zero_(); self.means[:k,:d].copy_(means)
        self.variances.fill_(1); self.variances[:k,:d].copy_(variances)
        self.active_components.fill_(k); self.is_fitted.fill_(True)
