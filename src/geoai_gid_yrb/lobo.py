"""Leave-one-basin-out split and metric helpers from lobo-results-part2.ipynb."""
from __future__ import annotations
import numpy as np


def make_lobo_arrays(by_basin, test_basin, cap=1500, val_fraction=0.12, seed=0):
    rng=np.random.default_rng(seed); xs=[]; ys=[]
    for basin,(x,y) in by_basin.items():
        if basin==test_basin: continue
        index=rng.permutation(len(x))[:cap]
        xs.append(x[index].copy()); ys.append(y[index].copy())
    x=np.concatenate(xs); y=np.concatenate(ys)
    order=rng.permutation(len(x)); x=x[order]; y=y[order]; nval=int(len(x)*val_fraction)
    xte,yte=by_basin[test_basin]
    return x[nval:],y[nval:],x[:nval],y[:nval],xte,yte


def trainable_state_dict(model):
    return {name:parameter.detach().cpu() for name,parameter in model.named_parameters()
            if parameter.requires_grad}
