"""Validation-point accuracy calculations from accuracy-mapping.ipynb."""
from __future__ import annotations
import numpy as np
from .metrics import user_producer_accuracy


def assess_table(rows, truth_column, prediction_column, classes=range(1,7),
                 class_names=None, area_weights=None):
    classes=list(classes); lookup={value:i for i,value in enumerate(classes)}
    cm=np.zeros((len(classes),len(classes)),dtype=int)
    for row in rows:
        truth=int(row[truth_column]); prediction=int(row[prediction_column])
        if truth in lookup and prediction in lookup:
            cm[lookup[truth],lookup[prediction]] += 1
    tp=np.diag(cm); oa=float(tp.sum()/cm.sum()) if cm.sum() else float('nan')
    names={i:class_names.get(classes[i],str(classes[i])) for i in range(len(classes))} if class_names else None
    up=user_producer_accuracy(cm,names,0)
    result={'overall_accuracy':oa,'confusion':cm.tolist(),**up}
    if area_weights and class_names:
        result['area_weighted_oa']=float(sum(
            area_weights.get(c,0.0)*up['producer_accuracy'].get(class_names[c],0.0)
            for c in classes if np.isfinite(up['producer_accuracy'].get(class_names[c],np.nan))))
    return result
