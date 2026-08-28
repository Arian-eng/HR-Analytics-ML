import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score, davies_bouldin_score
from .config import *
from .preprocessing import load_public
from .survey_analysis import load_survey


def _evaluate(Z):
    rows=[]
    for k in K_VALUES:
        km=KMeans(n_clusters=k,n_init=KMEANS_N_INIT,max_iter=KMEANS_MAX_ITER,random_state=RANDOM_STATE).fit(Z)
        kwargs={}
        if len(Z)>SILHOUETTE_SAMPLE_SIZE: kwargs={"sample_size":SILHOUETTE_SAMPLE_SIZE,"random_state":RANDOM_STATE}
        rows.append({"k":k,"sse":float(km.inertia_),"silhouette":float(silhouette_score(Z,km.labels_,**kwargs)),"davies_bouldin":float(davies_bouldin_score(Z,km.labels_))})
    return rows

def _figure(name,rows):
    fig,axes=plt.subplots(1,3,figsize=(13,3.8)); metrics=[("sse","SSE / Inertia"),("silhouette","Silhouette"),("davies_bouldin","Davies-Bouldin")]
    for ax,(key,label) in zip(axes,metrics): ax.plot([r['k'] for r in rows],[r[key] for r in rows],marker='o'); ax.set_xlabel('k'); ax.set_ylabel(label); ax.set_title(label)
    fig.suptitle(f"{name}: K-Means k=2..7"); fig.tight_layout(); fig.savefig(FIGURES_DIR/f"kmeans_{name}.png",dpi=160); plt.close(fig)

def run_public_kmeans(name,logger=print):
    raw,X,y,_,_=load_public(name); numeric=X.select_dtypes(include=np.number).copy(); numeric=numeric[[c for c in numeric if numeric[c].nunique(dropna=False)>1]]
    Z=Pipeline([("impute",SimpleImputer(strategy="median")),("scale",StandardScaler())]).fit_transform(numeric)
    rows=_evaluate(Z); best=max(rows,key=lambda r:r['silhouette']); final=KMeans(n_clusters=best['k'],n_init=KMEANS_N_INIT,max_iter=KMEANS_MAX_ITER,random_state=RANDOM_STATE).fit(Z); labels=final.labels_; sizes=pd.Series(labels).value_counts().sort_index().to_dict(); profiles=[{"cluster":int(c),"size":int((labels==c).sum()),"positive_target_share":float(y.to_numpy()[labels==c].mean())} for c in range(best['k'])]
    out={"dataset":name,"features":"numeric non-constant predictors only; target, identifiers and constant columns excluded","numeric_feature_count":len(numeric.columns),"k_results":rows,"selected_k":best['k'],"selection_rule":"maximum silhouette; SSE and Davies-Bouldin reported simultaneously","cluster_sizes":{str(k):int(v) for k,v in sizes.items()},"target_profile_after_clustering":profiles}
    (RESULTS_DIR/f"clustering_{name}.json").write_text(json.dumps(out,indent=2),encoding="utf-8"); _figure(name,rows); logger(f"{name}/KMeans: features={len(numeric.columns)} selected_k={best['k']} silhouette={best['silhouette']:.4f} SSE={best['sse']:.3f} DB={best['davies_bouldin']:.4f}"); return out

def run_ghrm_kmeans(logger=print):
    df,comp,_=load_survey(); features=GEE_PLUS_FEATURES; Z=StandardScaler().fit_transform(SimpleImputer(strategy="median").fit_transform(comp[features])); rows=_evaluate(Z); best=max(rows,key=lambda r:r['silhouette']); final=KMeans(n_clusters=best['k'],n_init=KMEANS_N_INIT,max_iter=KMEANS_MAX_ITER,random_state=RANDOM_STATE).fit(Z); labels=final.labels_; sizes=pd.Series(labels).value_counts().sort_index().to_dict(); prof=[]
    for c in range(best['k']):
        mask=labels==c; row={"cluster":int(c),"size":int(mask.sum()),"mean_FEP":float(comp.loc[mask,'FEP'].mean())}; row.update({f"mean_{f}":float(comp.loc[mask,f].mean()) for f in features}); prof.append(row)
    out={"dataset":"ghrm","features":features,"FEP_used_for_clustering":False,"k_results":rows,"selected_k":best['k'],"selection_rule":"maximum silhouette; SSE and Davies-Bouldin reported simultaneously","cluster_sizes":{str(k):int(v) for k,v in sizes.items()},"profiles_after_clustering":prof}
    (RESULTS_DIR/'clustering_ghrm.json').write_text(json.dumps(out,indent=2),encoding='utf-8'); _figure('ghrm',rows); logger(f"ghrm/KMeans: features=5 selected_k={best['k']} silhouette={best['silhouette']:.4f} SSE={best['sse']:.3f} DB={best['davies_bouldin']:.4f}"); return out
