import itertools, json, time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import LinearSVC
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV, StratifiedKFold, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from .config import *
from .preprocessing import load_public, make_preprocessor
from .evaluation import classification_metrics, bootstrap_classification, mcnemar_test

MODEL_ORDER = ["Random Forest", "Decision Tree", "LinearSVC", "MLPClassifier"]


def _search(model_name, X_train):
    scale = model_name in ("LinearSVC", "MLPClassifier")
    prep,_,_=make_preprocessor(X_train, scale_numeric=scale)
    if model_name == "Random Forest":
        estimator=RandomForestClassifier(random_state=RANDOM_STATE,n_jobs=-1)
        grid={"model__n_estimators":[200,400],"model__max_depth":[None,20],"model__min_samples_leaf":[1,2],"model__max_features":["sqrt"]}
    elif model_name == "Decision Tree":
        estimator=DecisionTreeClassifier(random_state=RANDOM_STATE)
        grid={"model__criterion":["gini","entropy"],"model__max_depth":[5,8,12],"model__min_samples_leaf":[2,5,10]}
    elif model_name == "LinearSVC":
        estimator=LinearSVC(random_state=RANDOM_STATE,class_weight="balanced",max_iter=20000)
        grid={"model__C":[0.1,1,10]}
    else:
        estimator=MLPClassifier(max_iter=500,early_stopping=True,random_state=RANDOM_STATE)
        grid={"model__hidden_layer_sizes":[(50,),(100,),(50,50)],"model__alpha":[0.0001,0.001],"model__learning_rate_init":[0.001,0.01],"model__activation":["relu","tanh"]}
    pipe=Pipeline([("preprocess",prep),("model",estimator)])
    cv=StratifiedKFold(n_splits=CLASSIFICATION_CV_FOLDS,shuffle=True,random_state=RANDOM_STATE)
    if model_name == "MLPClassifier":
        return RandomizedSearchCV(pipe,grid,n_iter=8,scoring="f1",cv=cv,random_state=RANDOM_STATE,n_jobs=-1,refit=True,return_train_score=False)
    return GridSearchCV(pipe,grid,scoring="f1",cv=cv,n_jobs=(1 if model_name=="Random Forest" else -1),refit=True,return_train_score=False)


def prepare_split(name):
    raw,X,y,path,constant=load_public(name)
    idx=np.arange(len(X))
    tr,te=train_test_split(idx,test_size=TEST_SIZE,stratify=y,random_state=RANDOM_STATE)
    return raw,X,y,path,constant,tr,te


def run_model(name, model_name, logger=print):
    raw,X,y,path,constant,tr,te=prepare_split(name)
    Xtr,Xte=X.iloc[tr],X.iloc[te]; ytr,yte=y.iloc[tr],y.iloc[te]
    t0=time.time(); search=_search(model_name,Xtr)
    logger(f"{name}/{model_name}: search_start train={len(tr)} test={len(te)}")
    search.fit(Xtr,ytr)
    pred=search.predict(Xte).astype(int)
    m=classification_metrics(yte.to_numpy(),pred)
    ci=bootstrap_classification(yte.to_numpy(),pred,CLASSIFICATION_BOOTSTRAPS,RANDOM_STATE)
    cm=confusion_matrix(yte,pred,labels=[0,1]).tolist()
    out={"dataset":name,"model":model_name,"rows":len(raw),"train_rows":len(tr),"test_rows":len(te),"positive_test":int(yte.sum()),
         "metrics":m,"bootstrap_95_ci":ci,"bootstrap_resamples":CLASSIFICATION_BOOTSTRAPS,"confusion_matrix":cm,
         "best_cv_f1":float(search.best_score_),"best_params":search.best_params_,"elapsed_seconds":float(time.time()-t0),
         "test_indices":te.tolist(),"y_true":yte.astype(int).tolist(),"y_pred":pred.tolist()}
    path_out=RESULTS_DIR/f"classification_{name}_{model_name.lower().replace(' ','_')}.json"
    path_out.write_text(json.dumps(out,indent=2,default=str),encoding="utf-8")
    logger(f"{name}/{model_name}: acc={m['accuracy']:.4f} precision={m['precision']:.4f} recall={m['recall']:.4f} f1={m['f1']:.4f} cv_f1={search.best_score_:.4f} elapsed={time.time()-t0:.1f}s")
    return out


def consolidate_dataset(name, logger=print):
    rows=[]
    for model in MODEL_ORDER:
        p=RESULTS_DIR/f"classification_{name}_{model.lower().replace(' ','_')}.json"
        rows.append(json.loads(p.read_text(encoding="utf-8")))
    y=rows[0]["y_true"]; idx=rows[0]["test_indices"]
    if any(r["test_indices"]!=idx or r["y_true"]!=y for r in rows): raise RuntimeError(f"{name}: models do not share identical test set")
    pairs=[]
    for a,b in itertools.combinations(rows,2):
        test=mcnemar_test(y,a["y_pred"],b["y_pred"]); test.update({"model_a":a["model"],"model_b":b["model"]}); pairs.append(test)
    out={"dataset":name,"methodology":{"split":"stratified 80/20, random_state=42","cv":"stratified 3-fold; select positive-class F1","bootstrap":"percentile 95% CI; 2000 resamples; seed=42","mcnemar":"all 6 pairs; exact binomial if discordant<25, else continuity-corrected chi-square"},"models":rows,"mcnemar":pairs}
    (RESULTS_DIR/f"classification_{name}.json").write_text(json.dumps(out,indent=2),encoding="utf-8")
    fig,axes=plt.subplots(1,4,figsize=(14,3.5))
    for ax,r in zip(axes,rows): ConfusionMatrixDisplay(np.array(r["confusion_matrix"]),display_labels=[0,1]).plot(ax=ax,colorbar=False); ax.set_title(r["model"])
    fig.suptitle(f"{name}: held-out confusion matrices"); fig.tight_layout(); fig.savefig(FIGURES_DIR/f"confusion_{name}.png",dpi=160); plt.close(fig)
    fig,ax=plt.subplots(figsize=(7,4)); x=np.arange(4); vals=[r["metrics"]["f1"] for r in rows]; los=[r["bootstrap_95_ci"]["f1"]["lower"] for r in rows]; his=[r["bootstrap_95_ci"]["f1"]["upper"] for r in rows]; err=np.array([np.array(vals)-np.array(los),np.array(his)-np.array(vals)])
    ax.bar(x,vals); ax.errorbar(x,vals,yerr=err,fmt="none",capsize=4); ax.set_xticks(x,[r["model"] for r in rows],rotation=20,ha="right"); ax.set_ylabel("F1"); ax.set_title(f"{name}: held-out F1 with 95% bootstrap CI"); fig.tight_layout(); fig.savefig(FIGURES_DIR/f"f1_ci_{name}.png",dpi=160); plt.close(fig)
    logger(f"{name}/McNemar: 6 pairwise tests complete")
    return out

RF_CANDIDATES = [
    {"n_estimators":n,"max_depth":d,"min_samples_leaf":leaf,"max_features":"sqrt"}
    for n in (200,400) for d in (None,20) for leaf in (1,2)
]

def run_rf_candidate(name, candidate_id, logger=print):
    from sklearn.model_selection import StratifiedKFold
    from sklearn.metrics import f1_score
    from joblib import Parallel, delayed
    raw,X,y,path,constant,tr,te=prepare_split(name); Xtr=X.iloc[tr]; ytr=y.iloc[tr]; params=RF_CANDIDATES[candidate_id]; cv=StratifiedKFold(n_splits=CLASSIFICATION_CV_FOLDS,shuffle=True,random_state=RANDOM_STATE); splits=list(cv.split(Xtr,ytr)); t0=time.time()
    logger(f"{name}/Random Forest candidate={candidate_id} params={params} cv_start parallel_folds=3")
    def one(a,b):
        prep,_,_=make_preprocessor(Xtr.iloc[a],scale_numeric=False); model=RandomForestClassifier(random_state=RANDOM_STATE,n_jobs=1,**params); pipe=Pipeline([('preprocess',prep),('model',model)]); pipe.fit(Xtr.iloc[a],ytr.iloc[a]); pred=pipe.predict(Xtr.iloc[b]); return float(f1_score(ytr.iloc[b],pred,zero_division=0))
    scores=Parallel(n_jobs=3,prefer='processes')(delayed(one)(a,b) for a,b in splits)
    for fold,sc in enumerate(scores,1): logger(f"{name}/RF candidate={candidate_id} fold={fold} f1={sc:.4f}")
    out={"dataset":name,"candidate_id":candidate_id,"params":params,"fold_f1":scores,"mean_cv_f1":float(np.mean(scores)),"elapsed_seconds":float(time.time()-t0)}; (RESULTS_DIR/f"rf_cv_{name}_{candidate_id}.json").write_text(json.dumps(out,indent=2),encoding='utf-8'); logger(f"{name}/RF candidate={candidate_id} mean_cv_f1={out['mean_cv_f1']:.4f} elapsed={out['elapsed_seconds']:.1f}s"); return out

def finalize_rf(name, logger=print):
    raw,X,y,path,constant,tr,te=prepare_split(name); candidates=[json.loads((RESULTS_DIR/f"rf_cv_{name}_{i}.json").read_text()) for i in range(len(RF_CANDIDATES))]; best=max(candidates,key=lambda r:r['mean_cv_f1']); Xtr,Xte=X.iloc[tr],X.iloc[te]; ytr,yte=y.iloc[tr],y.iloc[te]; t0=time.time(); prep,_,_=make_preprocessor(Xtr,scale_numeric=False); model=RandomForestClassifier(random_state=RANDOM_STATE,n_jobs=-1,**best['params']); pipe=Pipeline([('preprocess',prep),('model',model)]); pipe.fit(Xtr,ytr); pred=pipe.predict(Xte).astype(int); m=classification_metrics(yte.to_numpy(),pred); ci=bootstrap_classification(yte.to_numpy(),pred,CLASSIFICATION_BOOTSTRAPS,RANDOM_STATE); cm=confusion_matrix(yte,pred,labels=[0,1]).tolist(); out={"dataset":name,"model":"Random Forest","rows":len(raw),"train_rows":len(tr),"test_rows":len(te),"positive_test":int(yte.sum()),"metrics":m,"bootstrap_95_ci":ci,"bootstrap_resamples":CLASSIFICATION_BOOTSTRAPS,"confusion_matrix":cm,"best_cv_f1":best['mean_cv_f1'],"best_params":{f"model__{k}":v for k,v in best['params'].items()},"elapsed_seconds":float(time.time()-t0),"test_indices":te.tolist(),"y_true":yte.astype(int).tolist(),"y_pred":pred.tolist(),"rf_grid_candidates":candidates}; (RESULTS_DIR/f"classification_{name}_random_forest.json").write_text(json.dumps(out,indent=2,default=str),encoding='utf-8'); logger(f"{name}/Random Forest FINAL: best_candidate={best['candidate_id']} acc={m['accuracy']:.4f} precision={m['precision']:.4f} recall={m['recall']:.4f} f1={m['f1']:.4f} cv_f1={best['mean_cv_f1']:.4f}"); return out

from sklearn.model_selection import ParameterSampler
MLP_SPACE={"hidden_layer_sizes":[(50,),(100,),(50,50)],"alpha":[0.0001,0.001],"learning_rate_init":[0.001,0.01],"activation":["relu","tanh"]}
MLP_CANDIDATES=list(ParameterSampler(MLP_SPACE,n_iter=8,random_state=RANDOM_STATE))

def run_mlp_candidate(name,candidate_id,logger=print):
    from sklearn.metrics import f1_score
    from joblib import Parallel, delayed
    raw,X,y,path,constant,tr,te=prepare_split(name); Xtr=X.iloc[tr]; ytr=y.iloc[tr]; params=MLP_CANDIDATES[candidate_id]; cv=StratifiedKFold(n_splits=CLASSIFICATION_CV_FOLDS,shuffle=True,random_state=RANDOM_STATE); splits=list(cv.split(Xtr,ytr)); t0=time.time(); logger(f"{name}/MLP candidate={candidate_id} params={params} cv_start parallel_folds=3")
    def one_fold(a,b):
        prep,_,_=make_preprocessor(Xtr.iloc[a],scale_numeric=True); model=MLPClassifier(max_iter=500,early_stopping=True,random_state=RANDOM_STATE,**params); pipe=Pipeline([('preprocess',prep),('model',model)]); pipe.fit(Xtr.iloc[a],ytr.iloc[a]); pred=pipe.predict(Xtr.iloc[b]); return float(f1_score(ytr.iloc[b],pred,zero_division=0))
    scores=Parallel(n_jobs=CLASSIFICATION_CV_FOLDS)(delayed(one_fold)(a,b) for a,b in splits)
    for fold,sc in enumerate(scores,1): logger(f"{name}/MLP candidate={candidate_id} fold={fold} f1={sc:.4f}")
    out={"dataset":name,"candidate_id":candidate_id,"params":params,"fold_f1":scores,"mean_cv_f1":float(np.mean(scores)),"elapsed_seconds":float(time.time()-t0)}; (RESULTS_DIR/f"mlp_cv_{name}_{candidate_id}.json").write_text(json.dumps(out,indent=2,default=str),encoding='utf-8'); logger(f"{name}/MLP candidate={candidate_id} mean_cv_f1={out['mean_cv_f1']:.4f} elapsed={out['elapsed_seconds']:.1f}s"); return out

def finalize_mlp(name,logger=print):
    raw,X,y,path,constant,tr,te=prepare_split(name); candidates=[json.loads((RESULTS_DIR/f"mlp_cv_{name}_{i}.json").read_text()) for i in range(len(MLP_CANDIDATES))]; best=max(candidates,key=lambda r:r['mean_cv_f1']); Xtr,Xte=X.iloc[tr],X.iloc[te]; ytr,yte=y.iloc[tr],y.iloc[te]; t0=time.time(); prep,_,_=make_preprocessor(Xtr,scale_numeric=True); model=MLPClassifier(max_iter=500,early_stopping=True,random_state=RANDOM_STATE,**best['params']); pipe=Pipeline([('preprocess',prep),('model',model)]); pipe.fit(Xtr,ytr); pred=pipe.predict(Xte).astype(int); m=classification_metrics(yte.to_numpy(),pred); ci=bootstrap_classification(yte.to_numpy(),pred,CLASSIFICATION_BOOTSTRAPS,RANDOM_STATE); cm=confusion_matrix(yte,pred,labels=[0,1]).tolist(); out={"dataset":name,"model":"MLPClassifier","rows":len(raw),"train_rows":len(tr),"test_rows":len(te),"positive_test":int(yte.sum()),"metrics":m,"bootstrap_95_ci":ci,"bootstrap_resamples":CLASSIFICATION_BOOTSTRAPS,"confusion_matrix":cm,"best_cv_f1":best['mean_cv_f1'],"best_params":{f"model__{k}":v for k,v in best['params'].items()},"elapsed_seconds":float(time.time()-t0),"test_indices":te.tolist(),"y_true":yte.astype(int).tolist(),"y_pred":pred.tolist(),"randomized_8_candidates":candidates}; (RESULTS_DIR/f"classification_{name}_mlpclassifier.json").write_text(json.dumps(out,indent=2,default=str),encoding='utf-8'); logger(f"{name}/MLP FINAL: best_candidate={best['candidate_id']} acc={m['accuracy']:.4f} precision={m['precision']:.4f} recall={m['recall']:.4f} f1={m['f1']:.4f} cv_f1={best['mean_cv_f1']:.4f}"); return out
