import json, time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.svm import LinearSVR
from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV, KFold, train_test_split
from .config import *
from .evaluation import regression_metrics, bootstrap_regression, paired_bootstrap_difference

MODEL_ORDER=["Random Forest Regressor","Decision Tree Regressor","LinearSVR","MLPRegressor"]
VARIANTS={"Base":BASE_FEATURES,"GEE+":GEE_PLUS_FEATURES}

def _slug(x): return x.lower().replace('+','plus').replace(' ','_')
def cronbach_alpha(frame):
    x=frame.astype(float).to_numpy(); k=x.shape[1]; total=x.sum(axis=1)
    return float(k/(k-1)*(1-x.var(axis=0,ddof=1).sum()/total.var(ddof=1)))
def load_survey():
    df=pd.read_csv(DATA_DIR/SURVEY_FILE).rename(columns={"GDT3":"GTD3"})
    comp=pd.DataFrame(index=df.index); reliability={}
    for c,items in SURVEY_CONSTRUCTS.items():
        vals=df[items].apply(pd.to_numeric,errors="coerce"); comp[c]=vals.mean(axis=1,skipna=False)
        reliability[c]={"items":items,"alpha":cronbach_alpha(vals),"mean":float(comp[c].mean()),"std":float(comp[c].std(ddof=1))}
    return df,comp,reliability
def shared_split(comp):
    idx=np.arange(len(comp)); return train_test_split(idx,test_size=TEST_SIZE,random_state=RANDOM_STATE)
def _search(model_name, features):
    prep=ColumnTransformer([("num",Pipeline([("impute",SimpleImputer(strategy="median")),("scale",StandardScaler())]),features)])
    if model_name=="Random Forest Regressor":
        est=RandomForestRegressor(random_state=RANDOM_STATE,n_jobs=1); grid={"model__n_estimators":[200,400],"model__max_depth":[None,10],"model__min_samples_leaf":[1,2],"model__max_features":[1.0,"sqrt"]}
    elif model_name=="Decision Tree Regressor":
        est=DecisionTreeRegressor(random_state=RANDOM_STATE); grid={"model__max_depth":[3,5,8,None],"model__min_samples_leaf":[2,5,10]}
    elif model_name=="LinearSVR":
        est=LinearSVR(random_state=RANDOM_STATE,max_iter=100000); grid={"model__C":[0.1,1,10],"model__epsilon":[0,0.1,0.2]}
    else:
        est=MLPRegressor(random_state=RANDOM_STATE,max_iter=1000,early_stopping=True); grid={"model__hidden_layer_sizes":[(20,),(50,),(50,25)],"model__alpha":[0.0001,0.001],"model__learning_rate_init":[0.001,0.01]}
    pipe=Pipeline([("preprocess",prep),("model",est)]); cv=KFold(n_splits=REGRESSION_CV_FOLDS,shuffle=True,random_state=RANDOM_STATE)
    if model_name=="MLPRegressor": return RandomizedSearchCV(pipe,grid,n_iter=8,scoring="neg_root_mean_squared_error",cv=cv,random_state=RANDOM_STATE,n_jobs=-1,refit=True)
    return GridSearchCV(pipe,grid,scoring="neg_root_mean_squared_error",cv=cv,n_jobs=-1,refit=True)
def prepare_survey(logger=print):
    df,comp,reliability=load_survey(); tr,te=shared_split(comp)
    meta={"rows":len(df),"missing_cells":int(df.isna().sum().sum()),"reliability":reliability,"construct_correlations":comp.corr().to_dict(),"split":{"random_state":RANDOM_STATE,"train_rows":len(tr),"test_rows":len(te),"train_indices":tr.tolist(),"test_indices":te.tolist()}}
    (RESULTS_DIR/'ghrm_meta.json').write_text(json.dumps(meta,indent=2),encoding='utf-8')
    logger(f"GHRM: rows={len(comp)} train={len(tr)} test={len(te)} shared_split=True")
    for c,v in reliability.items(): logger(f"GHRM/{c}: alpha={v['alpha']:.4f} mean={v['mean']:.4f} sd={v['std']:.4f}")
    return meta
def run_survey_model(variant, model_name, logger=print):
    df,comp,_=load_survey(); tr,te=shared_split(comp); features=VARIANTS[variant]; y=comp['FEP']; t0=time.time(); search=_search(model_name,features)
    logger(f"GHRM/{variant}/{model_name}: search_start train={len(tr)} test={len(te)}")
    search.fit(comp.iloc[tr][features],y.iloc[tr]); pred=search.predict(comp.iloc[te][features]); m=regression_metrics(y.iloc[te].to_numpy(),pred); ci=bootstrap_regression(y.iloc[te].to_numpy(),pred,REGRESSION_BOOTSTRAPS,RANDOM_STATE)
    row={"variant":variant,"model":model_name,"features":features,"train_rows":len(tr),"test_rows":len(te),"metrics":m,"bootstrap_95_ci":ci,"bootstrap_resamples":REGRESSION_BOOTSTRAPS,"best_cv_rmse":float(-search.best_score_),"best_params":search.best_params_,"elapsed_seconds":float(time.time()-t0),"test_indices":te.tolist(),"y_true":y.iloc[te].tolist(),"y_pred":pred.tolist()}
    (RESULTS_DIR/f"ghrm_{_slug(variant)}_{_slug(model_name)}.json").write_text(json.dumps(row,indent=2),encoding='utf-8')
    logger(f"GHRM/{variant}/{model_name}: R2={m['r2']:.4f} MAE={m['mae']:.4f} RMSE={m['rmse']:.4f} CV_RMSE={-search.best_score_:.4f} elapsed={time.time()-t0:.1f}s")
    return row
def consolidate_survey(logger=print):
    meta=json.loads((RESULTS_DIR/'ghrm_meta.json').read_text()); results={}; predictions={}; y_true=None; test_idx=None
    for variant in VARIANTS:
        results[variant]=[]; predictions[variant]={}
        for model in MODEL_ORDER:
            r=json.loads((RESULTS_DIR/f"ghrm_{_slug(variant)}_{_slug(model)}.json").read_text()); results[variant].append(r); predictions[variant][model]=r['y_pred']; y_true=r['y_true'] if y_true is None else y_true; test_idx=r['test_indices'] if test_idx is None else test_idx
            if r['test_indices']!=test_idx or r['y_true']!=y_true: raise RuntimeError('GHRM models do not share same test set')
    paired={}
    for model in MODEL_ORDER:
        paired[model]=paired_bootstrap_difference(y_true,predictions['Base'][model],predictions['GEE+'][model],REGRESSION_BOOTSTRAPS,RANDOM_STATE); d=paired[model]['r2']; logger(f"GHRM/Base_vs_GEE+/{model}: delta_R2={d['delta']:.4f} CI=[{d['lower']:.4f},{d['upper']:.4f}] reliable={d['reliable_change']}")
    out={**meta,"regression":results,"paired_base_vs_gee_plus":paired,"predictions":{"y_true":y_true,**predictions}}
    (RESULTS_DIR/'ghrm.json').write_text(json.dumps(out,indent=2),encoding='utf-8')
    fig,ax=plt.subplots(figsize=(8,4.5)); x=np.arange(len(MODEL_ORDER)); w=.35; base=[results['Base'][i]['metrics']['rmse'] for i in range(4)]; plus=[results['GEE+'][i]['metrics']['rmse'] for i in range(4)]; ax.bar(x-w/2,base,w,label='Base'); ax.bar(x+w/2,plus,w,label='GEE+'); ax.set_xticks(x,MODEL_ORDER,rotation=20,ha='right'); ax.set_ylabel('Held-out RMSE'); ax.legend(); ax.set_title('GHRM: Base vs GEE+ on same 64 test cases'); fig.tight_layout(); fig.savefig(FIGURES_DIR/'ghrm_regression_rmse.png',dpi=160); plt.close(fig)
    return out
def run_survey(logger=print):
    prepare_survey(logger)
    for v in VARIANTS:
        for m in MODEL_ORDER: run_survey_model(v,m,logger)
    return consolidate_survey(logger)
