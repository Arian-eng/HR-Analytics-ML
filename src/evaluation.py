import numpy as np
from scipy.stats import binomtest, chi2
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, r2_score, mean_absolute_error, mean_squared_error


def classification_metrics(y_true, y_pred):
    return {"accuracy":float(accuracy_score(y_true,y_pred)),"precision":float(precision_score(y_true,y_pred,zero_division=0)),"recall":float(recall_score(y_true,y_pred,zero_division=0)),"f1":float(f1_score(y_true,y_pred,zero_division=0))}
def regression_metrics(y_true,y_pred):
    return {"r2":float(r2_score(y_true,y_pred)),"mae":float(mean_absolute_error(y_true,y_pred)),"rmse":float(mean_squared_error(y_true,y_pred)**0.5)}
def percentile_interval(values, confidence=.95):
    a=(1-confidence)/2; lo,hi=np.quantile(np.asarray(values,float),[a,1-a]); return [float(lo),float(hi)]

def bootstrap_classification(y_true,y_pred,n_resamples=2000,seed=42,batch=100):
    y=np.asarray(y_true,np.int8); p=np.asarray(y_pred,np.int8); n=len(y); rng=np.random.default_rng(seed); acc=[];prec=[];rec=[];f1=[]
    for start in range(0,n_resamples,batch):
        b=min(batch,n_resamples-start); idx=rng.integers(0,n,size=(b,n)); yt=y[idx]; yp=p[idx]
        tp=np.sum((yt==1)&(yp==1),axis=1); tn=np.sum((yt==0)&(yp==0),axis=1); fp=np.sum((yt==0)&(yp==1),axis=1); fn=np.sum((yt==1)&(yp==0),axis=1)
        acc.extend(((tp+tn)/n).tolist()); prec.extend(np.divide(tp,tp+fp,out=np.zeros_like(tp,dtype=float),where=(tp+fp)!=0).tolist()); rec.extend(np.divide(tp,tp+fn,out=np.zeros_like(tp,dtype=float),where=(tp+fn)!=0).tolist()); f1.extend(np.divide(2*tp,2*tp+fp+fn,out=np.zeros_like(tp,dtype=float),where=(2*tp+fp+fn)!=0).tolist())
    return {k:{"lower":percentile_interval(v)[0],"upper":percentile_interval(v)[1]} for k,v in {"accuracy":acc,"precision":prec,"recall":rec,"f1":f1}.items()}

def _reg_batch(y,p):
    err=y-p; mae=np.mean(np.abs(err),axis=1); rmse=np.sqrt(np.mean(err**2,axis=1)); mean=np.mean(y,axis=1,keepdims=True); sst=np.sum((y-mean)**2,axis=1); sse=np.sum(err**2,axis=1); r2=np.divide(sse,sst,out=np.full_like(sse,np.nan,dtype=float),where=sst!=0); r2=1-r2; return r2,mae,rmse

def bootstrap_regression(y_true,y_pred,n_resamples=4000,seed=42,batch=500):
    y=np.asarray(y_true,float); p=np.asarray(y_pred,float); n=len(y); rng=np.random.default_rng(seed); vals={"r2":[],"mae":[],"rmse":[]}
    for start in range(0,n_resamples,batch):
        b=min(batch,n_resamples-start); idx=rng.integers(0,n,size=(b,n)); r2,mae,rmse=_reg_batch(y[idx],p[idx]); vals['r2'].extend(r2[np.isfinite(r2)].tolist()); vals['mae'].extend(mae.tolist()); vals['rmse'].extend(rmse.tolist())
    return {k:{"lower":percentile_interval(v)[0],"upper":percentile_interval(v)[1]} for k,v in vals.items()}
def paired_bootstrap_difference(y_true,pred_base,pred_plus,n_resamples=4000,seed=42,batch=500):
    y=np.asarray(y_true,float); pb=np.asarray(pred_base,float); pp=np.asarray(pred_plus,float); n=len(y); rng=np.random.default_rng(seed); vals={"r2":[],"mae":[],"rmse":[]}; point_b=regression_metrics(y,pb); point_p=regression_metrics(y,pp)
    for start in range(0,n_resamples,batch):
        b=min(batch,n_resamples-start); idx=rng.integers(0,n,size=(b,n)); br2,bmae,brmse=_reg_batch(y[idx],pb[idx]); pr2,pmae,prmse=_reg_batch(y[idx],pp[idx]); vals['r2'].extend((pr2-br2)[np.isfinite(pr2-br2)].tolist()); vals['mae'].extend((pmae-bmae).tolist()); vals['rmse'].extend((prmse-brmse).tolist())
    out={}
    for k,v in vals.items():
        lo,hi=percentile_interval(v); out[k]={"delta":float(point_p[k]-point_b[k]),"lower":lo,"upper":hi,"reliable_change":bool(not(lo<=0<=hi))}
    return out
def mcnemar_test(y_true,pred_a,pred_b):
    y=np.asarray(y_true); a=np.asarray(pred_a)==y; b=np.asarray(pred_b)==y; b01=int(np.sum(a&~b)); b10=int(np.sum(~a&b)); d=b01+b10
    if d<25: p=float(binomtest(min(b01,b10),n=d,p=.5,alternative='two-sided').pvalue) if d else 1.0; stat=None; method='exact_binomial'
    else: stat=float((abs(b01-b10)-1)**2/d); p=float(chi2.sf(stat,1)); method='chi_square_continuity_corrected'
    return {"b01":b01,"b10":b10,"discordant":d,"statistic":stat,"p_value":p,"method":method,"significant_0_05":bool(p<.05)}
