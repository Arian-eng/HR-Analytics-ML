"""Supplementary, explicitly post-hoc checks; independent of Chapter 4 test selection."""
import json
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, KFold, GridSearchCV
from sklearn.dummy import DummyRegressor, DummyClassifier
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error, accuracy_score, f1_score

ROOT = Path(__file__).resolve().parents[1]
df = pd.read_csv(ROOT/'data/HRM_DATASETS.csv').rename(columns={'GDT3':'GTD3'})
groups = {'GRS':['GRS1','GRS2','GRS3','GRS4'], 'GTD':['GTD1','GTD2','GTD3','GTD4','GTD5'],
          'GPA':['GPA1','GPA2','GPA3','GPA4','GPA5','GPA6'], 'GCM':['GCM1','GCM2','GCM3','GCM4'],
          'FEP':['FEP1','FEP5','FEP7','FEP9']}
scores = pd.DataFrame({k:df[v].mean(axis=1) for k,v in groups.items()})
X, y = scores[['GRS','GTD','GPA','GCM']], scores.FEP
rows=[]
for seed in [42,43,44,45,46]:
    xt,xv,yt,yv=train_test_split(X,y,test_size=.2,random_state=seed)
    models={'Mean':DummyRegressor(strategy='mean'), 'OLS':LinearRegression(),
       'RandomForest':GridSearchCV(RandomForestRegressor(random_state=42,n_jobs=1),
          {'n_estimators':[100,200,400],'max_depth':[None,5,10],'max_features':['sqrt',1.0]},
          scoring='neg_root_mean_squared_error',cv=KFold(5,shuffle=True,random_state=42),n_jobs=1)}
    for name,model in models.items():
        model.fit(xt,yt); pred=model.predict(xv)
        rows.append({'seed':seed,'model':name,'r2':float(r2_score(yv,pred)),
                     'rmse':float(np.sqrt(mean_squared_error(yv,pred))),
                     'mae':float(mean_absolute_error(yv,pred))})
    print('Completed GHRM split',seed,flush=True)
classification=[]
for file,target,name in [('WA_Fn-UseC_-HR-Employee-Attrition__3_.csv','Attrition','IBM'),
                         ('aug_train.csv','target','Job Change'),('train_LZdllcl.csv','is_promoted','Promotion')]:
    data=pd.read_csv(ROOT/'data'/file).dropna(subset=[target]); yc=data[target]
    if target=='Attrition':yc=(yc=='Yes').astype(int)
    yt,yv=train_test_split(yc,test_size=.2,random_state=42,stratify=yc)
    for strategy in ['most_frequent','stratified']:
        model=DummyClassifier(strategy=strategy,random_state=42).fit(np.zeros((len(yt),1)),yt)
        pred=model.predict(np.zeros((len(yv),1)))
        classification.append({'dataset':name,'strategy':strategy,'accuracy':float(accuracy_score(yv,pred)),
                               'f1':float(f1_score(yv,pred,zero_division=0))})
holm={}
for name in ['ibm_attrition','job_change','promotion']:
    report=json.loads((ROOT/'results'/f'{name}_classification_report.json').read_text())
    pairs=report['mcnemar_pairwise']; ordered=sorted(pairs,key=lambda k:pairs[k]['p_value']); running=0
    holm[name]={}
    for rank,key in enumerate(ordered):
        running=max(running,(len(ordered)-rank)*pairs[key]['p_value'])
        holm[name][key]={'p_raw':pairs[key]['p_value'],'p_holm':min(1.,running)}
out={'purpose':'Post-hoc sensitivity and baseline checks, not new confirmatory hypotheses',
     'split_seeds':[42,43,44,45,46], 'ghrm_rows':rows,'classification_dummy':classification,'mcnemar_holm':holm,
     'caution':'Repeated holdouts overlap; these descriptive ranges are not independent-sample confidence intervals.'}
(ROOT/'results/defense_checks.json').write_text(json.dumps(out,indent=2))
print(json.dumps(out,indent=2))
