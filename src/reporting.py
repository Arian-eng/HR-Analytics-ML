import hashlib, json, platform, sys
from importlib.metadata import version
import pandas as pd
from .config import *


def sha256(path):
    h=hashlib.sha256()
    with open(path,'rb') as f:
        for b in iter(lambda:f.read(1024*1024),b''): h.update(b)
    return h.hexdigest()

def dataset_inventory():
    items=[]
    for name,info in DATASETS.items():
        p=DATA_DIR/info['file']; df=pd.read_csv(p); items.append({'dataset':name,'file':info['file'],'rows':len(df),'columns':len(df.columns),'missing_cells':int(df.isna().sum().sum()),'duplicate_rows':int(df.duplicated().sum()),'target_missing':int(df[info['target']].isna().sum()),'sha256':sha256(p),'role':info['role']})
    p=DATA_DIR/SURVEY_FILE; df=pd.read_csv(p); items.append({'dataset':'ghrm','file':SURVEY_FILE,'rows':len(df),'columns':len(df.columns),'missing_cells':int(df.isna().sum().sum()),'duplicate_rows':int(df.duplicated().sum()),'target_missing':None,'sha256':sha256(p),'role':'direct Green HRM survey evidence'})
    (RESULTS_DIR/'dataset_inventory.json').write_text(json.dumps(items,indent=2),encoding='utf-8'); return items

def write_manifest(started,finished,commands):
    inv=json.loads((RESULTS_DIR/'dataset_inventory.json').read_text()); manifest={'started_utc':started,'finished_utc':finished,'commands':commands,'random_state':RANDOM_STATE,'test_size':TEST_SIZE,'classification_cv_folds':CLASSIFICATION_CV_FOLDS,'regression_cv_folds':REGRESSION_CV_FOLDS,'classification_bootstraps':CLASSIFICATION_BOOTSTRAPS,'regression_bootstraps':REGRESSION_BOOTSTRAPS,'kmeans':{'k_values':list(K_VALUES),'n_init':KMEANS_N_INIT,'max_iter':KMEANS_MAX_ITER,'silhouette_sample_size_if_large':SILHOUETTE_SAMPLE_SIZE},'python':platform.python_version(),'packages':{p:version(p) for p in ['numpy','pandas','scikit-learn','scipy','matplotlib']},'datasets':inv}
    (RESULTS_DIR/'run_manifest.json').write_text(json.dumps(manifest,indent=2),encoding='utf-8'); return manifest

def write_validation_report():
    inv=json.loads((RESULTS_DIR/'dataset_inventory.json').read_text()); g=json.loads((RESULTS_DIR/'ghrm.json').read_text()); cls={n:json.loads((RESULTS_DIR/f'classification_{n}.json').read_text()) for n in DATASETS}; kms={n:json.loads((RESULTS_DIR/f'clustering_{n}.json').read_text()) for n in [*DATASETS,'ghrm']}
    L=['# Validation report — executed results only','', 'All numbers below are read from artifacts produced by the executed pipeline. The three public HR datasets are benchmark/proxy analyses and are **not** direct Green HRM evidence. The GHRM survey is the only direct Green HRM dataset in this repository.','', '## Data identity','', '| Dataset | Rows | Columns | Missing cells | SHA-256 |','|---|---:|---:|---:|---|']
    for r in inv: L.append(f"| {r['dataset']} | {r['rows']} | {r['columns']} | {r['missing_cells']} | `{r['sha256']}` |")
    L += ['', '## Classification — held-out 20% test set', '', '95% percentile bootstrap CI uses 2,000 resamples with seed 42. Model selection uses stratified 3-fold CV and positive-class F1.','', '| Dataset | Model | Accuracy | Precision | Recall | F1 | 95% CI F1 | CV F1 |','|---|---|---:|---:|---:|---:|---|---:|']
    for n,o in cls.items():
        for r in o['models']:
            m=r['metrics']; ci=r['bootstrap_95_ci']['f1']; L.append(f"| {n} | {r['model']} | {m['accuracy']:.4f} | {m['precision']:.4f} | {m['recall']:.4f} | {m['f1']:.4f} | [{ci['lower']:.4f}, {ci['upper']:.4f}] | {r['best_cv_f1']:.4f} |")
    L += ['', '## McNemar — all six model pairs per classification dataset','', '| Dataset | Pair | b01 | b10 | Method | p-value | Significant 0.05 |','|---|---|---:|---:|---|---:|---|']
    for n,o in cls.items():
        for r in o['mcnemar']: L.append(f"| {n} | {r['model_a']} / {r['model_b']} | {r['b01']} | {r['b10']} | {r['method']} | {r['p_value']:.6g} | {r['significant_0_05']} |")
    L += ['', '## GHRM regression — same 256/64 split for Base and GEE+','', '95% percentile bootstrap CI uses 4,000 resamples with seed 42. Hyperparameters are selected only inside the 256-row training set by 5-fold CV minimizing RMSE.','', '| Variant | Model | R² | MAE | RMSE | 95% CI RMSE | CV RMSE |','|---|---|---:|---:|---:|---|---:|']
    for v in ['Base','GEE+']:
        for r in g['regression'][v]:
            m=r['metrics']; ci=r['bootstrap_95_ci']['rmse']; L.append(f"| {v} | {r['model']} | {m['r2']:.4f} | {m['mae']:.4f} | {m['rmse']:.4f} | [{ci['lower']:.4f}, {ci['upper']:.4f}] | {r['best_cv_rmse']:.4f} |")
    L += ['', '### Paired Base → GEE+ bootstrap differences on the same 64 test cases','', '| Model | ΔR² (95% CI) | ΔMAE (95% CI) | ΔRMSE (95% CI) | Reliable change? |','|---|---|---|---|---|']
    for model,d in g['paired_base_vs_gee_plus'].items():
        f=lambda k:f"{d[k]['delta']:.4f} [{d[k]['lower']:.4f}, {d[k]['upper']:.4f}]"; rel=any(d[k]['reliable_change'] for k in ['r2','mae','rmse']); L.append(f"| {model} | {f('r2')} | {f('mae')} | {f('rmse')} | {rel} |")
    L += ['', '## GHRM construct reliability','', '| Construct | Cronbach alpha | Mean | SD |','|---|---:|---:|---:|']
    for c,r in g['reliability'].items(): L.append(f"| {c} | {r['alpha']:.4f} | {r['mean']:.4f} | {r['std']:.4f} |")
    L += ['', '## K-Means — k=2..7, target/IDs excluded','', '| Dataset | Numeric/construct features | Selected k | Silhouette | SSE | Davies-Bouldin |','|---|---:|---:|---:|---:|---:|']
    for n,o in kms.items():
        r=next(x for x in o['k_results'] if x['k']==o['selected_k']); fc=o.get('numeric_feature_count',len(o.get('features',[]))); L.append(f"| {n} | {fc} | {o['selected_k']} | {r['silhouette']:.4f} | {r['sse']:.3f} | {r['davies_bouldin']:.4f} |")
    L += ['', '## Scope and limitations','', '- The 320-row GHRM survey is the only dataset with direct GHRM constructs; results should not be generalized beyond its sampling frame without external validation.', '- IBM, Job Change, and Promotion are general HR benchmark/proxy datasets. Their outputs do not directly test Green HRM hypotheses.', '- K-Means is exploratory. Cluster membership and model importance/predictive performance are not causal effects.', '- A Base→GEE+ change is treated as reliable only when the paired-bootstrap interval excludes zero.']
    VALIDATION_DIR.mkdir(parents=True,exist_ok=True); (VALIDATION_DIR/'validation_report.md').write_text('\n'.join(L)+'\n',encoding='utf-8')
