import argparse, json, time
from datetime import datetime, timezone
from src.config import *
from src.reporting import dataset_inventory, write_manifest, write_validation_report
from src.classification import MODEL_ORDER, RF_CANDIDATES, MLP_CANDIDATES, run_model, consolidate_dataset, run_rf_candidate, finalize_rf, run_mlp_candidate, finalize_mlp
from src.survey_analysis import MODEL_ORDER as REG_MODEL_ORDER, VARIANTS, prepare_survey, run_survey_model, consolidate_survey, run_survey
from src.clustering import run_public_kmeans, run_ghrm_kmeans


def log(msg):
    print(msg,flush=True)
    RESULTS_DIR.mkdir(parents=True,exist_ok=True)
    with open(RESULTS_DIR/'execution_console.txt','a',encoding='utf-8') as f: f.write(str(msg)+'\n')

def prepare(clean=False):
    for d in (RESULTS_DIR,FIGURES_DIR,VALIDATION_DIR): d.mkdir(parents=True,exist_ok=True)
    if clean:
        for p in RESULTS_DIR.glob('*'): p.unlink() if p.is_file() else None
        for p in FIGURES_DIR.glob('*.png'): p.unlink()
        (VALIDATION_DIR/'validation_report.md').unlink(missing_ok=True)
    inv=dataset_inventory(); log('DATA: '+', '.join(f"{r['dataset']}={r['rows']} rows sha256={r['sha256']}" for r in inv))

def finalize():
    for n in DATASETS: consolidate_dataset(n,log)
    started=(RESULTS_DIR/'started_utc.txt').read_text().strip() if (RESULTS_DIR/'started_utc.txt').exists() else None
    finished=datetime.now(timezone.utc).isoformat(); write_manifest(started,finished,['python run_all.py --stage ...']); write_validation_report(); log(f'FINALIZE COMPLETE {finished}'); log('JSON: '+', '.join(sorted(p.name for p in RESULTS_DIR.glob('*.json')))); log('FIGURES: '+', '.join(sorted(p.name for p in FIGURES_DIR.glob('*.png')))); log('VALIDATION: validation/validation_report.md')

def run_everything():
    prepare(clean=True); (RESULTS_DIR/'started_utc.txt').write_text(datetime.now(timezone.utc).isoformat()); run_survey(log); run_ghrm_kmeans(log)
    for n in DATASETS:
        for m in MODEL_ORDER: run_model(n,m,log)
        run_public_kmeans(n,log)
    finalize()

def main():
    stages=['prepare','survey-prepare','survey-consolidate','ghrm-kmeans',*[f'ghrm:{v}:{m}' for v in VARIANTS for m in REG_MODEL_ORDER],*[f'{n}:{m}' for n in DATASETS for m in MODEL_ORDER], *[f'{n}:rf-c{i}' for n in DATASETS for i in range(len(RF_CANDIDATES))], *[f'{n}:rf-final' for n in DATASETS], *[f'{n}:mlp-c{i}' for n in DATASETS for i in range(len(MLP_CANDIDATES))], *[f'{n}:mlp-final' for n in DATASETS],*[f'{n}:kmeans' for n in DATASETS],'finalize']
    ap=argparse.ArgumentParser(); ap.add_argument('--stage',choices=stages); ap.add_argument('--clean',action='store_true'); args=ap.parse_args()
    if not args.stage: return run_everything()
    if args.stage=='prepare':
        prepare(clean=args.clean); (RESULTS_DIR/'started_utc.txt').write_text(datetime.now(timezone.utc).isoformat()); return
    if args.stage=='survey-prepare': return prepare_survey(log)
    if args.stage=='survey-consolidate': return consolidate_survey(log)
    if args.stage.startswith('ghrm:'):
        _,v,m=args.stage.split(':',2); return run_survey_model(v,m,log)
    if args.stage=='ghrm-kmeans': return run_ghrm_kmeans(log)
    if args.stage=='finalize': return finalize()
    n,thing=args.stage.split(':',1)
    if thing=='kmeans': return run_public_kmeans(n,log)
    if thing.startswith('rf-c'): return run_rf_candidate(n,int(thing[4:]),log)
    if thing=='rf-final': return finalize_rf(n,log)
    if thing.startswith('mlp-c'): return run_mlp_candidate(n,int(thing[5:]),log)
    if thing=='mlp-final': return finalize_mlp(n,log)
    return run_model(n,thing,log)
if __name__=='__main__': main()
