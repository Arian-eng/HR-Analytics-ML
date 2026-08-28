"""
Employee Promotion (n=54,808) - Chapter 3 compliant pipeline.

This is the largest dataset (43,846 train / 10,962 test after the 80/20
split) and Random Forest's grid search alone takes several minutes on a
single CPU. To keep each step within a practical runtime budget, this
dataset's classification suite is split into 3 sub-scripts rather than one
monolithic run - the logic is identical to run_ibm.py / run_jobchange.py,
just executed in stages:

    python _promotion_step1_per_model.py DecisionTree
    python _promotion_step1_per_model.py RandomForest
    python _promotion_step1_per_model.py LinearSVC
    python _promotion_step1_per_model.py MLPClassifier
    python _promotion_step2_combine.py       # McNemar + confusion matrices
    python _promotion_step3_tree_and_kmeans.py   # tree plot + K-Means

Each step writes its output under results/ and figures/ exactly like the
other datasets' single-script runs. This file just documents/runs that
sequence; run it with `python run_promotion.py` to execute all steps.
"""
import subprocess
import sys

STEPS = [
    ["_promotion_step1_per_model.py", "DecisionTree"],
    ["_promotion_step1_per_model.py", "RandomForest"],
    ["_promotion_step1_per_model.py", "LinearSVC"],
    ["_promotion_step1_per_model.py", "MLPClassifier"],
    ["_promotion_step2_combine.py"],
    ["_promotion_step3_tree_and_kmeans.py"],
]

if __name__ == "__main__":
    for step in STEPS:
        print(f"\n{'='*70}\nRunning: {' '.join(step)}\n{'='*70}")
        subprocess.run([sys.executable] + step, check=True, cwd="/home/claude/repo2/src")
