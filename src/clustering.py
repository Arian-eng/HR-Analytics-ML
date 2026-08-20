from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.metrics import silhouette_score, davies_bouldin_score

def _matrix(df, drop_cols):
    X = df.drop(columns=drop_cols, errors="ignore").copy()
    cat = X.select_dtypes(include=["object", "category", "bool"]).columns.tolist()
    num = [c for c in X.columns if c not in cat]
    tr = []
    if num: tr.append(("num", Pipeline([("impute", SimpleImputer(strategy="median")), ("scale", StandardScaler())]), num))
    if cat: tr.append(("cat", Pipeline([("impute", SimpleImputer(strategy="most_frequent")), ("onehot", OneHotEncoder(handle_unknown="ignore"))]), cat))
    return ColumnTransformer(tr).fit_transform(X)

def run_kmeans(df, target, ids, out_dir, figure_dir, name, extra_drop=None):
    drop = list(ids) + [target] + (extra_drop or [])
    Z = _matrix(df, drop)
    rows=[]
    for k in range(2,8):
        km=KMeans(n_clusters=k,n_init=20,max_iter=300,random_state=42).fit(Z)
        labels=km.labels_
        dense=Z.toarray() if hasattr(Z,"toarray") else Z
        rows.append({"Dataset":name,"k":k,"Inertia_SSE":km.inertia_,"Silhouette":silhouette_score(Z,labels),"Davies_Bouldin":davies_bouldin_score(dense,labels)})
    r=pd.DataFrame(rows); r.to_csv(Path(out_dir)/"kmeans_metrics.csv",index=False,encoding="utf-8-sig")
    Path(figure_dir).mkdir(parents=True,exist_ok=True)
    for col,title,ylabel in [("Inertia_SSE","K-Means Elbow","Inertia / SSE"),("Silhouette","K-Means Silhouette","Silhouette Score"),("Davies_Bouldin","K-Means Davies-Bouldin","Davies-Bouldin Index")]:
        fig,ax=plt.subplots(); ax.plot(r.k,r[col],marker="o"); ax.set_title(title); ax.set_xlabel("k"); ax.set_ylabel(ylabel); fig.tight_layout(); fig.savefig(Path(figure_dir)/f"{name}_{col}.png",dpi=180); plt.close(fig)
    best=int(r.loc[r.Silhouette.idxmax(),"k"])
    km=KMeans(n_clusters=best,n_init=20,max_iter=300,random_state=42).fit(Z)
    sizes=pd.Series(km.labels_).value_counts().sort_index().rename_axis("Cluster").reset_index(name="Size")
    sizes.to_csv(Path(out_dir)/"cluster_sizes.csv",index=False,encoding="utf-8-sig")
    return r,best,km.labels_
