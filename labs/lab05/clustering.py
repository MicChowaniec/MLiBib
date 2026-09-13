from __future__ import annotations
import argparse,json,sys
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np,pandas as pd
from sklearn.cluster import AgglomerativeClustering,KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import calinski_harabasz_score,davies_bouldin_score,silhouette_score
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler
ROOT=Path(__file__).resolve().parents[2]; sys.path.insert(0,str(ROOT/"src"))
from train import clean_transactions

FEATURES=["recency_days","frequency","items_purchased","monetary_value","mean_order_value","unique_products"]


def customer_features(clean):
 end=clean.InvoiceDate.max(); invoices=clean.groupby(["CustomerID","InvoiceNo"],as_index=False).agg(order_value=("line_value","sum")); orders=invoices.groupby("CustomerID").agg(frequency=("InvoiceNo","nunique"),mean_order_value=("order_value","mean")); customers=clean.groupby("CustomerID").agg(last_purchase=("InvoiceDate","max"),items_purchased=("Quantity","sum"),monetary_value=("line_value","sum"),unique_products=("StockCode","nunique")); customers=customers.join(orders); customers["recency_days"]=(end-customers.last_purchase).dt.days; return customers[FEATURES].reset_index()


def score(name,labels,x):
 valid=labels>=0; unique=np.unique(labels[valid]);
 if len(unique)<2:return {"model":name,"clusters":len(unique),"silhouette":None,"calinski_harabasz":None,"davies_bouldin":None}
 return {"model":name,"clusters":len(unique),"silhouette":silhouette_score(x[valid],labels[valid]),"calinski_harabasz":calinski_harabasz_score(x[valid],labels[valid]),"davies_bouldin":davies_bouldin_score(x[valid],labels[valid])}


def run(input_path,output_dir,random_state=42):
 output_dir.mkdir(parents=True,exist_ok=True); clean,quality=clean_transactions(pd.read_excel(input_path,sheet_name="Online Retail")); data=customer_features(clean); x=StandardScaler().fit_transform(np.log1p(data[FEATURES].clip(lower=0)))
 selection=[]
 for k in range(2,9):
  labels=KMeans(n_clusters=k,n_init=20,random_state=random_state).fit_predict(x); selection.append({"k":k,"silhouette":silhouette_score(x,labels),"calinski_harabasz":calinski_harabasz_score(x,labels),"davies_bouldin":davies_bouldin_score(x,labels)})
 select=pd.DataFrame(selection); select.to_csv(output_dir/"k_selection.csv",index=False); k=int(select.sort_values(["silhouette","davies_bouldin"],ascending=[False,True]).iloc[0].k)
 labels={"kmeans":KMeans(n_clusters=k,n_init=30,random_state=random_state).fit_predict(x),"gaussian_mixture":GaussianMixture(n_components=k,random_state=random_state,n_init=5).fit_predict(x),"agglomerative":AgglomerativeClustering(n_clusters=k).fit_predict(x)}
 comparison=pd.DataFrame([score(n,l,x) for n,l in labels.items()]).sort_values("silhouette",ascending=False); comparison.to_csv(output_dir/"model_comparison.csv",index=False); best=comparison.iloc[0].model; data["cluster"]=labels[best]; data.to_csv(output_dir/"customer_clusters.csv",index=False)
 profiles=data.groupby("cluster").agg(customers=("CustomerID","size"))
 for feature in FEATURES:
  profiles[f"{feature}_mean"]=data.groupby("cluster")[feature].mean(); profiles[f"{feature}_median"]=data.groupby("cluster")[feature].median()
 profiles["customer_share"]=profiles.customers/len(data); profiles.reset_index().to_csv(output_dir/"cluster_profiles.csv",index=False)
 pca=PCA(n_components=2,random_state=random_state); coords=pca.fit_transform(x); pd.DataFrame({"CustomerID":data.CustomerID,"PC1":coords[:,0],"PC2":coords[:,1],"cluster":data.cluster}).to_csv(output_dir/"pca_coordinates.csv",index=False)
 fig,ax=plt.subplots(figsize=(7,5))
 palette=plt.get_cmap("tab10")
 for cluster in sorted(data.cluster.unique()):
  mask=data.cluster.eq(cluster).to_numpy()
  ax.scatter(coords[mask,0],coords[mask,1],color=palette(int(cluster)%10),s=16,alpha=.65,label=f"Cluster {cluster}")
 ax.set(xlabel=f"PC1 ({pca.explained_variance_ratio_[0]:.1%})",ylabel=f"PC2 ({pca.explained_variance_ratio_[1]:.1%})",title=f"Customer clusters: {best}")
 ax.legend(title="Cluster",frameon=False)
 fig.tight_layout(); fig.savefig(output_dir/"clusters_pca.png",dpi=160); plt.close(fig)
 result={"customers":len(data),"selected_k":k,"selected_model":best,"pca_explained_variance":pca.explained_variance_ratio_.tolist(),"data_quality":quality}; (output_dir/"results.json").write_text(json.dumps(result,indent=2),encoding="utf-8"); return result


if __name__=="__main__":
 p=argparse.ArgumentParser(); p.add_argument("--input",type=Path,default=ROOT/"data"/"Online Retail 2.xlsx",help="Path to Online Retail 2.xlsx"); p.add_argument("--output-dir",type=Path,default=Path(__file__).parent/"outputs"); a=p.parse_args(); print(json.dumps(run(a.input,a.output_dir),indent=2))
