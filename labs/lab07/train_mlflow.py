from __future__ import annotations
import argparse,json,sys
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import f1_score,roc_auc_score
from sklearn.model_selection import train_test_split
ROOT=Path(__file__).resolve().parents[2]; sys.path.insert(0,str(ROOT/"src"))
from train import CATEGORICAL_FEATURES,NUMERIC_FEATURES,build_customer_dataset,clean_transactions,make_model
import mlflow
import mlflow.sklearn


def run(input_path,output_dir,tracking_uri,random_state=42):
 output_dir.mkdir(parents=True,exist_ok=True); mlflow.set_tracking_uri(tracking_uri); experiment_name="online-retail-repeat-purchase"; experiment=mlflow.get_experiment_by_name(experiment_name)
 if experiment is None: mlflow.create_experiment(experiment_name,artifact_location=(Path(__file__).parent/"mlruns").resolve().as_uri())
 mlflow.set_experiment(experiment_name); clean,_=clean_transactions(pd.read_excel(input_path,sheet_name="Online Retail")); data=build_customer_dataset(clean); x=data[[*NUMERIC_FEATURES,*CATEGORICAL_FEATURES]]; y=data.repeat_purchase; xtr,xte,ytr,yte=train_test_split(x,y,test_size=.2,stratify=y,random_state=random_state); rows=[]
 for c in [.1,1.0,10.0]:
  model=make_model(random_state); model.set_params(classifier__C=c); model.fit(xtr,ytr); prob=model.predict_proba(xte)[:,1]; pred=(prob>=.5).astype(int); metrics={"roc_auc":roc_auc_score(yte,prob),"f1":f1_score(yte,pred)}
  prediction_file=output_dir/f"predictions_c_{c}.csv"; pd.DataFrame({"actual":yte.to_numpy(),"predicted":pred,"probability":prob}).to_csv(prediction_file,index=False)
  with mlflow.start_run(run_name=f"logistic_C_{c}"):
   mlflow.log_params({"C":c,"class_weight":"balanced","random_state":random_state,"test_size":.2}); mlflow.log_metrics(metrics); mlflow.log_artifact(str(prediction_file),artifact_path="evaluation"); mlflow.sklearn.log_model(model,name="model",serialization_format="cloudpickle")
  rows.append({"C":c,**metrics})
 comparison=pd.DataFrame(rows).sort_values("roc_auc",ascending=False); comparison.to_csv(output_dir/"run_comparison.csv",index=False); result=comparison.iloc[0].to_dict(); (output_dir/"best_run.json").write_text(json.dumps(result,indent=2),encoding="utf-8"); return result


if __name__=="__main__":
 p=argparse.ArgumentParser(); p.add_argument("--input",type=Path,default=ROOT/"data"/"Online Retail 2.xlsx",help="Path to Online Retail 2.xlsx"); p.add_argument("--output-dir",type=Path,default=Path(__file__).parent/"outputs"); p.add_argument("--tracking-uri",default=f"sqlite:///{(Path(__file__).parent/'mlflow.db').resolve().as_posix()}"); a=p.parse_args(); print(json.dumps(run(a.input,a.output_dir,a.tracking_uri),indent=2))
