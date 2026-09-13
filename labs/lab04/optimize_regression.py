from __future__ import annotations
import argparse,json,sys
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np,pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor,RandomForestRegressor
from sklearn.inspection import permutation_importance
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error,mean_squared_error,r2_score
from sklearn.model_selection import GridSearchCV,KFold,cross_validate,train_test_split
from sklearn.neighbors import KNeighborsRegressor
from sklearn.pipeline import Pipeline
ROOT=Path(__file__).resolve().parents[2]; sys.path.insert(0,str(ROOT/"src")); sys.path.insert(0,str(ROOT/"labs"/"lab03"))
from train import NUMERIC_FEATURES,clean_transactions
from regression import build_regression_data,make_preprocessor


def run(input_path,output_dir,random_state=42):
 output_dir.mkdir(parents=True,exist_ok=True); clean,_=clean_transactions(pd.read_excel(input_path,sheet_name="Online Retail")); data=build_regression_data(clean); x=data[[*NUMERIC_FEATURES,"Country"]]; y=np.log1p(data.future_spend); xtr,xte,ytr,yte=train_test_split(x,y,test_size=.2,random_state=random_state); cv=KFold(5,shuffle=True,random_state=random_state)
 models={"ridge":Ridge(),"random_forest":RandomForestRegressor(n_estimators=250,random_state=random_state,n_jobs=-1),"hist_gradient_boosting":HistGradientBoostingRegressor(random_state=random_state),"knn":KNeighborsRegressor(n_neighbors=15,weights="distance")}; rows=[]
 for name,est in models.items():
  pipe=Pipeline([("preprocessing",make_preprocessor()),("model",est)]); scores=cross_validate(pipe,xtr,ytr,scoring={"mae":"neg_mean_absolute_error","rmse":"neg_root_mean_squared_error","r2":"r2"},cv=cv,n_jobs=-1); rows.append({"model":name,"cv_mae_log_mean":-scores["test_mae"].mean(),"cv_mae_log_std":scores["test_mae"].std(),"cv_rmse_log_mean":-scores["test_rmse"].mean(),"cv_rmse_log_std":scores["test_rmse"].std(),"cv_r2_log_mean":scores["test_r2"].mean(),"cv_r2_log_std":scores["test_r2"].std()})
 comp=pd.DataFrame(rows).sort_values("cv_rmse_log_mean"); comp.to_csv(output_dir/"model_comparison.csv",index=False); best=comp.iloc[0].model
 grids={"ridge":{"model__alpha":[.01,.1,1,10,100]},"random_forest":{"model__n_estimators":[200,400],"model__max_depth":[None,8,16],"model__min_samples_leaf":[1,4,10],"model__max_features":["sqrt",.8]},"hist_gradient_boosting":{"model__learning_rate":[.03,.08,.15],"model__max_iter":[100,250],"model__max_leaf_nodes":[15,31]},"knn":{"model__n_neighbors":[7,15,25,41],"model__weights":["uniform","distance"],"model__p":[1,2]}}
 search=GridSearchCV(Pipeline([("preprocessing",make_preprocessor()),("model",models[best])]),grids[best],scoring="neg_root_mean_squared_error",cv=cv,n_jobs=-1); search.fit(xtr,ytr); pred=np.maximum(np.expm1(search.predict(xte)),0); actual=np.expm1(yte); mse=mean_squared_error(actual,pred); metrics={"mae":mean_absolute_error(actual,pred),"mse":mse,"rmse":mse**.5,"r2":r2_score(actual,pred)}
 imp=permutation_importance(search.best_estimator_,xte,yte,scoring="neg_root_mean_squared_error",n_repeats=10,random_state=random_state,n_jobs=-1); pd.DataFrame({"feature":x.columns,"importance_mean":imp.importances_mean,"importance_std":imp.importances_std}).sort_values("importance_mean",ascending=False).to_csv(output_dir/"feature_importance.csv",index=False)
 residual=actual-pred; residual_diagnostics={"mean":float(residual.mean()),"median":float(np.median(residual)),"standard_deviation":float(residual.std()),"skewness":float(pd.Series(residual).skew()),"correlation_with_prediction":float(np.corrcoef(pred,residual)[0,1]),"underprediction_rate":float((residual>0).mean()),"zero_actual_rate":float((actual==0).mean())}
 res={"selected_model":best,"best_parameters":search.best_params_,"best_cv_rmse_log":-search.best_score_,"test_metrics":metrics,"residual_diagnostics":residual_diagnostics}; (output_dir/"results.json").write_text(json.dumps(res,indent=2),encoding="utf-8"); pd.DataFrame({"actual":actual,"predicted":pred,"residual":residual}).to_csv(output_dir/"predictions.csv",index=False)
 fig,axs=plt.subplots(1,2,figsize=(11,4.8)); axs[0].scatter(pred,residual,alpha=.5,s=18,color="#2878b5"); axs[0].axhline(0,color="black",ls="--"); axs[0].set_xscale("symlog",linthresh=100); axs[0].set_yscale("symlog",linthresh=100); axs[0].set(xlabel="Predicted spend (symmetric log scale)",ylabel="Residual (symmetric log scale)",title="Residuals versus predictions"); low,high=np.quantile(residual,[.01,.99]); axs[1].hist(residual[(residual>=low)&(residual<=high)],bins=35,color="#2878b5"); axs[1].set(xlabel="Residual (central 98%)",ylabel="Customers",title="Residual distribution"); fig.tight_layout(); fig.savefig(output_dir/"residual_analysis.png",dpi=160); plt.close(fig); return res


if __name__=="__main__":
 p=argparse.ArgumentParser(); p.add_argument("--input",type=Path,default=ROOT/"data"/"Online Retail 2.xlsx",help="Path to Online Retail 2.xlsx"); p.add_argument("--output-dir",type=Path,default=Path(__file__).parent/"outputs"); a=p.parse_args(); print(json.dumps(run(a.input,a.output_dir),indent=2))
