from __future__ import annotations
import argparse, json, sys
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

ROOT=Path(__file__).resolve().parents[2]; sys.path.insert(0,str(ROOT/"src"))
from train import OBSERVATION_END, PREDICTION_END, NUMERIC_FEATURES, clean_transactions, build_customer_dataset, IQRClipper


def build_regression_data(clean):
    base=build_customer_dataset(clean).drop(columns="repeat_purchase")
    future=clean.loc[clean.InvoiceDate.gt(OBSERVATION_END)&clean.InvoiceDate.le(PREDICTION_END)].groupby("CustomerID").line_value.sum().rename("future_spend")
    return base.merge(future,how="left",left_on="CustomerID",right_index=True).fillna({"future_spend":0.0})


def make_preprocessor():
    num=Pipeline([("imputer",SimpleImputer(strategy="median")),("outliers",IQRClipper()),("scaler",StandardScaler())])
    cat=Pipeline([("imputer",SimpleImputer(strategy="most_frequent")),("encoder",OneHotEncoder(handle_unknown="ignore",sparse_output=False))])
    return ColumnTransformer([("numeric",num,NUMERIC_FEATURES),("categorical",cat,["Country"])])


def run(input_path,output_dir,random_state=42):
    output_dir.mkdir(parents=True,exist_ok=True)
    clean,quality=clean_transactions(pd.read_excel(input_path,sheet_name="Online Retail")); data=build_regression_data(clean)
    x=data[[*NUMERIC_FEATURES,"Country"]]; y=data.future_spend
    xtr,xte,ytr,yte=train_test_split(x,y,test_size=.2,random_state=random_state)
    model=Pipeline([("preprocessing",make_preprocessor()),("model",LinearRegression())]); model.fit(xtr,ytr); pred=np.maximum(model.predict(xte),0)
    numeric_pipeline=model.named_steps["preprocessing"].named_transformers_["numeric"]
    imputer=numeric_pipeline.named_steps["imputer"]
    clipper=numeric_pipeline.named_steps["outliers"]
    train_numeric=imputer.transform(xtr[NUMERIC_FEATURES]); test_numeric=imputer.transform(xte[NUMERIC_FEATURES])
    train_outliers=(train_numeric<clipper.lower_)|(train_numeric>clipper.upper_); test_outliers=(test_numeric<clipper.lower_)|(test_numeric>clipper.upper_)
    outlier_report=pd.DataFrame({"feature":NUMERIC_FEATURES,"lower_iqr_limit":clipper.lower_,"upper_iqr_limit":clipper.upper_,"training_values_capped":train_outliers.sum(axis=0),"test_values_capped":test_outliers.sum(axis=0)})
    outlier_report.to_csv(output_dir/"outlier_report.csv",index=False)
    mse=mean_squared_error(yte,pred); metrics={"customers":len(data),"train_customers":len(xtr),"test_customers":len(xte),"outlier_method":"1.5 x IQR capping fitted on training features","training_feature_values_capped":int(train_outliers.sum()),"test_feature_values_capped":int(test_outliers.sum()),"mae":mean_absolute_error(yte,pred),"mse":mse,"rmse":mse**.5,"r2":r2_score(yte,pred),"data_quality":quality}
    residual=yte-pred
    (output_dir/"metrics.json").write_text(json.dumps(metrics,indent=2),encoding="utf-8"); pd.DataFrame({"actual":yte,"predicted":pred,"residual":residual}).to_csv(output_dir/"predictions.csv",index=False)

    with plt.style.context("seaborn-v0_8-whitegrid"):
        fig,ax=plt.subplots(figsize=(9,6),facecolor="#f8fafc")
        ax.set_facecolor("white")
        points=ax.scatter(
            pred,
            residual,
            c=np.log1p(np.abs(residual)),
            cmap="viridis",
            s=30,
            alpha=.7,
            edgecolors="white",
            linewidths=.35,
        )
        ax.axhspan(-metrics["mae"],metrics["mae"],color="#10b981",alpha=.1,label=f"MAE range (+/- {metrics['mae']:.0f})")
        ax.axhline(0,color="#0f172a",lw=1.4,ls="--")
        ax.set_xscale("symlog",linthresh=100)
        ax.set_yscale("symlog",linthresh=100)
        ax.set(
            xlabel="Predicted future spend (symmetric log scale)",
            ylabel="Residual: actual - predicted (symmetric log scale)",
        )
        ax.set_xlim(left=0)
        fig.suptitle("Linear regression residuals",fontsize=17,fontweight="bold",y=.98)
        fig.text(.5,.935,"All test customers remain visible; logarithmic scaling separates the dense centre.",ha="center",color="#475569",fontsize=10)
        ax.legend(loc="lower left",frameon=False)
        colorbar=fig.colorbar(points,ax=ax,pad=.02)
        colorbar.set_label("Error magnitude: log(1 + |residual|)")
        ax.grid(True,which="major",color="#e2e8f0",linewidth=.8)
        ax.grid(False,which="minor")
        fig.tight_layout(rect=(0,0,1,.92))
        fig.savefig(output_dir/"residuals.png",dpi=180,bbox_inches="tight")
        plt.close(fig)

        fig,ax=plt.subplots(figsize=(7,6),facecolor="#f8fafc")
        ax.set_facecolor("white")
        ax.scatter(yte,pred,s=30,alpha=.65,color="#2563eb",edgecolors="white",linewidths=.35)
        limit=float(max(yte.max(),pred.max()))
        ax.plot([0,limit],[0,limit],ls="--",lw=1.5,color="#0f172a",label="Ideal prediction")
        ax.set_xscale("symlog",linthresh=100); ax.set_yscale("symlog",linthresh=100)
        ax.set(xlabel="Actual future spend (symmetric log scale)",ylabel="Predicted future spend (symmetric log scale)",title="Linear regression: actual versus predicted")
        ax.legend(frameon=False); ax.grid(True,which="major",color="#e2e8f0",linewidth=.8); ax.grid(False,which="minor")
        fig.tight_layout(); fig.savefig(output_dir/"actual_vs_predicted.png",dpi=180,bbox_inches="tight"); plt.close(fig)

    feature_names=model.named_steps["preprocessing"].get_feature_names_out()
    pd.DataFrame({"feature":feature_names,"coefficient":model.named_steps["model"].coef_}).assign(absolute_coefficient=lambda d: d.coefficient.abs()).sort_values("absolute_coefficient",ascending=False).to_csv(output_dir/"coefficients.csv",index=False)
    return metrics


if __name__=="__main__":
    p=argparse.ArgumentParser(); p.add_argument("--input",type=Path,default=ROOT/"data"/"Online Retail 2.xlsx",help="Path to Online Retail 2.xlsx"); p.add_argument("--output-dir",type=Path,default=Path(__file__).parent/"outputs"); a=p.parse_args(); print(json.dumps(run(a.input,a.output_dir),indent=2))
