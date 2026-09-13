from __future__ import annotations
import argparse,json,random
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np,pandas as pd
from PIL import Image
import torch
from torch import nn
from torch.optim import Adam
from torch.utils.data import DataLoader,Dataset
from torchvision import transforms
from sklearn.metrics import accuracy_score,confusion_matrix,f1_score,precision_score,recall_score,roc_auc_score


class PCamDataset(Dataset):
 def __init__(self,csv_file,transform=None): self.annotations=pd.read_csv(csv_file); self.root=Path(csv_file).parent.parent; self.transform=transform
 def __len__(self): return len(self.annotations)
 def __getitem__(self,idx):
  row=self.annotations.iloc[idx]; path=self.root/Path(row.filename); image=Image.open(path).convert("RGB"); image=self.transform(image) if self.transform else image; return image,torch.tensor(float(row.label),dtype=torch.float32)


class SimpleCNN(nn.Module):
 def __init__(self):
  super().__init__(); self.conv1=nn.Conv2d(3,32,3,padding=1); self.conv2=nn.Conv2d(32,64,3,padding=1); self.conv3=nn.Conv2d(64,128,3,padding=1); self.pool=nn.MaxPool2d(2,2); self.fc1=nn.Linear(128*12*12,256); self.fc2=nn.Linear(256,1)
 def forward(self,x): x=self.pool(torch.relu(self.conv1(x))); x=self.pool(torch.relu(self.conv2(x))); x=self.pool(torch.relu(self.conv3(x))); x=torch.flatten(x,1); x=torch.relu(self.fc1(x)); return torch.sigmoid(self.fc2(x)).squeeze(1)


def run(data_dir,output_dir,epochs=5,random_state=42):
 torch.manual_seed(random_state); np.random.seed(random_state); random.seed(random_state); data=Path(data_dir); output_dir.mkdir(parents=True,exist_ok=True)
 train_transform=transforms.Compose([transforms.Resize((96,96)),transforms.RandomHorizontalFlip(),transforms.RandomRotation(15),transforms.ColorJitter(brightness=.2,contrast=.2),transforms.ToTensor(),transforms.Normalize([.5]*3,[.5]*3)])
 val_test_transform=transforms.Compose([transforms.Resize((96,96)),transforms.ToTensor(),transforms.Normalize([.5]*3,[.5]*3)])
 train_dataset=PCamDataset(data/"train_labels.csv",train_transform); val_dataset=PCamDataset(data/"validation_labels.csv",val_test_transform); test_dataset=PCamDataset(data/"test_labels.csv",val_test_transform)
 train_dataloader=DataLoader(train_dataset,batch_size=8,shuffle=True); val_dataloader=DataLoader(val_dataset,batch_size=32,shuffle=False); test_dataloader=DataLoader(test_dataset,batch_size=32,shuffle=False)
 device=torch.device("cuda" if torch.cuda.is_available() else "cpu"); cnn_model=SimpleCNN().to(device); criterion=nn.BCELoss(); optimizer=Adam(cnn_model.parameters(),lr=.0005); train_losses=[]; val_losses=[]
 for _ in range(epochs):
  cnn_model.train(); total_train_loss=0.0
  for images,labels in train_dataloader:
   images,labels=images.to(device),labels.to(device); optimizer.zero_grad(); outputs=cnn_model(images); loss=criterion(outputs,labels); loss.backward(); optimizer.step(); total_train_loss+=loss.item()*len(labels)
  cnn_model.eval(); total_val_loss=0.0
  with torch.no_grad():
   for images,labels in val_dataloader:
    images,labels=images.to(device),labels.to(device); total_val_loss+=criterion(cnn_model(images),labels).item()*len(labels)
  train_losses.append(total_train_loss/len(train_dataset)); val_losses.append(total_val_loss/len(val_dataset))
 test_pred_probs=[]; test_pred_labels=[]; actual=[]; cnn_model.eval()
 with torch.no_grad():
  for images,labels in test_dataloader:
   outputs=cnn_model(images.to(device)); test_pred_probs.extend(outputs.cpu().numpy()); test_pred_labels.extend(torch.round(outputs).cpu().numpy()); actual.extend(labels.numpy())
 probs=np.asarray(test_pred_probs); pred=np.asarray(test_pred_labels,dtype=int); actual=np.asarray(actual,dtype=int); metrics={"device":str(device),"epochs":epochs,"train_samples":len(train_dataset),"validation_samples":len(val_dataset),"test_samples":len(test_dataset),"accuracy":accuracy_score(actual,pred),"precision":precision_score(actual,pred,zero_division=0),"recall":recall_score(actual,pred,zero_division=0),"f1":f1_score(actual,pred,zero_division=0),"roc_auc":roc_auc_score(actual,probs),"confusion_matrix":confusion_matrix(actual,pred).tolist()}
 torch.save(cnn_model.state_dict(),output_dir/"cnn_model.pt"); pd.DataFrame({"actual":actual,"predicted":pred,"probability":probs}).to_csv(output_dir/"predictions.csv",index=False); pd.DataFrame({"epoch":range(1,epochs+1),"training_loss":train_losses,"validation_loss":val_losses}).to_csv(output_dir/"training_history.csv",index=False); (output_dir/"metrics.json").write_text(json.dumps(metrics,indent=2),encoding="utf-8")
 fig,ax=plt.subplots(figsize=(6.5,4.8)); ax.plot(range(1,epochs+1),train_losses,label="Training"); ax.plot(range(1,epochs+1),val_losses,label="Validation"); ax.set(xlabel="Epoch",ylabel="Binary cross-entropy",title="CNN training dynamics"); ax.legend(); fig.tight_layout(); fig.savefig(output_dir/"loss_curves.png",dpi=160); plt.close(fig); return metrics


if __name__=="__main__":
 root=Path(__file__).resolve().parents[2]; p=argparse.ArgumentParser(); p.add_argument("--data-dir",type=Path,default=root/"data",help="Directory containing PCam images and label CSV files"); p.add_argument("--output-dir",type=Path,default=Path(__file__).parent/"outputs"); p.add_argument("--epochs",type=int,default=5); a=p.parse_args(); print(json.dumps(run(a.data_dir,a.output_dir,a.epochs),indent=2))
