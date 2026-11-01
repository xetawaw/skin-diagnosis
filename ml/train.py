import argparse, copy, json, random
from pathlib import Path
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader
from sklearn.metrics import balanced_accuracy_score, classification_report, confusion_matrix, f1_score
from sklearn.model_selection import StratifiedGroupKFold
from torchvision.models import efficientnet_b0, EfficientNet_B0_Weights
from ml.dataset import CLASS_NAMES, HAM10000Dataset, load_metadata

def make_model(pretrained=True):
    model = efficientnet_b0(weights=EfficientNet_B0_Weights.DEFAULT if pretrained else None)
    model.classifier[1] = nn.Linear(model.classifier[1].in_features, len(CLASS_NAMES))
    return model

def seed_all(seed):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)

@torch.no_grad()
def evaluate(model, loader, device):
    model.eval(); truth=[]; pred=[]
    for x,y in loader:
        pred.extend(model(x.to(device)).argmax(1).cpu().tolist()); truth.extend(y.tolist())
    labels=list(range(len(CLASS_NAMES)))
    return {'macro_f1':float(f1_score(truth,pred,labels=labels,average='macro',zero_division=0)), 'balanced_accuracy':float(balanced_accuracy_score(truth,pred)), 'classification_report':classification_report(truth,pred,labels=labels,target_names=CLASS_NAMES,output_dict=True,zero_division=0), 'confusion_matrix':confusion_matrix(truth,pred,labels=labels).tolist()}

def main():
    p=argparse.ArgumentParser(); p.add_argument('--data-root',default='data/HAM10000'); p.add_argument('--output-dir',default='artifacts'); p.add_argument('--folds',type=int,default=5); p.add_argument('--epochs',type=int,default=10); p.add_argument('--batch-size',type=int,default=32); p.add_argument('--learning-rate',type=float,default=1e-4); p.add_argument('--weight-decay',type=float,default=1e-4); p.add_argument('--num-workers',type=int,default=4); p.add_argument('--seed',type=int,default=42); a=p.parse_args()
    seed_all(a.seed); device=torch.device('cuda' if torch.cuda.is_available() else 'cpu'); df=load_metadata(a.data_root); out=Path(a.output_dir); out.mkdir(parents=True,exist_ok=True); results=[]
    splitter=StratifiedGroupKFold(n_splits=a.folds,shuffle=True,random_state=a.seed)
    for fold,(train_idx,val_idx) in enumerate(splitter.split(df,df.label,groups=df.lesion_id)):
        seed_all(a.seed+fold); train=df.iloc[train_idx]; val=df.iloc[val_idx]
        assert set(train.lesion_id).isdisjoint(set(val.lesion_id)), 'Lesion leakage detected'
        counts=np.bincount(train.label,minlength=len(CLASS_NAMES))
        if (counts==0).any(): raise ValueError(f'Fold {fold} missing class in training: {counts}')
        weights=1/counts.astype(float); weights/=weights.mean()
        criterion=nn.CrossEntropyLoss(weight=torch.tensor(weights,dtype=torch.float32,device=device))
        train_dl=DataLoader(HAM10000Dataset(train,True),batch_size=a.batch_size,shuffle=True,num_workers=a.num_workers,pin_memory=device.type=='cuda')
        val_dl=DataLoader(HAM10000Dataset(val),batch_size=a.batch_size,num_workers=a.num_workers,pin_memory=device.type=='cuda')
        model=make_model().to(device); optim=torch.optim.AdamW(model.parameters(),lr=a.learning_rate,weight_decay=a.weight_decay); best=-1; best_metrics=None
        assignment=df[['image_id','lesion_id','dx']].copy(); assignment['partition']='train'; assignment.loc[val_idx,'partition']='validation'; assignment.to_csv(out/f'fold_{fold}_assignments.csv',index=False)
        print(f'Fold {fold}: train {len(train)}, validation {len(val)}, class counts {counts.tolist()}',flush=True)
        for epoch in range(1,a.epochs+1):
            model.train(); total=0
            for x,y in train_dl:
                x,y=x.to(device),y.to(device); optim.zero_grad(set_to_none=True); loss=criterion(model(x),y); loss.backward(); optim.step(); total+=loss.item()*len(x)
            metrics=evaluate(model,val_dl,device); print(f"fold={fold} epoch={epoch} loss={total/len(train):.4f} macro_f1={metrics['macro_f1']:.4f} balanced_accuracy={metrics['balanced_accuracy']:.4f}",flush=True)
            if metrics['macro_f1']>best:
                best=metrics['macro_f1']; best_metrics=metrics
                torch.save({'model_name':'efficientnet_b0','model_state_dict':copy.deepcopy({k:v.detach().cpu() for k,v in model.state_dict().items()}),'class_names':CLASS_NAMES,'fold':fold,'epoch':epoch,'validation_macro_f1':best,'image_size':224},out/f'fold_{fold}_best.pt')
        (out/f'fold_{fold}_metrics.json').write_text(json.dumps(best_metrics,indent=2)); results.append(best_metrics)
    scores=[r['macro_f1'] for r in results]; bas=[r['balanced_accuracy'] for r in results]
    summary={'fold_macro_f1':scores,'macro_f1_mean':float(np.mean(scores)),'macro_f1_std':float(np.std(scores)),'fold_balanced_accuracy':bas,'balanced_accuracy_mean':float(np.mean(bas)),'balanced_accuracy_std':float(np.std(bas)),'note':'Epoch selection uses the same folds: use nested grouped CV or an independent grouped test for final claims.'}
    (out/'cross_validation_summary.json').write_text(json.dumps(summary,indent=2)); print(json.dumps(summary,indent=2))
if __name__=='__main__': main()
