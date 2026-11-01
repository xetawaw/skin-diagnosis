import argparse
import torch
from ml.dataset import CLASS_NAMES, IMAGE_SIZE
from ml.train import make_model

def main():
    p=argparse.ArgumentParser(); p.add_argument('--checkpoint',default='artifacts/fold_0_best.pt'); p.add_argument('--output',default='artifacts/efficientnet_b0.ts'); a=p.parse_args()
    ckpt=torch.load(a.checkpoint,map_location='cpu',weights_only=True)
    if ckpt['class_names']!=CLASS_NAMES: raise ValueError('Class order mismatch')
    model=make_model(False); model.load_state_dict(ckpt['model_state_dict']); model.eval(); x=torch.randn(1,3,IMAGE_SIZE,IMAGE_SIZE)
    traced=torch.jit.trace(model,x)
    with torch.no_grad(): torch.testing.assert_close(model(x),traced(x),rtol=1e-4,atol=1e-5)
    traced.save(a.output); print('Saved',a.output)
if __name__=='__main__': main()
