import io
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from ml.dataset import IMAGE_SIZE, build_transforms

class GradCAM:
    def __init__(self, model, layer):
        self.model=model; self.activations=None; self.gradients=None
        self.fwd=layer.register_forward_hook(self._forward); self.bwd=layer.register_full_backward_hook(self._backward)
    def _forward(self, module, inputs, output): self.activations=output.detach()
    def _backward(self, module, grad_input, grad_output): self.gradients=grad_output[0].detach()
    def close(self): self.fwd.remove(); self.bwd.remove()
    def generate(self, x):
        self.model.eval(); self.model.zero_grad(set_to_none=True)
        with torch.enable_grad():
            logits=self.model(x); index=int(logits.argmax(1).item()); logits[0,index].backward()
        if self.activations is None or self.gradients is None: raise RuntimeError('Grad-CAM hook failed')
        cam=F.relu((self.gradients.mean((2,3),keepdim=True)*self.activations).sum(1,keepdim=True))
        cam=F.interpolate(cam,size=x.shape[-2:],mode='bilinear',align_corners=False)[0,0]
        cam=(cam-cam.min())/(cam.max()-cam.min()+1e-8)
        return logits.detach(),index,cam.detach().cpu().numpy()

def explain_image(model,image,device):
    x=build_transforms(False)(image.convert('RGB')).unsqueeze(0).to(device); engine=GradCAM(model,model.features[-1])
    try:
        logits,index,cam=engine.generate(x); probs=torch.softmax(logits,dim=1)[0].cpu().tolist()
        base=np.asarray(image.convert('RGB').resize((IMAGE_SIZE,IMAGE_SIZE),Image.Resampling.BILINEAR),dtype=np.float32)/255
        heat=np.stack([np.clip(1.5-np.abs(4*cam-3),0,1),np.clip(1.5-np.abs(4*cam-2),0,1),np.clip(1.5-np.abs(4*cam-1),0,1)],axis=-1)
        output=io.BytesIO(); Image.fromarray((np.clip(.55*base+.45*heat,0,1)*255).astype(np.uint8)).save(output,format='PNG')
        return index,probs,output.getvalue()
    finally: engine.close()
