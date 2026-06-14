set -e
pip install --no-cache-dir -q torch torchvision timm onnxscript onnx 2>&1 | tail -1
python - <<'PY'
import torch, torch.nn as nn, torch.nn.functional as F, numpy as np, os
# Force GELU tanh-approximation everywhere (TFLite-builtin-friendly; no Erf/Flex op)
_g = F.gelu
F.gelu = lambda x, approximate='none': _g(x, approximate='tanh')
m = torch.hub.load("otroshi/edgeface","edgeface_xs_gamma_06",pretrained=True,trust_repo=True)
for mod in m.modules():
    if isinstance(mod, nn.GELU): mod.approximate = 'tanh'
m.eval()
x = torch.randn(1,3,112,112)
with torch.no_grad(): ref = m(x).numpy()
np.save("/out/ref_input.npy", x.numpy()); np.save("/out/ref_embedding.npy", ref)
torch.onnx.export(m, x, "/out/edgeface_xs.onnx",
    input_names=["input"], output_names=["embedding"], opset_version=13, dynamo=False)
print("EXPORT_OK size=", os.path.getsize("/out/edgeface_xs.onnx"), "norm=", float(np.linalg.norm(ref)))
PY
