set -e
cd /work
python - <<'PY'
import numpy as np, tensorflow as tf, glob
from PIL import Image
imgs=[]; names=[]
for p in sorted(glob.glob("calib/*")):
    im=Image.open(p).convert("RGB").resize((112,112))
    imgs.append(((np.asarray(im,np.float32)/255.0)-0.5)/0.5); names.append(p)
imgs=np.stack(imgs); print("calib",imgs.shape)
def rep():
    for a in imgs: yield [a[None].astype(np.float32)]
c=tf.lite.TFLiteConverter.from_saved_model("tf_out"); fp32=c.convert()
open("edgeface_xs_fp32_tf214.tflite","wb").write(fp32)
c=tf.lite.TFLiteConverter.from_saved_model("tf_out")
c.optimizations=[tf.lite.Optimize.DEFAULT]; c.representative_dataset=rep
c.target_spec.supported_ops=[tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
# float io (int8 internal)
int8=c.convert(); open("edgeface_xs_int8.tflite","wb").write(int8)
print("sizes fp32",len(fp32),"int8",len(int8))
def run(model,x):
    it=tf.lite.Interpreter(model_path=model); it.allocate_tensors()
    inp,out=it.get_input_details()[0],it.get_output_details()[0]
    xi=x.astype(np.float32)
    if inp["dtype"]==np.int8:
        s,z=inp["quantization"]; xi=np.clip(np.round(xi/s+z),-128,127).astype(np.int8)
    it.set_tensor(inp["index"],xi[None]); it.invoke()
    e=it.get_tensor(out["index"]).reshape(-1).astype(np.float32)
    if out["dtype"]==np.int8:
        s,z=out["quantization"]; e=(e-z)*s
    return e
face=imgs[[i for i,n in enumerate(names) if "test_face.jpg" in n][0]]
ef=run("edgeface_xs_fp32_tf214.tflite",face); ei=run("edgeface_xs_int8.tflite",face)
print(f"cosine(int8,fp32) real face = {np.dot(ef,ei)/(np.linalg.norm(ef)*np.linalg.norm(ei)+1e-12):.4f}")
x=np.load("ref_input.npy"); ref=np.load("ref_embedding.npy").reshape(-1)
efr=run("edgeface_xs_fp32_tf214.tflite",np.transpose(x,(0,2,3,1))[0])
print(f"cosine(fp32_tf214,torch) = {np.dot(efr,ref)/(np.linalg.norm(efr)*np.linalg.norm(ref)+1e-12):.6f}")
PY
ls -lh /work/edgeface_xs_int8.tflite /work/edgeface_xs_fp32_tf214.tflite
