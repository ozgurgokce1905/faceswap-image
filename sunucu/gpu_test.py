"""
gpu_test.py - Hangi hızlandırıcının çalıştığını bulur: CUDA (NVIDIA) -> DirectML (Intel/AMD) -> CPU.
Sonucu data/provider.txt dosyasına yazar (cuda / directml / cpu). Çıkış kodu 0 = GPU var, 1 = sadece CPU.
"""
import os
import sys
import time
import traceback

import numpy as np
import onnxruntime as ort
from onnx import TensorProto, helper

HOME = os.environ.get('FACESWAP_HOME') or os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(HOME, 'data')
os.makedirs(DATA, exist_ok=True)

print('python', sys.version.split()[0])
available = ort.get_available_providers()
print('onnxruntime', ort.__version__, '| derlenmiş sağlayıcılar:', available)

try:
    if hasattr(ort, 'preload_dlls') and 'CUDAExecutionProvider' in available:
        ort.preload_dlls()
except Exception as e:
    print('preload_dlls:', e)

# Test modeli: küçük bir matris çarpımı (GPU'da gerçekten çalışsın)
x = helper.make_tensor_value_info('X', TensorProto.FLOAT, [1, 256, 256])
y = helper.make_tensor_value_info('Y', TensorProto.FLOAT, [1, 256, 256])
w = helper.make_tensor('W', TensorProto.FLOAT, [256, 256], np.random.rand(256, 256).astype(np.float32).ravel().tolist())
node = helper.make_node('MatMul', ['X', 'W'], ['Y'])
graph = helper.make_graph([node], 'g', [x], [y], initializer=[w])
model = helper.make_model(graph, opset_imports=[helper.make_opsetid('', 17)])
model.ir_version = 8
model_bytes = model.SerializeToString()
inp = {'X': np.random.rand(1, 256, 256).astype(np.float32)}


def try_provider(name, label):
    if name not in available:
        print('-', label, ': bu onnxruntime paketinde yok')
        return False
    try:
        so = ort.SessionOptions()
        so.log_severity_level = 3
        sess = ort.InferenceSession(model_bytes, so, providers=[name, 'CPUExecutionProvider'])
        if sess.get_providers()[0] != name:
            print('-', label, ': yüklenemedi, CPU’ya düştü')
            return False
        sess.run(None, inp)
        t0 = time.time()
        for _ in range(20):
            sess.run(None, inp)
        ms = (time.time() - t0) / 20 * 1000
        print('+', label, ': ÇALIŞIYOR  (%.2f ms / çalıştırma)' % ms)
        return True
    except Exception as e:
        msg = str(e).strip().splitlines()
        print('-', label, ': HATA:', msg[0] if msg else e)
        return False


def try_openvino():
    if 'OpenVINOExecutionProvider' not in available:
        print('- OpenVINO (Intel): bu onnxruntime paketinde yok')
        return False
    for device in ('GPU', 'CPU'):
        try:
            so = ort.SessionOptions()
            so.log_severity_level = 3
            sess = ort.InferenceSession(model_bytes, so, providers=[('OpenVINOExecutionProvider', {'device_type': device, 'precision': 'FP32'}), 'CPUExecutionProvider'])
            if 'OpenVINOExecutionProvider' not in sess.get_providers():
                print('- OpenVINO (%s): yüklenemedi (openvino.dll eksik olabilir), CPU’ya düştü' % device)
                return False
            sess.run(None, inp)
            t0 = time.time()
            for _ in range(20):
                sess.run(None, inp)
            ms = (time.time() - t0) / 20 * 1000
            print('+ OpenVINO (%s): ÇALIŞIYOR  (%.2f ms / çalıştırma)' % (device, ms))
            return device == 'GPU'
        except Exception as e:
            msg = str(e).strip().splitlines()
            print('- OpenVINO (%s): HATA: %s' % (device, msg[0] if msg else e))
    return False


# İstenen sıra: komut satırından verilen tercih önce denenir (ör. gpu_test.py openvino)
prefer = sys.argv[1].lower() if len(sys.argv) > 1 else ''
chosen = 'cpu'
order = ['cuda', 'openvino', 'directml']
if prefer in order:
    order.remove(prefer)
    order.insert(0, prefer)
for name in order:
    if name == 'cuda' and try_provider('CUDAExecutionProvider', 'CUDA (NVIDIA)'):
        chosen = 'cuda'; break
    if name == 'openvino' and try_openvino():
        chosen = 'openvino'; break
    if name == 'directml' and try_provider('DmlExecutionProvider', 'DirectML (Intel/AMD)'):
        chosen = 'directml'; break
if chosen == 'cpu':
    try_provider('CPUExecutionProvider', 'CPU')
if prefer == 'cpu':
    chosen = 'cpu'

with open(os.path.join(DATA, 'provider.txt'), 'w') as f:
    f.write(chosen)
print('SEÇİLEN HIZLANDIRICI:', chosen)
sys.exit(0 if chosen != 'cpu' else 1)
