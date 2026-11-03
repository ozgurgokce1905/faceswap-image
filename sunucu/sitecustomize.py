# FaceSwap: venv içindeki her python başlangıcında hızlandırıcı DLL'lerini bulunur hale getirir.
#  - CUDA/cuDNN: pip nvidia paketlerinden (onnxruntime.preload_dlls)
#  - OpenVINO : pip openvino paketinin libs klasörü (openvino.dll, tbb vs.)
import os as _os
import sys as _sys

try:
    _site = _os.path.dirname(_os.path.abspath(__file__))
    _ov = _os.path.join(_site, 'openvino', 'libs')
    if _os.path.isdir(_ov):
        _os.environ['PATH'] = _ov + _os.pathsep + _os.environ.get('PATH', '')
        if hasattr(_os, 'add_dll_directory'):
            _os.add_dll_directory(_ov)
except Exception:
    pass

try:
    if _os.path.isdir(_os.path.join(_site, 'nvidia')):
        import onnxruntime as _ort
        if hasattr(_ort, 'preload_dlls'):
            _ort.preload_dlls()
except Exception:
    pass
