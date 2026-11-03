"""
ff_bridge.py - FaceFusion'ı sunucu süreci içinde kullanmak için köprü.

Görevleri:
  * FaceFusion durumunu (state) tek sefer başlatmak (yüz dedektörü ayarları vb.)
  * Bir videonun belirli bir karesindeki yüzleri, FaceFusion'ın kendi dedektörü ve
    kendi sıralamasıyla (left-right) bulmak -> telefon bu sırayı "reference_face_position"
    olarak geri gönderir, böylece seçilen kişi birebir aynı olur.
  * Gerekli modelleri önceden indirmek (warm-up).
"""
import os
import sys
import threading

FACESWAP_HOME = os.environ.get('FACESWAP_HOME') or os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FACEFUSION_DIR = os.environ.get('FACESWAP_FF_DIR') or os.path.join(FACESWAP_HOME, 'facefusion')

# FaceFusion göreli yollarla çalışır (facefusion/processors/modules, .assets/models) -> cwd oraya alınır
os.chdir(FACEFUSION_DIR)
if FACEFUSION_DIR not in sys.path:
    sys.path.insert(0, FACEFUSION_DIR)

# CUDA / cuDNN DLL'lerini pip nvidia paketlerinden önceden yükle (Windows'ta conda yoksa şart)
try:
    import onnxruntime as _ort
    if hasattr(_ort, 'preload_dlls'):
        _ort.preload_dlls()
except Exception:
    pass

from facefusion import content_analyser, face_classifier, face_detector, face_landmarker, face_recognizer, logger, state_manager  # noqa: E402
from facefusion.args import apply_args  # noqa: E402
from facefusion.face_creator import get_many_faces  # noqa: E402
from facefusion.face_selector import sort_faces_by_order  # noqa: E402
from facefusion.program import create_program  # noqa: E402
from facefusion.vision import count_video_frame_total, detect_video_fps, detect_video_resolution, read_video_frame  # noqa: E402

_lock = threading.Lock()
_ready = False

# Yüz tespiti ayarları. DİKKAT: presets.py içindeki değerlerle BİREBİR aynı olmalı,
# yoksa seçilen kişinin sırası (index) kayar.
DETECTOR_ARGS = [
    '--face-detector-model', 'yolo_face',
    '--face-detector-size', '640x640',
    '--face-detector-score', '0.5',
    '--face-landmarker-model', '2dfan4',
    '--face-selector-order', 'left-right',
]


def _tune_gpu_sessions() -> None:
    """GPU'da onnxruntime'ın CPU iş parçacıklarını kısar.

    Varsayılanda her model oturumu çekirdek sayısı kadar (26+) 'spin' eden
    CPU thread'i açar; 8-10 oturum x 26 thread birbirini boğar, GPU %7'de
    kalır, CPU %100 olur (RunPod'da görüldü). GPU'da ağır işi kart yaptığı
    için 2 thread yeter.
    """
    import onnxruntime as ort
    import facefusion.inference_manager as im
    if getattr(im, '_faceswap_tuned', False):
        return
    _orig = ort.InferenceSession

    def _session(path, sess_options=None, providers=None, **kw):
        so = sess_options or ort.SessionOptions()
        so.intra_op_num_threads = 2
        so.inter_op_num_threads = 1
        so.add_session_config_entry('session.intra_op.allow_spinning', '0')
        so.add_session_config_entry('session.inter_op.allow_spinning', '0')
        return _orig(path, sess_options=so, providers=providers, **kw)

    im.InferenceSession = _session
    im._faceswap_tuned = True
    try:
        import cv2
        cv2.setNumThreads(4)
    except Exception:
        pass


def init(execution_provider: str = 'cuda', log_level: str = 'warn') -> None:
    """FaceFusion state'ini başlatır ve küçük ortak modelleri indirir (tek sefer)."""
    global _ready
    with _lock:
        if _ready:
            return
        if execution_provider in ('cuda', 'tensorrt'):
            _tune_gpu_sessions()
        program = create_program()
        argv = ['headless-run', '--execution-providers', execution_provider, '--log-level', log_level] + DETECTOR_ARGS
        args = vars(program.parse_args(argv))
        apply_args(args, state_manager.init_item)
        logger.init(log_level)
        for module in (content_analyser, face_classifier, face_detector, face_landmarker, face_recognizer):
            if not module.pre_check():
                raise RuntimeError('Model indirilemedi: ' + module.__name__)
        _ready = True


def video_info(video_path: str) -> dict:
    fps = detect_video_fps(video_path)
    total = count_video_frame_total(video_path)
    res = detect_video_resolution(video_path) or (0, 0)
    return {
        'fps': float(fps or 0),
        'frame_count': int(total or 0),
        'width': int(res[0]),
        'height': int(res[1]),
        'duration': (float(total) / float(fps)) if fps else 0.0,
    }


def detect_faces_in_frame(video_path: str, frame_number: int) -> dict:
    """
    Verilen karedeki yüzleri döndürür. Sıra = FaceFusion'ın 'left-right' sırası.
    Dönen her yüz: index, bbox (piksel ve normalize), score, gender, age
    """
    init_if_needed()
    with _lock:
        frame = read_video_frame(video_path, frame_number)
        if frame is None:
            return {'frame_number': frame_number, 'width': 0, 'height': 0, 'faces': []}
        height, width = frame.shape[0], frame.shape[1]
        faces = get_many_faces([frame])
        faces = sort_faces_by_order(faces, 'left-right')
        result = []
        for index, face in enumerate(faces):
            x1, y1, x2, y2 = [float(v) for v in face.bounding_box]
            x1, y1 = max(0.0, x1), max(0.0, y1)
            x2, y2 = min(float(width), x2), min(float(height), y2)
            result.append({
                'index': index,
                'bbox': [x1, y1, x2, y2],
                'bbox_norm': [x1 / width, y1 / height, x2 / width, y2 / height],
                'score': float(face.score_set.get('detector', 0.0)),
                'gender': face.gender,
                'age': list(face.age) if hasattr(face.age, '__iter__') else face.age,
            })
        return {'frame_number': frame_number, 'width': width, 'height': height, 'faces': result, 'frame': frame}


def init_if_needed() -> None:
    if not _ready:
        init(os.environ.get('FACESWAP_PROVIDER', 'cuda'))


# ----------------------------------------------------------------------------- işlem (sunucu süreci içinde)
# FaceFusion'ı ayrı bir süreç olarak başlatmak yerine burada çalıştırıyoruz:
# modeller bir kez yüklenir ve bellekte kalır (video_memory_strategy=tolerant).

import tqdm as _tqdm_module  # noqa: E402

from facefusion import core as _core, process_manager as _pm  # noqa: E402

_progress_callback = None


class _HookedTqdm(_tqdm_module.tqdm):
    """FaceFusion'ın tqdm çubuklarını yakalayıp ilerlemeyi sunucuya bildirir (konsola da yazmaya devam eder)."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._report()

    def update(self, n=1):
        result = super().update(n)
        self._report()
        return result

    def _report(self):
        cb = _progress_callback
        if cb:
            try:
                cb(str(self.desc or ''), int(self.n or 0), int(self.total or 0))
            except Exception:
                pass


def _install_tqdm_hook() -> None:
    import facefusion.content_analyser
    import facefusion.download
    import facefusion.ffmpeg
    import facefusion.workflows.to_video
    for module in (facefusion.content_analyser, facefusion.download, facefusion.ffmpeg, facefusion.workflows.to_video):
        module.tqdm = _HookedTqdm


_install_tqdm_hook()


def run_swap(argv, progress_cb=None, is_cancelled=None, log=None) -> int:
    """
    argv: presets.build_args(...) çıktısı ('headless-run' ile başlar).
    Dönüş: FaceFusion hata kodu (0 = başarılı, 2 = ön kontrol, 3 = içerik filtresi, 4 = durduruldu).
    Çağıran taraf ff_bridge._lock'u tutmalı.
    """
    global _progress_callback
    init_if_needed()
    program = create_program()
    try:
        args = vars(program.parse_args(argv))
    except SystemExit:
        if log:
            log('Argüman hatası: ' + ' '.join(argv) + '\n')
        return 1
    # Modeller bellekte kalsın (her işte yeniden yüklenmesin)
    args['video_memory_strategy'] = 'tolerant'
    apply_args(args, state_manager.set_item)
    logger.init(args.get('log_level') or 'info')
    _progress_callback = progress_cb
    try:
        if not _core.common_pre_check():
            return 2
        if not _core.processors_pre_check():
            return 2
        if is_cancelled and is_cancelled():
            return 4
        return int(_core.conditional_process())
    finally:
        _progress_callback = None
        _pm.end()


def request_stop() -> None:
    """Süren işi durdurur (FaceFusion döngüleri is_process_stopping() ile çıkar)."""
    if _pm.is_processing():
        _pm.stop()


def clear_models() -> None:
    """Yüklü işlemci modellerini bellekten atar (dedektör modelleri kalır)."""
    from facefusion.processors.core import get_processors_modules
    available = ['face_swapper', 'face_enhancer', 'expression_restorer', 'frame_enhancer', 'lip_syncer', 'face_editor', 'age_modifier', 'deep_swapper', 'background_remover', 'frame_colorizer']
    for module in get_processors_modules(available):
        try:
            module.clear_inference_pool()
        except Exception:
            pass
