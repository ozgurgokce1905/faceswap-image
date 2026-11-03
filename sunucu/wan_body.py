"""Tüm vücut modu (Wan2.2-Animate-14B replacement): kurulum durumu + iş çalıştırma.

Yüz modundan farkı: seçilen kişinin yüzü, saçı, kıyafeti ve vücudu referans
fotoğraftaki kişiyle değiştirilir; hareketler ve arka plan korunur.
Ağır model (14B) olduğu için yalnızca büyük GPU'da (H100/A100 80 GB) çalışır.

Kalite seviyeleri (preset): hizli = 480p/12 adım, kaliteli = 720p/20 adım, ultra = 720p/30 adım.
"""
import json
import os
import re
import shutil
import subprocess
import threading
import time
from typing import Callable, Optional

HERE = os.path.dirname(os.path.abspath(__file__))
FS = os.environ.get('FACESWAP_HOME', os.path.dirname(HERE))
WAN_HOME = os.environ.get('WAN_HOME', '/workspace/wan')
WAN_REPO = os.path.join(WAN_HOME, 'Wan2.2')
WAN_PY = os.path.join(WAN_HOME, 'venv', 'bin', 'python')
WAN_CKPT = os.path.join(WAN_HOME, 'Wan2.2-Animate-14B')
STAGE_FILE = os.path.join(FS, 'data', 'wan_stage.txt')
SETUP_SH = os.path.join(FS, 'runpod', 'wan_setup.sh')

BODY_PRESETS = {
    'hizli': {'label': 'Hızlı (480p)', 'area': (832, 480), 'steps': 12},
    'kaliteli': {'label': 'Kaliteli (720p)', 'area': (1280, 720), 'steps': 20},
    'ultra': {'label': 'Ultra (720p, 30 adım)', 'area': (1280, 720), 'steps': 30},
}
# Tahmini saniye / çıktı karesi (H100, 30 fps). Gerçek ölçümlerle güncellenir.
BODY_SECONDS_PER_FRAME = {'hizli': 0.9, 'kaliteli': 2.5, 'ultra': 3.6}

_setup_proc: Optional[subprocess.Popen] = None
_setup_lock = threading.Lock()


def available() -> bool:
    """Bu makinede tüm vücut modu çalışabilir mi (CUDA + kurulum scripti var)?"""
    return os.environ.get('FACESWAP_PROVIDER', 'cuda') == 'cuda' and os.path.isfile(SETUP_SH)


def is_ready() -> bool:
    return os.path.isfile(os.path.join(WAN_CKPT, '.ok')) and os.path.isfile(os.path.join(WAN_HOME, 'venv', '.ok'))


def stage() -> str:
    try:
        with open(STAGE_FILE, encoding='utf-8') as f:
            return f.read().strip()
    except Exception:
        return ''


def status() -> dict:
    installing = _setup_proc is not None and _setup_proc.poll() is None
    s = stage()
    if is_ready():
        st = 'ready'
    elif installing:
        st = 'installing'
    elif s.startswith('HATA'):
        st = 'error'
    else:
        st = 'none'
    return {'available': available(), 'status': st, 'stage': s, 'presets': {k: v['label'] for k, v in BODY_PRESETS.items()},
            'seconds_per_frame': BODY_SECONDS_PER_FRAME}


def start_setup() -> bool:
    """Kurulumu arka planda başlatır (zaten kuruluysa/çalışıyorsa False)."""
    global _setup_proc
    with _setup_lock:
        if is_ready():
            return False
        if _setup_proc is not None and _setup_proc.poll() is None:
            return False
        env = dict(os.environ, WAN_HOME=WAN_HOME, FACESWAP_HOME=FS)
        _setup_proc = subprocess.Popen(['bash', SETUP_SH], env=env, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        return True


def wait_setup(timeout: float, is_cancelled: Callable[[], bool], on_stage: Callable[[str], None]) -> None:
    t0 = time.time()
    start_setup()
    while not is_ready():
        if is_cancelled():
            raise RuntimeError('İptal edildi')
        if time.time() - t0 > timeout:
            raise RuntimeError('Tüm vücut kurulumu zaman aşımına uğradı: ' + stage())
        s = stage()
        if s.startswith('HATA'):
            raise RuntimeError('Tüm vücut kurulumu başarısız: ' + s)
        on_stage('Tüm vücut kurulumu: ' + (s or 'başlıyor'))
        time.sleep(3)


def run_body_swap(video_path: str, ref_image: str, out_path: str, work_dir: str, preset: str,
                  ref_frame: int, face_cx: float, face_cy: float,
                  on_progress: Callable[[str, float], None], is_cancelled: Callable[[], bool],
                  log: Callable[[str], None], trim_start: Optional[int] = None, trim_end: Optional[int] = None,
                  seed: int = 42) -> dict:
    """Tam akış: ön işleme -> Wan üretimi -> kırpma/ses birleştirme. Bilgi sözlüğü döner."""
    p = BODY_PRESETS.get(preset, BODY_PRESETS['kaliteli'])
    os.makedirs(work_dir, exist_ok=True)
    prep_dir = os.path.join(work_dir, 'prep')
    gen_out = os.path.join(work_dir, 'wan_out.mp4')
    env = dict(os.environ)
    env['PYTHONPATH'] = os.path.join(HERE, 'decord_shim') + os.pathsep + WAN_REPO + os.pathsep + env.get('PYTHONPATH', '')
    env['PYTHONUNBUFFERED'] = '1'

    # 1) Ön işleme (kişi seçimli): ~1-3 dk
    on_progress('Kişi takip ediliyor ve maske çıkarılıyor', 2)
    cmd = [WAN_PY, os.path.join(HERE, 'wan_prep.py'), '--wan', WAN_REPO, '--ckpt', os.path.join(WAN_CKPT, 'process_checkpoint'),
           '--video', video_path, '--ref', ref_image, '--out', prep_dir,
           '--area', str(p['area'][0]), str(p['area'][1]), '--fps', '30',
           '--ref-frame', str(ref_frame), '--face-x', '%.4f' % face_cx, '--face-y', '%.4f' % face_cy]
    if trim_start is not None:
        cmd += ['--trim-start', str(trim_start)]
    if trim_end is not None:
        cmd += ['--trim-end', str(trim_end)]
    log('$ ' + ' '.join(cmd) + '\n')
    _run(cmd, env, WAN_REPO, log, is_cancelled, lambda line: _prep_progress(line, on_progress))
    info_path = os.path.join(prep_dir, 'info.json')
    if not os.path.isfile(info_path):
        raise RuntimeError('Ön işleme çıktı üretmedi (log\'a bak)')
    with open(info_path) as f:
        info = json.load(f)
    n_frames = int(info['frames'])
    clips = max(1, -(-(n_frames - 1) // 76))
    on_progress('Vücut üretiliyor (%d parça)' % clips, 10)

    # 2) Üretim
    mem_gb = gpu_memory_gb()
    cmd = [WAN_PY, 'generate.py', '--task', 'animate-14B', '--ckpt_dir', WAN_CKPT, '--src_root_path', prep_dir,
           '--refert_num', '1', '--replace_flag', '--use_relighting_lora', '--save_file', gen_out,
           '--sample_steps', str(p['steps']), '--base_seed', str(seed)]
    if mem_gb >= 70:
        cmd += ['--offload_model', 'False']  # H100/A100 80 GB: her şey GPU'da, en hızlı
    else:
        cmd += ['--offload_model', 'True', '--t5_cpu', '--convert_model_dtype']  # 48 GB kartlar (L40S/A6000): bellek tasarrufu
    log('GPU belleği: %.0f GB -> %s\n' % (mem_gb, 'tam yükleme' if mem_gb >= 70 else 'bellek tasarrufu modu'))
    log('$ ' + ' '.join(cmd) + '\n')
    state = {'clip': 0, 'clips': clips, 'steps': p['steps']}

    def gen_line(line: str):
        m = re.search(r'target frames:\s*(\d+)', line)
        if m:
            state['clips'] = max(1, (int(m.group(1)) - 1) // 76)
        m = re.search(r'(\d+)/(\d+)\s*\[', line)
        if m:
            step, total = int(m.group(1)), int(m.group(2))
            if total == state['steps']:
                frac = (state['clip'] + step / max(total, 1)) / state['clips']
                on_progress('Vücut üretiliyor (parça %d/%d)' % (min(state['clip'] + 1, state['clips']), state['clips']),
                            10 + 82 * min(frac, 1.0))
                if step == total:
                    state['clip'] += 1

    t_gen = time.time()
    _run(cmd, env, WAN_REPO, log, is_cancelled, gen_line)
    if not os.path.isfile(gen_out):
        raise RuntimeError('Wan çıktı üretmedi (log\'a bak)')
    gen_seconds = time.time() - t_gen

    # 3) Son işlem: siyah bantları kırp, orijinal çözünürlüğe getir, sesi ekle, uzunluğu gerçek kareye kısalt
    on_progress('Video birleştiriliyor', 93)
    W, H = int(info['width']), int(info['height'])
    ow, oh = int(info['orig_w']), int(info['orig_h'])
    s = min(W / ow, H / oh)
    cw, ch = int(ow * s) // 2 * 2, int(oh * s) // 2 * 2
    cx, cy = (W - cw) // 2, (H - ch) // 2
    dur = n_frames / 30.0
    start_sec = (info.get('start') or 0) / float(info.get('video_fps') or 30.0)
    vf = 'crop=%d:%d:%d:%d,scale=%d:%d:flags=lanczos' % (cw, ch, cx, cy, ow // 2 * 2, oh // 2 * 2)
    cmd = ['ffmpeg', '-y', '-loglevel', 'error', '-i', gen_out, '-ss', '%.3f' % start_sec, '-i', video_path,
           '-map', '0:v:0', '-map', '1:a:0?', '-t', '%.3f' % dur, '-vf', vf,
           '-c:v', 'libx264', '-preset', 'veryfast', '-crf', '17', '-pix_fmt', 'yuv420p', '-c:a', 'aac', '-b:a', '160k',
           '-movflags', '+faststart', out_path]
    log('$ ' + ' '.join(cmd) + '\n')
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode != 0 or not os.path.isfile(out_path):
        log(r.stderr[-2000:] + '\n')
        raise RuntimeError('ffmpeg birleştirme başarısız')
    shutil.rmtree(prep_dir, ignore_errors=True)
    try:
        os.remove(gen_out)
    except Exception:
        pass
    return {'frames': n_frames, 'gen_seconds': round(gen_seconds), 'clips': clips, 'size': (W, H)}


def gpu_memory_gb() -> float:
    try:
        out = subprocess.run(['nvidia-smi', '--query-gpu=memory.total', '--format=csv,noheader,nounits'], capture_output=True, text=True, timeout=10).stdout
        return float(out.strip().splitlines()[0]) / 1024.0
    except Exception:
        return 80.0


def _prep_progress(line: str, on_progress):
    if 'Kişiler tespit' in line:
        on_progress('Kişi takip ediliyor', 3)
    elif 'Poz çıkar' in line:
        on_progress('Poz çıkarılıyor', 5)
    elif 'maskesi' in line:
        on_progress('Kişi maskesi çıkarılıyor', 7)
    elif 'Videolar yaz' in line:
        on_progress('Ön işleme bitiyor', 9)


def _run(cmd, env, cwd, log, is_cancelled, on_line):
    proc = subprocess.Popen(cmd, cwd=cwd, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1,
                            errors='replace')
    buf = ''
    while True:
        ch = proc.stdout.read(1)
        if not ch:
            break
        if ch in '\r\n':
            if buf.strip():
                log(buf + '\n')
                try:
                    on_line(buf)
                except Exception:
                    pass
            buf = ''
        else:
            buf += ch
        if is_cancelled():
            proc.kill()
            raise RuntimeError('İptal edildi')
    proc.wait()
    if buf.strip():
        log(buf + '\n')
    if proc.returncode != 0:
        raise RuntimeError('Alt süreç hata verdi (kod %s), log\'a bak' % proc.returncode)
