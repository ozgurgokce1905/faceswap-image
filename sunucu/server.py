"""
server.py - FaceSwap sunucusu (FastAPI). Telefon uygulaması bu API ile konuşur.

Akış:
  1. POST /videos                  -> video yükle (id, fps, kare sayısı döner)
  2. GET  /videos/{id}/frame/{n}   -> n. karenin JPEG'i (telefon zaman çubuğunda gezer)
  3. GET  /videos/{id}/faces/{n}   -> n. karedeki yüzler (kutular + küçük resimler)
  4. POST /faces                   -> kütüphaneye yüz fotoğrafı ekle
  5. POST /jobs                    -> {video_id, frame_number, face_index, face_ids, preset} -> işi başlat
  6. GET  /jobs/{id}               -> durum + yüzde ilerleme
  7. GET  /jobs/{id}/result        -> biten videoyu indir
"""
import base64
import json
import os
import shutil
import socket
import subprocess
import sys
import threading
import time
import uuid
from typing import List, Optional

import cv2
from fastapi import FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.responses import FileResponse, JSONResponse, Response
from pydantic import BaseModel

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import ff_bridge  # noqa: E402  (bu import cwd'yi facefusion klasörüne alır)
import presets  # noqa: E402
import wan_body  # noqa: E402  (tüm vücut modu)

HOME = ff_bridge.FACESWAP_HOME
FF_DIR = ff_bridge.FACEFUSION_DIR
DATA = os.path.join(HOME, 'data')
VIDEOS = os.path.join(DATA, 'videos')
FACES = os.path.join(DATA, 'faces')
JOBS = os.path.join(DATA, 'jobs')
TEMP = os.path.join(DATA, 'temp')
for d in (VIDEOS, FACES, JOBS, TEMP):
    os.makedirs(d, exist_ok=True)

PROVIDER = os.environ.get('FACESWAP_PROVIDER', 'cuda')
THREADS = int(os.environ.get('FACESWAP_THREADS', '4'))
PORT = int(os.environ.get('FACESWAP_PORT', '8765'))

from contextlib import asynccontextmanager  # noqa: E402


@asynccontextmanager
async def _lifespan(_app):
    _startup()
    yield


app = FastAPI(title='FaceSwap Sunucu', version='0.2', lifespan=_lifespan)

# ----------------------------------------------------------------------------- erişim kodu
# RunPod'da adres herkese açık; FACESWAP_TOKEN tanımlıysa her istekte X-Token başlığı (ya da ?token=) zorunlu.
TOKEN = os.environ.get('FACESWAP_TOKEN', '').strip()
_last_activity = time.time()


@app.middleware('http')
async def _auth(request, call_next):
    global _last_activity
    path = request.url.path
    if path != '/ping':
        # Durum sorguları (health, iş listesi) "kullanım" sayılmaz; uygulama açık unutulsa da pod kapanır
        if request.method != 'GET' or '/frame/' in path or '/faces/' in path or path.endswith('/result'):
            _last_activity = time.time()
        if TOKEN:
            given = request.headers.get('x-token') or request.query_params.get('token') or ''
            if given != TOKEN:
                return JSONResponse({'detail': 'Erişim kodu hatalı'}, status_code=401)
    return await call_next(request)


@app.get('/ping')
def ping():
    return {'ok': True}


# ----------------------------------------------------------------------------- otomatik kapanma (RunPod)
IDLE_MINUTES = int(os.environ.get('FACESWAP_IDLE_MINUTES', '15'))
POD_ID = os.environ.get('RUNPOD_POD_ID', '')


def _idle_watcher():
    """RunPod'da: iş yokken ve IDLE_MINUTES dakika istek gelmezse pod kendini siler (veri network volume'da kalır)."""
    while True:
        time.sleep(30)
        if not POD_ID or IDLE_MINUTES <= 0:
            continue
        if _current_job is not None or _queue:
            continue
        if time.time() - _last_activity > IDLE_MINUTES * 60:
            print('[FaceSwap] %d dk boşta kaldı, pod kapatılıyor.' % IDLE_MINUTES, flush=True)
            _terminate_self()
            time.sleep(120)


def _terminate_self():
    try:
        r = subprocess.call(['runpodctl', 'remove', 'pod', POD_ID])
        if r == 0:
            return
    except Exception as e:
        print('runpodctl hata:', e)
    key = os.environ.get('RUNPOD_API_KEY', '')
    if key:
        try:
            import requests
            requests.delete('https://rest.runpod.io/v1/pods/' + POD_ID, headers={'Authorization': 'Bearer ' + key}, timeout=20)
        except Exception as e:
            print('REST kapatma hata:', e)


# ----------------------------------------------------------------------------- yardımcılar

def _read_json(path, default):
    try:
        with open(path, 'r', encoding='utf-8') as f:
            return json.load(f)
    except Exception:
        return default


def _write_json(path, data):
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=1)


def _jpeg_b64(frame_bgr, max_side=128) -> str:
    h, w = frame_bgr.shape[:2]
    scale = max_side / float(max(h, w)) if max(h, w) > max_side else 1.0
    if scale < 1.0:
        frame_bgr = cv2.resize(frame_bgr, (int(w * scale), int(h * scale)))
    ok, buf = cv2.imencode('.jpg', frame_bgr, [cv2.IMWRITE_JPEG_QUALITY, 85])
    return base64.b64encode(buf.tobytes()).decode('ascii') if ok else ''


def _lan_ips() -> List[str]:
    ips = set()
    try:
        for info in socket.getaddrinfo(socket.gethostname(), None, socket.AF_INET):
            ip = info[4][0]
            if not ip.startswith('127.'):
                ips.add(ip)
    except Exception:
        pass
    return sorted(ips)



# Varsayılan tahminler (saniye / kare); tamamlanan işlerden öğrenilir
DEFAULT_SPF = {'hizli': 1.5, 'kaliteli': 3.0, 'ultra': 6.0}


def _seconds_per_frame() -> dict:
    est = dict(DEFAULT_SPF)
    with _job_lock:
        jobs = list(_jobs.values())
    for preset in est:
        samples = [j for j in jobs if j.get('status') == 'done' and j.get('preset') == preset and j.get('frames') and j.get('elapsed') and j.get('mode', 'yuz') == 'yuz']
        samples = sorted(samples, key=lambda j: j.get('finished', 0))[-3:]
        if samples:
            est[preset] = round(sum(j['elapsed'] / float(j['frames']) for j in samples) / len(samples), 2)
    return est

# ----------------------------------------------------------------------------- sağlık

@app.get('/health')
def health():
    info = {'ok': True, 'provider': PROVIDER, 'ips': _lan_ips(), 'port': PORT, 'presets': {k: v['label'] for k, v in presets.PRESETS.items()},
            'busy': _current_job is not None, 'seconds_per_frame': _seconds_per_frame(),
            'pod_id': POD_ID or None, 'idle_minutes': IDLE_MINUTES if POD_ID else None,
            'idle_seconds': int(time.time() - _last_activity), 'body': wan_body.status()}
    if PROVIDER == 'openvino':
        try:
            import openvino
            info['openvino_devices'] = openvino.Core().available_devices
        except Exception as e:
            info['openvino_devices'] = str(e)
    try:
        import onnxruntime as ort
        info['onnxruntime'] = ort.__version__
        info['providers'] = ort.get_available_providers()
    except Exception as e:  # pragma: no cover
        info['onnxruntime_error'] = str(e)
    return info


def _require_idle():
    """Video işlenirken kare/yüz istekleri bekletilmez; telefon 'meşgul' mesajı gösterir."""
    if _current_job is not None:
        raise HTTPException(409, 'Sunucu şu an bir videoyu işliyor, bitince tekrar dene.')


@app.post('/code')
async def update_code(request: Request):
    """Kod güncellemesi (sunucu + runpod klasörleri tar.gz). Açar ve sunucuyu
    yeniden başlatır; start.sh döngüsü 2 sn içinde tekrar kaldırır. Paket ve
    model kurulumu tekrarlanmaz (10 dk yerine 10 sn)."""
    _require_idle()
    tmp = os.path.join(HOME, 'code_update.tar.gz')
    with open(tmp, 'wb') as out:
        async for chunk in request.stream():
            out.write(chunk)
    r = subprocess.run(['tar', 'xzf', tmp, '-C', HOME], capture_output=True, text=True)
    os.remove(tmp)
    if r.returncode != 0:
        raise HTTPException(400, 'Arşiv açılamadı: ' + r.stderr[-500:])

    def _restart():
        time.sleep(0.5)
        os._exit(3)
    threading.Thread(target=_restart, daemon=True).start()
    return {'ok': True, 'restarting': True}


@app.post('/models/clear')
def models_clear():
    """Belleği boşaltır (yüklü modelleri atar). Bir sonraki işte yeniden yüklenir."""
    _require_idle()
    with ff_bridge._lock:
        ff_bridge.clear_models()
    return {'ok': True}


@app.get('/wan/status')
def wan_status():
    return wan_body.status()


@app.post('/wan/setup')
def wan_setup():
    """Tüm vücut modunun kurulumunu (paketler + ~50 GB model) arka planda başlatır."""
    if not wan_body.available():
        raise HTTPException(400, 'Tüm vücut modu bu sunucuda kullanılamaz (CUDA gerekli).')
    started = wan_body.start_setup()
    return {'ok': True, 'started': started, **wan_body.status()}


# ----------------------------------------------------------------------------- videolar

@app.post('/videos')
async def upload_video(file: UploadFile = File(...)):
    ext = os.path.splitext(file.filename or '')[1].lower() or '.mp4'
    if ext not in ('.mp4', '.mov', '.m4v', '.webm', '.avi', '.mkv'):
        raise HTTPException(400, 'Desteklenmeyen video formatı: ' + ext)
    vid = uuid.uuid4().hex[:12]
    path = os.path.join(VIDEOS, vid + ext)
    with open(path, 'wb') as out:
        while True:
            chunk = await file.read(1024 * 1024)
            if not chunk:
                break
            out.write(chunk)
    info = ff_bridge.video_info(path)
    if info['frame_count'] <= 0:
        os.remove(path)
        raise HTTPException(400, 'Video okunamadı')
    meta = {'id': vid, 'path': path, 'name': file.filename, **info}
    _write_json(os.path.join(VIDEOS, vid + '.json'), meta)
    return meta


def _video_meta(vid: str) -> dict:
    meta = _read_json(os.path.join(VIDEOS, vid + '.json'), None)
    if not meta or not os.path.isfile(meta.get('path', '')):
        raise HTTPException(404, 'Video bulunamadı')
    return meta


@app.get('/videos')
def list_videos():
    items = []
    for name in sorted(os.listdir(VIDEOS)):
        if name.endswith('.json'):
            m = _read_json(os.path.join(VIDEOS, name), None)
            if m:
                items.append({k: m[k] for k in ('id', 'name', 'fps', 'frame_count', 'width', 'height', 'duration')})
    return items


@app.delete('/videos/{vid}')
def delete_video(vid: str):
    meta = _video_meta(vid)
    for p in (meta['path'], os.path.join(VIDEOS, vid + '.json')):
        if os.path.exists(p):
            os.remove(p)
    return {'ok': True}


@app.get('/videos/{vid}/frame/{frame_number}')
def video_frame(vid: str, frame_number: int, max_side: int = 960):
    meta = _video_meta(vid)
    frame_number = max(0, min(frame_number, meta['frame_count'] - 1))
    _require_idle()
    with ff_bridge._lock:
        frame = ff_bridge.read_video_frame(meta['path'], frame_number)
    if frame is None:
        raise HTTPException(404, 'Kare okunamadı')
    h, w = frame.shape[:2]
    if max(h, w) > max_side:
        s = max_side / float(max(h, w))
        frame = cv2.resize(frame, (int(w * s), int(h * s)))
    ok, buf = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
    return Response(content=buf.tobytes(), media_type='image/jpeg')


@app.get('/videos/{vid}/faces/{frame_number}')
def video_faces(vid: str, frame_number: int):
    meta = _video_meta(vid)
    frame_number = max(0, min(frame_number, meta['frame_count'] - 1))
    _require_idle()
    result = ff_bridge.detect_faces_in_frame(meta['path'], frame_number)
    frame = result.pop('frame', None)
    for face in result['faces']:
        if frame is not None:
            x1, y1, x2, y2 = [int(v) for v in face['bbox']]
            pad = int(0.25 * max(x2 - x1, y2 - y1))
            crop = frame[max(0, y1 - pad):y2 + pad, max(0, x1 - pad):x2 + pad]
            face['thumb'] = _jpeg_b64(crop, 128) if crop.size else ''
    return result


# ----------------------------------------------------------------------------- yüz kütüphanesi

@app.post('/faces')
async def add_face(file: UploadFile = File(...), name: str = Form('')):
    data = await file.read()
    import numpy as np
    arr = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
    if arr is None:
        raise HTTPException(400, 'Fotoğraf okunamadı')
    fid = uuid.uuid4().hex[:12]
    path = os.path.join(FACES, fid + '.jpg')
    # Çok büyük fotoğrafları 1600px'e indir (hız), kalite yüksek kalsın
    h, w = arr.shape[:2]
    if max(h, w) > 1600:
        s = 1600 / float(max(h, w))
        arr = cv2.resize(arr, (int(w * s), int(h * s)), interpolation=cv2.INTER_AREA)
    cv2.imwrite(path, arr, [cv2.IMWRITE_JPEG_QUALITY, 95])
    meta = {'id': fid, 'name': name or fid, 'path': path, 'created': time.time(), 'thumb': _jpeg_b64(arr, 160)}
    _write_json(os.path.join(FACES, fid + '.json'), meta)
    return {k: meta[k] for k in ('id', 'name', 'thumb')}


@app.get('/faces')
def list_faces():
    items = []
    for name in sorted(os.listdir(FACES)):
        if name.endswith('.json'):
            m = _read_json(os.path.join(FACES, name), None)
            if m and os.path.isfile(m.get('path', '')):
                items.append({k: m[k] for k in ('id', 'name', 'thumb')})
    return items


@app.delete('/faces/{fid}')
def delete_face(fid: str):
    for p in (os.path.join(FACES, fid + '.jpg'), os.path.join(FACES, fid + '.json')):
        if os.path.exists(p):
            os.remove(p)
    return {'ok': True}


# ----------------------------------------------------------------------------- işler

class JobRequest(BaseModel):
    video_id: str
    frame_number: int
    face_index: int
    face_ids: List[str]            # kütüphaneden 1+ yüz (birden fazla verilirse FaceFusion ortalamasını alır)
    preset: str = presets.DEFAULT_PRESET
    watermark: bool = True
    trim_start: Optional[int] = None   # kare numarası (dahil); None = baştan
    trim_end: Optional[int] = None     # kare numarası (hariç); None = sona kadar
    mode: str = 'yuz'                  # 'yuz' = sadece yüz (FaceFusion), 'vucut' = tüm vücut (Wan2.2 Animate)


_jobs: dict = {}
_job_lock = threading.Lock()
_queue: List[str] = []
_current_job: Optional[str] = None

STAGE_MAP = {
    'downloading': ('Model indiriliyor', 0, 3),
    'analysing': ('Video inceleniyor', 3, 6),
    'extracting': ('Kareler çıkarılıyor', 6, 12),
    'processing': ('Yüz değiştiriliyor', 12, 92),
    'merging': ('Video birleştiriliyor', 92, 97),
}


def _set(job_id: str, **kw):
    with _job_lock:
        _jobs[job_id].update(kw)
        _write_json(os.path.join(JOBS, job_id, 'job.json'), _jobs[job_id])


def _worker():
    global _current_job
    while True:
        job_id = None
        with _job_lock:
            if _queue:
                job_id = _queue.pop(0)
        if not job_id:
            time.sleep(0.5)
            continue
        _current_job = job_id
        try:
            _run_job(job_id)
        except Exception as e:
            _set(job_id, status='error', error='Sunucu hatası: %s' % e, percent=0)
            _append_log(job_id, 'HATA: %s\n' % e)
        _current_job = None


def _append_log(job_id: str, text: str):
    try:
        with open(os.path.join(JOBS, job_id, 'log.txt'), 'a', encoding='utf-8') as f:
            f.write(text)
    except Exception:
        pass


def _run_job(job_id: str):
    job = _jobs[job_id]
    job_dir = os.path.join(JOBS, job_id)
    out_raw = os.path.join(job_dir, 'raw.mp4')
    out_final = os.path.join(job_dir, 'sonuc.mp4')
    temp_dir = os.path.join(TEMP, job_id)
    os.makedirs(temp_dir, exist_ok=True)
    if job.get('mode') == 'vucut':
        return _run_body_job(job_id, job, job_dir, out_raw, out_final, temp_dir)

    argv = presets.build_args(
        job['preset'], job['source_paths'], job['target_path'], out_raw,
        job['frame_number'], job['face_index'], temp_dir, os.path.join(job_dir, 'ffjobs'),
        execution_provider=PROVIDER, thread_count=THREADS,
        trim_start=job.get('trim_start'), trim_end=job.get('trim_end'))
    _set(job_id, status='running', stage='Hazırlık (modeller yükleniyor)', percent=1, error=None,
         started=time.time(), command='facefusion.py ' + ' '.join(argv))
    _append_log(job_id, 'facefusion.py ' + ' '.join(argv) + '\n')

    last = {'key': None}

    def on_progress(desc: str, n: int, total: int):
        stage = (desc or '').lower().strip(': ')
        label, lo, hi = STAGE_MAP.get(stage, (stage or 'İşleniyor', 12, 92))
        pct = int(100.0 * n / total) if total else 0
        overall = lo + (hi - lo) * pct / 100.0
        key = (stage, pct)
        if key == last['key']:
            return
        last['key'] = key
        _set(job_id, stage=label, percent=round(overall, 1), stage_percent=pct, done=int(n), total=int(total or 0))

    def is_cancelled() -> bool:
        return _jobs[job_id].get('status') == 'cancelled'

    t0 = time.time()
    with ff_bridge._lock:
        code = ff_bridge.run_swap(argv, on_progress, is_cancelled, log=lambda s: _append_log(job_id, s))
    _append_log(job_id, 'Bitti: kod %s, %.0f sn\n' % (code, time.time() - t0))

    shutil.rmtree(temp_dir, ignore_errors=True)
    if is_cancelled():
        return
    if code != 0 or not os.path.isfile(out_raw):
        msg = {2: 'Ön kontrol başarısız: kaynak fotoğrafta yüz bulunamadı ya da model yüklenemedi (log\'a bak).',
               3: 'İçerik filtresi bu videoyu işlemeyi reddetti.',
               4: 'İşlem durduruldu.'}.get(code, 'FaceFusion hata verdi (kod %s). log\'a bak.' % code)
        _set(job_id, status='error', error=msg, percent=0)
        return

    _finish_job(job_id, job, out_raw, out_final, t0, int(_jobs[job_id].get('total') or 0))


def _run_body_job(job_id: str, job: dict, job_dir: str, out_raw: str, out_final: str, temp_dir: str):
    """Tüm vücut: seçilen kişinin yüz+saç+kıyafet+vücudu referans fotoğraftaki kişiyle değişir."""
    t0 = time.time()
    _set(job_id, status='running', stage='Hazırlık', percent=1, error=None, started=time.time())

    def is_cancelled() -> bool:
        return _jobs[job_id].get('status') == 'cancelled'

    def on_progress(stage: str, pct: float):
        _set(job_id, stage=stage, percent=round(pct, 1), stage_percent=int(pct))

    log = lambda s: _append_log(job_id, s)  # noqa: E731
    try:
        if not wan_body.is_ready():
            wan_body.wait_setup(60 * 40, is_cancelled, lambda s: _set(job_id, stage=s, percent=1))
        # Dokunulan yüzün merkezi (referans karede, normalize)
        det = ff_bridge.detect_faces_in_frame(job['target_path'], int(job['frame_number']))  # kendi kilidini alır
        faces = det.get('faces') or []
        if not faces:
            raise RuntimeError('Referans karede yüz bulunamadı')
        face = faces[min(int(job['face_index']), len(faces) - 1)]
        x1, y1, x2, y2 = face['bbox_norm']
        cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
        _append_log(job_id, 'kişi: yüz %d merkez (%.3f, %.3f)\n' % (job['face_index'], cx, cy))
        info = wan_body.run_body_swap(job['target_path'], job['source_paths'][0], out_raw, temp_dir, job['preset'],
                                      int(job['frame_number']), cx, cy, on_progress, is_cancelled, log,
                                      trim_start=job.get('trim_start'), trim_end=job.get('trim_end'))
    except Exception as e:
        shutil.rmtree(temp_dir, ignore_errors=True)
        if is_cancelled():
            return
        _append_log(job_id, 'HATA: %s\n' % e)
        _set(job_id, status='error', error=str(e), percent=0)
        return
    shutil.rmtree(temp_dir, ignore_errors=True)
    _append_log(job_id, 'Bitti: %s, %.0f sn\n' % (info, time.time() - t0))
    _finish_job(job_id, job, out_raw, out_final, t0, int(info.get('frames') or 0))


def _finish_job(job_id: str, job: dict, out_raw: str, out_final: str, t0: float, frames: int):
    if job.get('watermark', True):
        _set(job_id, stage='Filigran ekleniyor', percent=97)
        if not _add_watermark(out_raw, out_final, os.path.dirname(out_final)):
            shutil.copyfile(out_raw, out_final)
    else:
        shutil.move(out_raw, out_final)
    if os.path.exists(out_raw) and os.path.exists(out_final) and out_raw != out_final:
        os.remove(out_raw)
    _set(job_id, status='done', stage='Tamamlandı', percent=100, finished=time.time(),
         elapsed=round(time.time() - t0), frames=frames, result_size=os.path.getsize(out_final))


def _add_watermark(src: str, dst: str, job_dir: str) -> bool:
    """Sağ alt köşeye küçük, yarı saydam 'AI' etiketi (ffmpeg overlay)."""
    try:
        import numpy as np
        wm = np.zeros((40, 110, 4), dtype=np.uint8)
        cv2.rectangle(wm, (0, 0), (109, 39), (0, 0, 0, 110), -1)
        cv2.putText(wm, 'AI', (8, 30), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255, 200), 2, cv2.LINE_AA)
        wm_path = os.path.join(job_dir, 'wm.png')
        cv2.imwrite(wm_path, wm)
        cmd = ['ffmpeg', '-y', '-loglevel', 'error', '-i', src, '-i', wm_path,
               '-filter_complex', '[0:v][1:v]overlay=W-w-16:H-h-16', '-c:v', 'libx264', '-preset', 'veryfast',
               '-crf', '18', '-pix_fmt', 'yuv420p', '-c:a', 'copy', '-movflags', '+faststart', dst]
        return subprocess.call(cmd) == 0 and os.path.isfile(dst)
    except Exception:
        return False


@app.post('/jobs')
def create_job(req: JobRequest):
    meta = _video_meta(req.video_id)
    if req.preset not in presets.PRESETS:
        raise HTTPException(400, 'Bilinmeyen ön ayar: ' + req.preset)
    if not req.face_ids:
        raise HTTPException(400, 'En az bir yüz seçmelisin')
    if req.mode not in ('yuz', 'vucut'):
        raise HTTPException(400, 'Bilinmeyen mod: ' + req.mode)
    if req.mode == 'vucut':
        if not wan_body.available():
            raise HTTPException(400, 'Tüm vücut modu bu sunucuda yok (RunPod H100/A100 gerekir).')
        if not wan_body.is_ready():
            wan_body.start_setup()
    sources = []
    for fid in req.face_ids:
        p = os.path.join(FACES, fid + '.jpg')
        if not os.path.isfile(p):
            raise HTTPException(404, 'Yüz bulunamadı: ' + fid)
        sources.append(p)
    job_id = time.strftime('%Y%m%d-%H%M%S') + '-' + uuid.uuid4().hex[:4]
    job_dir = os.path.join(JOBS, job_id)
    os.makedirs(job_dir, exist_ok=True)
    job = {
        'id': job_id, 'status': 'queued', 'stage': 'Sırada', 'percent': 0, 'error': None,
        'video_id': req.video_id, 'video_name': meta.get('name'), 'target_path': meta['path'],
        'frame_number': int(req.frame_number), 'face_index': int(req.face_index),
        'face_ids': req.face_ids, 'source_paths': sources, 'preset': req.preset,
        'watermark': req.watermark, 'created': time.time(),
        'trim_start': req.trim_start, 'trim_end': req.trim_end,
        'frame_total': int(meta.get('frame_count') or 0), 'mode': req.mode,
    }
    if job['trim_start'] is not None and job['trim_end'] is not None and job['trim_end'] <= job['trim_start']:
        raise HTTPException(400, 'Bitiş karesi başlangıçtan büyük olmalı')
    if job['trim_start'] is not None and not (job['trim_start'] <= job['frame_number']) or \
       job['trim_end'] is not None and not (job['frame_number'] < job['trim_end']):
        # referans kare seçilen aralığın dışındaysa aralığın ilk karesini referans yap
        job['frame_number'] = int(job['trim_start'] or 0)
    with _job_lock:
        _jobs[job_id] = job
        _queue.append(job_id)
    _write_json(os.path.join(job_dir, 'job.json'), job)
    return _public(job)


def _public(job: dict) -> dict:
    keys = ('id', 'status', 'stage', 'percent', 'stage_percent', 'done', 'total', 'error', 'video_id', 'video_name',
            'frame_number', 'face_index', 'face_ids', 'preset', 'watermark', 'created', 'started', 'finished', 'elapsed', 'frames', 'result_size', 'trim_start', 'trim_end', 'frame_total', 'mode')
    out = {k: job.get(k) for k in keys}
    if job.get('started') and job.get('status') == 'running' and job.get('percent', 0) > 12:
        elapsed = time.time() - job['started']
        out['eta_seconds'] = int(elapsed * (100 - job['percent']) / max(job['percent'], 1))
    return out


@app.get('/jobs')
def list_jobs():
    with _job_lock:
        items = sorted(_jobs.values(), key=lambda j: j.get('created', 0), reverse=True)
    return [_public(j) for j in items[:50]]


@app.get('/jobs/{job_id}')
def get_job(job_id: str):
    job = _jobs.get(job_id)
    if not job:
        raise HTTPException(404, 'İş bulunamadı')
    return _public(job)


@app.get('/jobs/{job_id}/result')
def job_result(job_id: str):
    job = _jobs.get(job_id)
    if not job or job.get('status') != 'done':
        raise HTTPException(404, 'Sonuç hazır değil')
    path = os.path.join(JOBS, job_id, 'sonuc.mp4')
    return FileResponse(path, media_type='video/mp4', filename='faceswap_%s.mp4' % job_id)


@app.get('/jobs/{job_id}/log')
def job_log(job_id: str):
    path = os.path.join(JOBS, job_id, 'log.txt')
    if not os.path.isfile(path):
        raise HTTPException(404, 'Log yok')
    with open(path, 'rb') as f:
        data = f.read()[-20000:]
    return Response(content=data.replace(b'\r', b'\n'), media_type='text/plain; charset=utf-8')


@app.post('/jobs/{job_id}/cancel')
def cancel_job(job_id: str):
    job = _jobs.get(job_id)
    if not job:
        raise HTTPException(404, 'İş bulunamadı')
    with _job_lock:
        if job_id in _queue:
            _queue.remove(job_id)
    if job.get('status') in ('queued', 'running'):
        _set(job_id, status='cancelled', stage='İptal edildi')
        if _current_job == job_id:
            ff_bridge.request_stop()
    return _public(_jobs[job_id])


@app.delete('/jobs/{job_id}')
def delete_job(job_id: str):
    job = _jobs.get(job_id)
    if job and job.get('status') in ('queued', 'running'):
        cancel_job(job_id)
    with _job_lock:
        _jobs.pop(job_id, None)
    shutil.rmtree(os.path.join(JOBS, job_id), ignore_errors=True)
    return {'ok': True}


# ----------------------------------------------------------------------------- başlangıç

def _load_jobs():
    for name in sorted(os.listdir(JOBS)):
        j = _read_json(os.path.join(JOBS, name, 'job.json'), None)
        if j:
            if j.get('status') in ('queued', 'running'):
                j['status'] = 'error'
                j['error'] = 'Sunucu yeniden başlatıldı'
            _jobs[j['id']] = j


def _startup():
    _load_jobs()
    threading.Thread(target=_worker, daemon=True).start()
    threading.Thread(target=_warm, daemon=True).start()
    threading.Thread(target=_idle_watcher, daemon=True).start()
    if os.environ.get('FACESWAP_WAN', '1') == '1' and wan_body.available() and not wan_body.is_ready():
        print('[FaceSwap] Tüm vücut modu arka planda kuruluyor...')
        wan_body.start_setup()


def _warm():
    try:
        ff_bridge.init(PROVIDER)
        print('[FaceSwap] Yüz dedektörü hazır (%s).' % PROVIDER)
    except Exception as e:
        print('[FaceSwap] UYARI: dedektör başlatılamadı: %s' % e)


if __name__ == '__main__':
    import uvicorn
    print('[FaceSwap] Sunucu başlıyor. Telefondan bağlanmak için: http://%s:%d' % ((_lan_ips() or ['<bilgisayar-ip>'])[0], PORT))
    uvicorn.run(app, host='0.0.0.0', port=PORT, log_level='warning')
