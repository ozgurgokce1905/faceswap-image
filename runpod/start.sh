#!/bin/bash
# FaceSwap - RunPod pod başlangıç scripti.
# Kalıcı veri /workspace (network volume) üzerinde: venv, facefusion, modeller, yüz kütüphanesi, sonuçlar.
# İlk açılış ~10-15 dk (kurulum + model indirme), sonrakiler ~1-2 dk.
set -u
fail() { echo "[BOOT] HATA: $*"; echo "HATA: $* (log: /log)" > /workspace/fs/data/boot_stage.txt; sleep infinity; }
W=/workspace
FS=$W/fs
mkdir -p $FS/data $FS/logs
LOG=$FS/logs/boot.log
exec > >(tee -a "$LOG") 2>&1
echo "===== $(date -u) açılış ====="
step() { echo "[BOOT] $*"; echo "$*" > $FS/data/boot_stage.txt; }
step "Başlıyor"


# 1) Kod bootstrap tarafından /workspace/fs altına açıldı
[ -f $FS/sunucu/server.py ] || { step "HATA: sunucu kodu yok"; sleep infinity; }

# 2) ffmpeg (statik, volume'da kalıcı)
if [ ! -x $W/bin/ffmpeg ]; then
  step "ffmpeg kuruluyor"
  mkdir -p $W/bin && cd /tmp
  if curl -fsSL -o ff.tar.xz https://johnvansickle.com/ffmpeg/releases/ffmpeg-release-amd64-static.tar.xz; then
    tar xf ff.tar.xz && cp ffmpeg-*-static/ffmpeg ffmpeg-*-static/ffprobe $W/bin/
  else
    apt-get update -qq && apt-get install -y -qq ffmpeg && ln -sf $(which ffmpeg) $W/bin/ffmpeg && ln -sf $(which ffprobe) $W/bin/ffprobe
  fi
fi
export PATH=$W/bin:$PATH
command -v ffmpeg >/dev/null || fail "ffmpeg kurulamadı"
command -v curl >/dev/null || { apt-get update -qq && apt-get install -y -qq curl; }

# 3) FaceFusion (sabit sürüm)
FF_COMMIT=decc6c9
if [ ! -f $W/facefusion/facefusion.py ]; then
  step "FaceFusion indiriliyor"
  git clone -q https://github.com/facefusion/facefusion.git $W/facefusion || fail "FaceFusion indirilemedi"
  git -C $W/facefusion checkout -q $FF_COMMIT
fi

# 4) Python ortamı (volume'da kalıcı)
PY=$W/venv/bin/python
# Eski/yarım kalmış kurulum varsa (paketler import edilemiyorsa) baştan kur
if [ -f $W/venv/.ok ] && ! $PY -c "import numpy, onnxruntime, cv2, gradio, fastapi" 2>/dev/null; then
  echo "venv bozuk, yeniden kurulacak"; rm -f $W/venv/.ok
fi
if [ ! -f $W/venv/.ok ]; then
  step "Python paketleri kuruluyor (ilk açılış, 5-8 dk)"
  rm -rf $W/venv
  PYBIN=$(command -v python3.12 || command -v python3)
  $PYBIN -c "import sys; assert sys.version_info >= (3,12)" || fail "Python 3.12 yok (image: $($PYBIN --version))"
  $PYBIN -m venv $W/venv || { apt-get update -qq && apt-get install -y -qq python3-venv python3.12-venv >/dev/null 2>&1; $PYBIN -m venv $W/venv || fail "venv oluşturulamadı"; }
  $PY -m pip install -q --upgrade pip
  grep -v '^onnxruntime' $W/facefusion/requirements.txt > /tmp/req.txt
  $W/venv/bin/pip install --no-input -r /tmp/req.txt "onnxruntime-gpu==1.24.4" > /tmp/pip1.log 2>&1 || { tail -n 30 /tmp/pip1.log; fail "pip (FaceFusion paketleri) başarısız"; }
  tail -n 2 /tmp/pip1.log
  $W/venv/bin/pip install --no-input nvidia-cuda-runtime-cu12 nvidia-cudnn-cu12 nvidia-cublas-cu12 nvidia-cufft-cu12 \
      nvidia-curand-cu12 nvidia-cuda-nvrtc-cu12 nvidia-nvjitlink-cu12 fastapi uvicorn python-multipart requests > /tmp/pip2.log 2>&1 || { tail -n 30 /tmp/pip2.log; fail "pip (nvidia/fastapi) başarısız"; }
  tail -n 2 /tmp/pip2.log
  $PY -c "import onnxruntime, cv2, numpy, gradio, fastapi" || fail "Python paketleri eksik (pip hatası, log'a bak)"
  touch $W/venv/.ok
fi
SITE=$($PY -c "import site;print(site.getsitepackages()[0])")
cp $FS/sunucu/sitecustomize.py $SITE/sitecustomize.py

# 5) GPU testi
export FACESWAP_HOME=$FS FACESWAP_FF_DIR=$W/facefusion FACESWAP_PROVIDER=cuda FACESWAP_THREADS=${FACESWAP_THREADS:-4}
export PYTHONUNBUFFERED=1 PYTHONIOENCODING=utf-8
step "GPU testi"
$PY $FS/sunucu/gpu_test.py || echo "UYARI: GPU testi başarısız"
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader || true

# 6) Modeller (volume'da kalıcı; ilk seferde ~2.5 GB)
step "Modeller kontrol ediliyor"
$PY $FS/sunucu/warmup.py || echo "UYARI: bazı modeller indirilemedi"

# 7) Sunucu
step "Sunucu açılıyor"
[ -f /tmp/boot_status.pid ] && kill $(cat /tmp/boot_status.pid) 2>/dev/null; sleep 2
cd $FS
# Sunucu döngüsü: /code ile kod güncellenince çıkış kodu 3 verir, 2 sn sonra tekrar kalkar.
while true; do
  $PY $FS/sunucu/server.py
  rc=$?
  echo "[BOOT] sunucu çıktı (kod $rc)"
  [ "$rc" = "3" ] || break
  step "Sunucu yeniden başlıyor (kod güncellendi)"
  sleep 2
done
