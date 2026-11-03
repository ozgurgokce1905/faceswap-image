#!/bin/bash
# Tüm vücut modu (Wan2.2-Animate-14B, replacement) kurulumu. Tekrar çalıştırılabilir (idempotent).
# Kullanım: bash wan_setup.sh   (ortam: WAN_HOME=/workspace/wan, FS=/workspace/fs)
set -u
WAN_HOME=${WAN_HOME:-/workspace/wan}
FS=${FACESWAP_HOME:-/workspace/fs}
STAGE=$FS/data/wan_stage.txt
mkdir -p $WAN_HOME $FS/data $FS/logs
LOG=$FS/logs/wan_setup.log
exec > >(tee -a "$LOG") 2>&1
step() { echo "[WAN] $*"; echo "$*" > $STAGE; }
fail() { echo "[WAN] HATA: $*"; echo "HATA: $*" > $STAGE; exit 1; }
echo "===== $(date -u) wan kurulum ====="
# Aynı anda ikinci kurulum çalışmasın (sunucu yeniden başlarsa tekrar tetiklenebilir)
exec 9>$WAN_HOME/.setup.lock
flock -n 9 || { echo "[WAN] başka bir kurulum zaten çalışıyor, çıkılıyor"; exit 0; }

# 1) Kod (sabit sürüm)
WAN_COMMIT=1ea34ff48f87168174e12956e200b1d908b1c5ff
if [ ! -f $WAN_HOME/Wan2.2/generate.py ]; then
  step "Wan2.2 kodu indiriliyor"
  git clone -q https://github.com/Wan-Video/Wan2.2.git $WAN_HOME/Wan2.2 || fail "Wan2.2 indirilemedi"
  git -C $WAN_HOME/Wan2.2 checkout -q $WAN_COMMIT
fi

# 2) Python ortamı: imajdaki torch 2.8 (cu128) kullanılır, geri kalanı venv'e kurulur
PY=$WAN_HOME/venv/bin/python
if [ -f $WAN_HOME/venv/.ok ] && ! $PY -c "import torch, flash_attn, diffusers, sam2, onnxruntime, cv2, numpy; assert numpy.__version__ < '2'" 2>/dev/null; then
  echo "wan venv bozuk, yeniden kurulacak"; rm -f $WAN_HOME/venv/.ok
fi
if [ ! -f $WAN_HOME/venv/.ok ]; then
  step "Tüm vücut paketleri kuruluyor (ilk sefer ~5 dk)"
  rm -rf $WAN_HOME/venv
  # torch'u olan python'u bul (RunPod imajında sistem python'unda ya da /venv altında olabilir)
  PYBIN=""
  for cand in python3.12 python3 /venv/main/bin/python /opt/venv/bin/python /workspace/venv/bin/python; do
    c=$(command -v $cand 2>/dev/null || echo $cand)
    [ -x "$c" ] && $c -c "import torch, sys; assert sys.version_info >= (3,12)" 2>/dev/null && { PYBIN=$c; break; }
  done
  NEED_TORCH=0
  if [ -z "$PYBIN" ]; then PYBIN=$(command -v python3.12 || command -v python3); NEED_TORCH=1; echo "torch'lu python yok, venv'e kurulacak"; fi
  $PYBIN -m venv --system-site-packages $WAN_HOME/venv || fail "venv oluşturulamadı"
  $PY -m pip install -q --upgrade pip
  PIP=$WAN_HOME/venv/bin/pip
  $PIP install --no-input -q setuptools wheel packaging ninja  # SAM2 kaynak kurulumu için şart (hazır imajda sistem setuptools yok)
  if [ $NEED_TORCH = 1 ]; then
    step "torch kuruluyor (~3 GB)"
    $PIP install --no-input "torch==2.8.*" "torchvision" "torchaudio" --index-url https://download.pytorch.org/whl/cu128 > /tmp/wan_torch.log 2>&1 || { tail -n 20 /tmp/wan_torch.log; fail "torch kurulamadı"; }
  fi
  $PY -c "import torch; print('torch', torch.__version__, 'cuda', torch.cuda.is_available())" || fail "torch import edilemedi"
  # flash-attn: hazır wheel (derleme saatler sürer). torch 2.8 / py3.12 / cu12
  PYTAG=$($PY -c "import sys;print(f'cp{sys.version_info[0]}{sys.version_info[1]}')")
  TORCHTAG=$($PY -c "import torch;v=torch.__version__.split('+')[0].split('.');print(f'torch{v[0]}.{v[1]}')")
  ABI=$($PY -c "import torch;print('TRUE' if torch.compiled_with_cxx11_abi() else 'FALSE')")
  FA_OK=0
  for url in \
    "https://github.com/Dao-AILab/flash-attention/releases/download/v2.8.3/flash_attn-2.8.3%2Bcu12${TORCHTAG}cxx11abi${ABI}-${PYTAG}-${PYTAG}-linux_x86_64.whl" \
    "https://huggingface.co/strangertoolshf/flash_attention_2_wheelhouse/resolve/main/wheelhouse-flash_attn-2.8.3/linux_x86_64/${TORCHTAG}/cu12/${PYTAG}/flash_attn-2.8.3+cu12${TORCHTAG}cxx11abi${ABI}-${PYTAG}-${PYTAG}-linux_x86_64.whl" \
    "https://huggingface.co/strangertoolshf/flash_attention_2_wheelhouse/resolve/main/wheelhouse-flash_attn-2.8.3/linux_x86_64/${TORCHTAG}/cu12/abi${ABI}/${PYTAG}/flash_attn-2.8.3+cu12${TORCHTAG}cxx11abi${ABI}-${PYTAG}-${PYTAG}-linux_x86_64.whl"; do
    echo "flash-attn deneniyor: $url"
    if $PIP install --no-input "$url" > /tmp/fa.log 2>&1; then FA_OK=1; break; else tail -n 3 /tmp/fa.log; fi
  done
  [ $FA_OK = 1 ] || fail "flash-attn hazır wheel bulunamadı ($TORCHTAG $PYTAG abi$ABI)"
  grep -v -E '^(torch|torchvision|torchaudio|flash_attn)' $WAN_HOME/Wan2.2/requirements.txt > /tmp/wan_req.txt
  $PIP install --no-input -r /tmp/wan_req.txt "numpy<2" "onnxruntime-gpu==1.24.4" peft pandas matplotlib loguru sentencepiece moviepy hydra-core omegaconf huggingface_hub > /tmp/wan_pip1.log 2>&1 || { tail -n 30 /tmp/wan_pip1.log; fail "pip (Wan paketleri) başarısız"; }
  tail -n 1 /tmp/wan_pip1.log
  SAM2_BUILD_CUDA=0 $PIP install --no-input --no-build-isolation "git+https://github.com/facebookresearch/sam2.git@0e78a118995e66bb27d78518c4bd9a3e95b4e266" > /tmp/wan_pip2.log 2>&1 || { tail -n 30 /tmp/wan_pip2.log; fail "pip (SAM2) başarısız"; }
  $PY -c "import torch, flash_attn, diffusers, sam2, onnxruntime, cv2, numpy, easydict, imageio; assert numpy.__version__ < '2'; print('wan paketleri OK, torch', torch.__version__, 'numpy', numpy.__version__)" || fail "Wan paketleri eksik"
  # onnxruntime (poz/dedektör) CUDA kütüphanelerini torch'un nvidia paketlerinden bulsun
  SITE=$($PY -c "import site;print(site.getsitepackages()[0])")
  cat > $SITE/sitecustomize.py <<'PYC'
try:
    import onnxruntime as _ort
    if hasattr(_ort, 'preload_dlls'):
        _ort.preload_dlls()
except Exception:
    pass
PYC
  touch $WAN_HOME/venv/.ok
fi

# 3) Modeller (~50 GB; FLUX hariç)
CK=$WAN_HOME/Wan2.2-Animate-14B
if [ ! -f $CK/.ok ]; then
  step "Tüm vücut modelleri indiriliyor (~50 GB, ilk sefer 5-15 dk)"
  export HF_HUB_ENABLE_HF_TRANSFER=0
  $PY -m huggingface_hub.commands.huggingface_cli download Wan-AI/Wan2.2-Animate-14B --local-dir $CK --exclude "process_checkpoint/FLUX*" "*.md" "assets/*" > /tmp/wan_dl.log 2>&1 \
    || $WAN_HOME/venv/bin/huggingface-cli download Wan-AI/Wan2.2-Animate-14B --local-dir $CK --exclude "process_checkpoint/FLUX*" "*.md" "assets/*" > /tmp/wan_dl.log 2>&1 \
    || { tail -n 20 /tmp/wan_dl.log; fail "Model indirilemedi"; }
  [ -f $CK/process_checkpoint/det/yolov10m.onnx ] || fail "process_checkpoint eksik"
  [ -f $CK/process_checkpoint/sam2/sam2_hiera_large.pt ] || fail "sam2 eksik"
  ls $CK/*.safetensors >/dev/null 2>&1 || ls $CK/*.pth >/dev/null 2>&1 || fail "DiT ağırlıkları eksik"
  touch $CK/.ok
fi
du -sh $CK 2>/dev/null
step "HAZIR"
echo "[WAN] kurulum tamam"
