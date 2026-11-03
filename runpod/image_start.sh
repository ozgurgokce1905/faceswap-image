#!/bin/bash
# Hazır imajla açılan pod'un başlangıç scripti. Kurulum yok: kod /opt/fs, modeller /opt/facefusion/.assets.
# Veri (yüz kütüphanesi, videolar, işler) /workspace/fs/data altında (pod diski).
set -u
W=/workspace
FS=$W/fs
mkdir -p $FS/data $FS/logs
LOG=$FS/logs/boot.log
exec > >(tee -a "$LOG") 2>&1
echo "===== $(date -u) açılış (imaj) ====="
step() { echo "[BOOT] $*"; echo "$*" > $FS/data/boot_stage.txt; }
step "Başlıyor"

# Durum sunucusu (sunucu kalkana kadar /ping 'booting' döner; /code ile kod güncellenebilir)
cat > /tmp/boot_status.py <<'PY'
import http.server, json, os
F='/workspace/fs/data/boot_stage.txt'; L='/workspace/fs/logs/boot.log'
class H(http.server.BaseHTTPRequestHandler):
    def do_POST(self):
        if self.path.startswith('/code'):
            n=int(self.headers.get('Content-Length','0')); d=self.rfile.read(n)
            open('/workspace/fs/code_upload.tar.gz','wb').write(d)
            self.send_response(200); self.end_headers(); self.wfile.write(b'{"ok":true}')
        else:
            self.send_response(404); self.end_headers()
    def do_GET(self):
        if self.path.startswith('/log'):
            try: b=open(L,'rb').read()[-6000:]
            except Exception as e: b=str(e).encode()
            self.send_response(200); self.send_header('Content-Type','text/plain; charset=utf-8'); self.end_headers(); self.wfile.write(b); return
        try: s=open(F,encoding='utf-8').read().strip()
        except Exception: s='Basliyor'
        b=json.dumps({'ok':False,'booting':True,'stage':s},ensure_ascii=False).encode()
        self.send_response(200 if self.path.startswith('/ping') else 503)
        self.send_header('Content-Type','application/json; charset=utf-8'); self.end_headers(); self.wfile.write(b)
    def log_message(self,*a): pass
http.server.ThreadingHTTPServer(('0.0.0.0',8765),H).serve_forever()
PY
python3 /tmp/boot_status.py & echo $! > /tmp/boot_status.pid

# Kod: imajdaki sürüm /workspace/fs'e kopyalanır (daha önce /code ile yüklenmiş daha yeni kod varsa korunur)
IMG_VER=$(cat /opt/fs/VERSION 2>/dev/null || echo 0)
CUR_VER=$(cat $FS/VERSION 2>/dev/null || echo -1)
if [ ! -f $FS/sunucu/server.py ] || [ "$IMG_VER" != "$CUR_VER" ]; then
  step "Kod kopyalanıyor"
  rm -rf $FS/sunucu $FS/runpod
  cp -r /opt/fs/sunucu /opt/fs/runpod $FS/
  cp /opt/fs/VERSION $FS/VERSION 2>/dev/null || echo 0 > $FS/VERSION
fi
# Açılışta /code ile kod gönderildiyse (uygulama 'Kod bekleniyor' görmeden gönderebilir) aç
if [ -f $FS/code_upload.tar.gz ]; then tar xzf $FS/code_upload.tar.gz -C $FS && rm -f $FS/code_upload.tar.gz; fi

export PATH=/opt/bin:/opt/venv/bin:$PATH
export FACESWAP_HOME=$FS FACESWAP_FF_DIR=/opt/facefusion FACESWAP_PROVIDER=cuda FACESWAP_THREADS=${FACESWAP_THREADS:-8}
export WAN_HOME=/opt/wan WAN_CKPT=/workspace/wan/Wan2.2-Animate-14B
export PYTHONUNBUFFERED=1 PYTHONIOENCODING=utf-8
PY=/opt/venv/bin/python

step "GPU testi"
$PY $FS/sunucu/gpu_test.py || echo "UYARI: GPU testi başarısız"
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader || true

step "Modeller kontrol ediliyor"
$PY $FS/sunucu/warmup.py || echo "UYARI: bazı modeller indirilemedi"

step "Sunucu açılıyor"
[ -f /tmp/boot_status.pid ] && kill $(cat /tmp/boot_status.pid) 2>/dev/null; sleep 2
cd $FS
while true; do
  $PY $FS/sunucu/server.py
  rc=$?
  echo "[BOOT] sunucu çıktı (kod $rc)"
  [ "$rc" = "3" ] || break
  step "Sunucu yeniden başlıyor (kod güncellendi)"
  sleep 2
done
python3 /tmp/boot_status.py & echo $! > /tmp/boot_status.pid
echo "HATA: sunucu durdu (log: /log)" > $FS/data/boot_stage.txt
sleep infinity
