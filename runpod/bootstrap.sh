mkdir -p /workspace/fs/data /workspace/fs/logs; cd /workspace/fs; echo "Kod bekleniyor" > data/boot_stage.txt
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
for i in $(seq 1 150); do [ -f code_upload.tar.gz ] && break; sleep 2; done
if [ -f code_upload.tar.gz ]; then sleep 1; tar xzf code_upload.tar.gz && rm -f code_upload.tar.gz; fi
if [ -f runpod/start.sh ]; then bash runpod/start.sh; fi
python3 /tmp/boot_status.py & echo $! > /tmp/boot_status.pid
grep -q HATA data/boot_stage.txt || echo "HATA: sunucu durdu (log: /log)" > data/boot_stage.txt; sleep infinity
