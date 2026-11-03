r"""
cli_test.py - Telefon uygulaması olmadan sunucuyu komut satırından denemek için.

Örnek:
  python sunucu\cli_test.py --video C:\video.mp4 --face C:\yuz.jpg
  python sunucu\cli_test.py --video C:\video.mp4 --face C:\yuz.jpg --frame 120 --index 1 --preset ultra

Adımlar: videoyu yükler -> seçilen karedeki yüzleri listeler -> yüzü kütüphaneye ekler
         -> işi başlatır -> yüzde ilerlemeyi gösterir -> sonucu indirir.
"""
import argparse
import os
import sys
import time

import requests


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument('--server', default='http://127.0.0.1:8765')
    ap.add_argument('--video', required=True)
    ap.add_argument('--face', required=True, help='kütüphaneye eklenecek yüz fotoğrafı')
    ap.add_argument('--frame', type=int, default=0, help='kişiyi seçeceğin kare numarası')
    ap.add_argument('--index', type=int, default=None, help='o karede soldan kaçıncı yüz (0 = ilk). Verilmezse listeler ve sorar')
    ap.add_argument('--preset', default='kaliteli', choices=['hizli', 'kaliteli', 'ultra'])
    ap.add_argument('--no-watermark', action='store_true')
    ap.add_argument('--out', default=None)
    a = ap.parse_args()
    s = a.server.rstrip('/')

    r = requests.get(s + '/health', timeout=10).json()
    print('Sunucu:', r.get('provider'), r.get('onnxruntime'), r.get('providers'))

    print('Video yükleniyor...')
    with open(a.video, 'rb') as f:
        v = requests.post(s + '/videos', files={'file': (os.path.basename(a.video), f, 'video/mp4')}, timeout=600).json()
    print('  id=%s fps=%.2f kare=%d %dx%d süre=%.1fs' % (v['id'], v['fps'], v['frame_count'], v['width'], v['height'], v['duration']))

    faces = requests.get('%s/videos/%s/faces/%d' % (s, v['id'], a.frame), timeout=120).json()
    print('Kare %d: %d yüz bulundu' % (a.frame, len(faces['faces'])))
    for fc in faces['faces']:
        x1, y1, x2, y2 = [int(t) for t in fc['bbox']]
        print('  [%d] x=%d y=%d w=%d h=%d skor=%.2f %s' % (fc['index'], x1, y1, x2 - x1, y2 - y1, fc['score'], fc.get('gender', '')))
    if not faces['faces']:
        print('Bu karede yüz yok, başka --frame dene.')
        return 1
    index = a.index
    if index is None:
        index = int(input('Hangi yüz değişsin? [0]: ') or '0')

    with open(a.face, 'rb') as f:
        face = requests.post(s + '/faces', files={'file': (os.path.basename(a.face), f, 'image/jpeg')}, data={'name': os.path.basename(a.face)}, timeout=120).json()
    print('Yüz kütüphaneye eklendi:', face['id'])

    job = requests.post(s + '/jobs', json={
        'video_id': v['id'], 'frame_number': a.frame, 'face_index': index,
        'face_ids': [face['id']], 'preset': a.preset, 'watermark': not a.no_watermark,
    }, timeout=30).json()
    if 'id' not in job:
        print('HATA:', job)
        return 1
    print('İş başladı:', job['id'], 'ön ayar:', a.preset)

    last = None
    t0 = time.time()
    while True:
        j = requests.get('%s/jobs/%s' % (s, job['id']), timeout=30).json()
        line = '%5.1f%%  %s' % (j.get('percent') or 0, j.get('stage') or '')
        if j.get('done') and j.get('total'):
            line += '  (%d/%d kare)' % (j['done'], j['total'])
        if j.get('eta_seconds') is not None:
            line += '  kalan ~%ds' % j['eta_seconds']
        if line != last:
            print('\r' + line + ' ' * 20, end='', flush=True)
            last = line
        if j['status'] in ('done', 'error', 'cancelled'):
            print()
            break
        time.sleep(1)
    print('Süre: %.0f sn' % (time.time() - t0))
    if j['status'] != 'done':
        print('Sonuç:', j['status'], j.get('error'))
        print('Log:', s + '/jobs/' + job['id'] + '/log')
        return 1

    out = a.out or os.path.splitext(a.video)[0] + '_faceswap.mp4'
    with requests.get('%s/jobs/%s/result' % (s, job['id']), stream=True, timeout=600) as r:
        with open(out, 'wb') as f:
            for chunk in r.iter_content(1024 * 1024):
                f.write(chunk)
    print('Kaydedildi:', out)
    return 0


if __name__ == '__main__':
    sys.exit(main())
