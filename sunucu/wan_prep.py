"""Tüm vücut modu ön işleme (Wan2.2-Animate replacement) — KİŞİ SEÇİMLİ.

Resmî preprocess_data.py yalnız tek kişilik videolar için (en büyük kişiyi alır).
Burada kullanıcının dokunduğu yüzün bulunduğu kişi, kare kare takip edilerek
(bbox IoU) seçilir; poz, yüz kırpma ve SAM2 maskesi yalnız o kişi için üretilir.

Çıktı (out_dir): src_ref.png, src_pose.mp4, src_face.mp4, src_bg.mp4, src_mask.mp4
Kullanım:
  python wan_prep.py --wan /workspace/wan/Wan2.2 --ckpt .../process_checkpoint \
     --video in.mp4 --ref face.jpg --out dir --area 1280 720 --fps 30 \
     --ref-frame 0 --face-x 0.62 --face-y 0.35
"""
import argparse
import json
import os
import sys
import time

import cv2
import numpy as np


def log(*a):
    print('[PREP]', *a, flush=True)


def iou(a, b):
    x1, y1 = max(a[0], b[0]), max(a[1], b[1])
    x2, y2 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0, x2 - x1) * max(0, y2 - y1)
    ua = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / ua if ua > 0 else 0.0


def center(b):
    return ((b[0] + b[2]) / 2, (b[1] + b[3]) / 2)


class FakeDetector:
    """Pose2d'nin beklediği arayüz; her çağrıda sıradaki karenin seçili kutusunu verir."""

    def __init__(self, boxes):
        self.boxes = boxes
        self.i = 0

    def preprocess(self, image):
        return image, np.array(image.shape[:2])

    def __call__(self, img, shape):
        b = self.boxes[min(self.i, len(self.boxes) - 1)]
        self.i += 1
        return [[{'bbox': np.array([b[0], b[1], b[2], b[3], 1.0], dtype=np.float32), 'track_id': 0}]]


def detect_all(det, image, min_score=0.3):
    """Yolo ile karedeki TÜM kişileri döndürür: [x1,y1,x2,y2,score]."""
    img, shape = det.preprocess(image)
    out = det.session.run(None, {det.session.get_inputs()[0].name: img[None]})[0]
    res = det.postprocess(out[0], shape, cat_id=det.cat_id)
    boxes = []
    if len(res):
        for r in res:
            if (int(r[-1]) + 1 in det.cat_id) and r[-2] >= min_score:
                boxes.append([float(r[0]), float(r[1]), float(r[2]), float(r[3]), float(r[4])])
    return boxes


def track_person(all_boxes, ref_idx, point, frame_wh):
    """Referans karede noktayı içeren kişiyi seç, ileri-geri IoU ile takip et."""
    n = len(all_boxes)
    chosen = [None] * n
    W, H = frame_wh
    px, py = point

    def pick_at(i, prev):
        boxes = all_boxes[i]
        if not boxes:
            return prev
        if prev is None:
            inside = [b for b in boxes if b[0] <= px <= b[2] and b[1] <= py <= b[3]]
            if inside:
                return min(inside, key=lambda b: (b[2] - b[0]) * (b[3] - b[1]))
            return min(boxes, key=lambda b: (center(b)[0] - px) ** 2 + (center(b)[1] - py) ** 2)
        best = max(boxes, key=lambda b: iou(b, prev))
        if iou(best, prev) >= 0.1:
            return best
        cx, cy = center(prev)
        near = min(boxes, key=lambda b: (center(b)[0] - cx) ** 2 + (center(b)[1] - cy) ** 2)
        lim = 0.5 * max(prev[2] - prev[0], prev[3] - prev[1])
        if ((center(near)[0] - cx) ** 2 + (center(near)[1] - cy) ** 2) ** 0.5 < lim:
            return near
        return prev  # kişi bu karede bulunamadı: son kutuyu koru

    ref = pick_at(ref_idx, None)
    if ref is None:
        raise RuntimeError('Referans karede kişi bulunamadı')
    chosen[ref_idx] = ref
    prev = ref
    for i in range(ref_idx + 1, n):
        prev = pick_at(i, prev)
        chosen[i] = prev
    prev = ref
    for i in range(ref_idx - 1, -1, -1):
        prev = pick_at(i, prev)
        chosen[i] = prev
    return chosen


def write_video(path, frames, fps):
    import imageio
    w = imageio.get_writer(path, fps=fps, codec='libx264', quality=9, macro_block_size=1, ffmpeg_params=['-pix_fmt', 'yuv420p'])
    for f in frames:
        w.append_data(np.ascontiguousarray(f))
    w.close()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--wan', required=True, help='Wan2.2 repo klasörü')
    ap.add_argument('--ckpt', required=True, help='process_checkpoint klasörü')
    ap.add_argument('--video', required=True)
    ap.add_argument('--ref', required=True, help='referans kişi fotoğrafı')
    ap.add_argument('--out', required=True)
    ap.add_argument('--area', type=int, nargs=2, default=[1280, 720])
    ap.add_argument('--fps', type=int, default=30)
    ap.add_argument('--ref-frame', type=int, default=0, help='orijinal videoda referans kare no')
    ap.add_argument('--face-x', type=float, default=0.5, help='dokunulan yüz merkezi (0-1)')
    ap.add_argument('--face-y', type=float, default=0.5)
    ap.add_argument('--trim-start', type=int, default=None, help='orijinal kare no')
    ap.add_argument('--trim-end', type=int, default=None)
    ap.add_argument('--iterations', type=int, default=3)
    ap.add_argument('--k', type=int, default=7)
    args = ap.parse_args()

    prep_dir = os.path.join(args.wan, 'wan', 'modules', 'animate', 'preprocess')
    sys.path.insert(0, prep_dir)
    os.makedirs(args.out, exist_ok=True)

    import torch  # noqa
    from pose2d import Pose2d
    from pose2d_utils import AAPoseMeta
    from utils import resize_by_area, get_frame_indices, get_face_bboxes, get_aug_mask, get_mask_body_img
    from human_visualization import draw_aapose_by_meta_new
    import sam2.modeling.sam.transformer as transformer
    transformer.USE_FLASH_ATTN = False
    transformer.MATH_KERNEL_ON = True
    transformer.OLD_GPU = True
    from sam_utils import build_sam2_video_predictor

    t0 = time.time()
    # 1) Kareler (fps'e yeniden örneklenmiş, alan ölçeklenmiş)
    log('Kareler okunuyor')
    cap = cv2.VideoCapture(args.video)
    video_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    frame_num = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()
    start = args.trim_start or 0
    end = args.trim_end if args.trim_end is not None else frame_num
    end = min(end, frame_num)
    span = max(1, end - start)
    target_num = max(1, int(span / video_fps * args.fps))
    idxs = [start + i for i in get_frame_indices(span, video_fps, target_num, args.fps)]
    wanted = sorted(set(idxs))
    raw = {}
    cap = cv2.VideoCapture(args.video)
    i = 0
    last = wanted[-1]
    while i <= last:
        ok, fr = cap.read()
        if not ok:
            break
        if i in raw or i in wanted:
            if i in wanted:
                raw[i] = cv2.cvtColor(fr, cv2.COLOR_BGR2RGB)
        i += 1
    cap.release()
    prev = None
    frames = []
    for ix in idxs:
        f = raw.get(ix, prev)
        if f is None:
            continue
        prev = f
        frames.append(f)
    orig_h, orig_w = frames[0].shape[:2]
    frames = [resize_by_area(f, args.area[0] * args.area[1], divisor=16) for f in frames]
    H, W = frames[0].shape[:2]
    log(f'{len(frames)} kare, {orig_w}x{orig_h} -> {W}x{H}, fps {video_fps:.2f} -> {args.fps}')

    # Referans kare: orijinal kare no -> yeni dizideki indeks; dokunulan nokta -> yeni koordinat
    ref_idx = int(np.argmin([abs(ix - args.ref_frame) for ix in idxs]))
    # resize_by_area orantıyı koruyup kenarlardan kırpabilir/pad edebilir; merkezden ölçekle
    sx, sy = W / orig_w, H / orig_h
    s = min(sx, sy) if abs(sx - sy) > 1e-3 else sx
    px = (args.face_x * orig_w - orig_w / 2) * s + W / 2
    py = (args.face_y * orig_h - orig_h / 2) * s + H / 2
    log(f'referans kare indeksi {ref_idx}, nokta ({px:.0f},{py:.0f})')

    # 2) Kişi tespiti + takip
    log('Kişiler tespit ediliyor')
    pose2d = Pose2d(checkpoint=os.path.join(args.ckpt, 'pose2d', 'vitpose_h_wholebody.onnx'),
                    detector_checkpoint=os.path.join(args.ckpt, 'det', 'yolov10m.onnx'))
    det = pose2d.detector
    all_boxes = []
    for n, f in enumerate(frames):
        # Pose2d.load_images listeyi BGR->RGB çevirir; tespit de aynı dönüşümle yapılsın
        all_boxes.append(detect_all(det, cv2.cvtColor(f, cv2.COLOR_BGR2RGB)))
        if n % 100 == 0:
            log(f'tespit {n}/{len(frames)}')
    chosen = track_person(all_boxes, ref_idx, (px, py), (W, H))
    with open(os.path.join(args.out, 'track.json'), 'w') as fh:
        json.dump({'ref_idx': ref_idx, 'point': [px, py], 'boxes': chosen}, fh)

    # 3) Poz (yalnız seçili kişi)
    log('Poz çıkarılıyor')
    pose2d.detector = FakeDetector(chosen)
    metas = pose2d(frames)

    face_images = []
    for idx, meta in enumerate(metas):
        x1, x2, y1, y2 = get_face_bboxes(meta['keypoints_face'][:, :2], scale=1.3, image_shape=(H, W))
        face = frames[idx][y1:y2, x1:x2]
        if face.size == 0:
            face = frames[idx]
        face_images.append(cv2.resize(face, (512, 512)))

    # 4) Referans görsel
    refer_img = cv2.imread(args.ref)
    if refer_img is None:
        raise RuntimeError('Referans fotoğraf okunamadı')
    cv2.imwrite(os.path.join(args.out, 'src_ref.png'), refer_img)
    refer_img = refer_img[..., ::-1]

    aa_metas = [AAPoseMeta.from_humanapi_meta(m) for m in metas]
    cond_images = []
    for meta in aa_metas:
        canvas = np.zeros((H, W, 3), dtype=np.uint8)
        cond_images.append(draw_aapose_by_meta_new(canvas, meta))

    # 5) SAM2 maske (seçili kişinin vücut noktalarıyla)
    log('Kişi maskesi (SAM2) çıkarılıyor')
    # apply_postprocessing=False: delik doldurma SAM2'nin CUDA eklentisini (_C) ister, eklentisiz kurulumda yok; maske zaten genişletiliyor
    predictor = build_sam2_video_predictor('sam2_hiera_l.yaml', os.path.join(args.ckpt, 'sam2', 'sam2_hiera_large.pt'), apply_postprocessing=False)
    masks = get_mask(predictor, frames, 400, metas)

    bg_images, aug_masks = [], []
    for frame, mask in zip(frames, masks):
        if args.iterations > 0:
            _, m = get_mask_body_img(frame, mask, iterations=args.iterations, k=args.k)
            m = get_aug_mask(m, w_len=1, h_len=1)
        else:
            m = mask
        bg_images.append((frame * (1 - m[:, :, None])).astype(np.uint8))
        aug_masks.append(m)

    log('Videolar yazılıyor')
    write_video(os.path.join(args.out, 'src_face.mp4'), face_images, args.fps)
    write_video(os.path.join(args.out, 'src_pose.mp4'), cond_images, args.fps)
    write_video(os.path.join(args.out, 'src_bg.mp4'), bg_images, args.fps)
    write_video(os.path.join(args.out, 'src_mask.mp4'), [np.stack([m * 255] * 3, axis=2).astype(np.uint8) for m in aug_masks], args.fps)
    with open(os.path.join(args.out, 'info.json'), 'w') as fh:
        json.dump({'frames': len(frames), 'width': W, 'height': H, 'fps': args.fps, 'orig_w': orig_w, 'orig_h': orig_h,
                   'video_fps': video_fps, 'start': start, 'end': end, 'seconds': time.time() - t0}, fh)
    log(f'tamam ({time.time() - t0:.0f} sn)')


def get_mask(predictor, frames, th_step, kp2ds_all):
    """Resmî ProcessPipeline.get_mask kopyası (seçili kişinin noktalarıyla çalışır)."""
    frame_num = len(frames)
    num_step = 1 if frame_num < th_step else (frame_num + th_step) // th_step
    all_mask = []
    for index in range(num_step):
        each_frames = frames[index * th_step:(index + 1) * th_step]
        kp2ds = kp2ds_all[index * th_step:(index + 1) * th_step]
        if len(each_frames) > 4:
            key_frame_num = 4
        elif 4 >= len(each_frames) > 0:
            key_frame_num = 1
        else:
            continue
        key_frame_step = len(kp2ds) // key_frame_num
        key_frame_index_list = list(range(0, len(kp2ds), key_frame_step))
        key_points_index = [0, 1, 2, 5, 8, 11, 10, 13]
        key_frame_body_points_list = []
        for key_frame_index in key_frame_index_list:
            pts = []
            body = kp2ds[key_frame_index]['keypoints_body']
            for each_index in key_points_index:
                kp = body[each_index]
                if kp is None:
                    continue
                pts.append(kp)
            pts = np.array(pts)[:, :2]
            wh = np.array([[kp2ds[0]['width'], kp2ds[0]['height']]])
            key_frame_body_points_list.append((pts * wh).astype(np.int32))
        state = predictor.init_state_v2(frames=each_frames)
        predictor.reset_state(state)
        for ann_frame_idx, points in zip(key_frame_index_list, key_frame_body_points_list):
            labels = np.array([1] * points.shape[0], np.int32)
            predictor.add_new_points(inference_state=state, frame_idx=ann_frame_idx, obj_id=1, points=points, labels=labels)
        segs = {}
        for out_frame_idx, out_obj_ids, out_mask_logits in predictor.propagate_in_video(state):
            segs[out_frame_idx] = {oid: (out_mask_logits[i] > 0.0).cpu().numpy() for i, oid in enumerate(out_obj_ids)}
        for out_frame_idx in range(len(segs)):
            for _, m in segs[out_frame_idx].items():
                all_mask.append(m[0].astype(np.uint8))
    return all_mask


if __name__ == '__main__':
    main()
