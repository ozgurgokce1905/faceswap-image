"""
presets.py - Kalite ön ayarları. Her ön ayar FaceFusion headless-run argümanları üretir.

Gerçekçilik için kullanılan parçalar:
  * face_swapper  hyperswap_1a_256 + pixel boost (512/768): yüzün kendisi
  * face_enhancer gfpgan_1.4 / codeformer, düşük blend: cilt dokusunu toparlar ama "plastik" yapmaz
  * expression_restorer (live_portrait): orijinal mimikleri geri getirir -> en çok gerçekçilik katan adım
  * face mask box + occlusion: el, saç, gözlük gibi yüzün önündeki nesneleri korur
  * face_mask_blur 0.3-0.4: kenarları yumuşak geçirir
"""

# DİKKAT: ff_bridge.DETECTOR_ARGS ile aynı değerler (seçilen kişinin sırası kaymasın diye)
DETECTOR_ARGS = [
    '--face-detector-model', 'yolo_face',
    '--face-detector-size', '640x640',
    '--face-detector-score', '0.5',
    '--face-landmarker-model', '2dfan4',
    '--face-selector-order', 'left-right',
]

PRESETS = {
    # Hızlı önizleme: sadece yüz değiştirme, 512 boost
    'hizli': {
        'label': 'Hızlı',
        'processors': ['face_swapper'],
        'args': [
            '--face-swapper-model', 'hyperswap_1a_256',
            '--face-swapper-pixel-boost', '256x256',
            '--face-mask-types', 'box', 'occlusion',
            '--face-mask-blur', '0.3',
            '--output-video-quality', '85',
            '--output-video-preset', 'veryfast',
        ],
    },
    # Dengeli: değiştir + hafif netleştirme
    'kaliteli': {
        'label': 'Kaliteli',
        'processors': ['face_swapper', 'face_enhancer'],
        'args': [
            '--face-swapper-model', 'hyperswap_1a_256',
            '--face-swapper-pixel-boost', '512x512',
            '--face-enhancer-model', 'gfpgan_1.4',
            '--face-enhancer-blend', '50',
            '--face-mask-types', 'box', 'occlusion',
            '--face-mask-blur', '0.35',
            '--output-video-quality', '90',
            '--output-video-preset', 'fast',
        ],
    },
    # En gerçekçi: değiştir + mimikleri geri getir + hafif netleştirme, 768 boost
    'ultra': {
        'label': 'Ultra (en gerçekçi)',
        'processors': ['face_swapper', 'expression_restorer', 'face_enhancer'],
        'args': [
            '--face-swapper-model', 'hyperswap_1a_256',
            '--face-swapper-pixel-boost', '768x768',
            '--expression-restorer-model', 'live_portrait',
            '--expression-restorer-factor', '80',
            '--face-enhancer-model', 'codeformer',
            '--face-enhancer-blend', '40',
            '--face-mask-types', 'box', 'occlusion',
            '--face-mask-blur', '0.4',
            '--output-video-quality', '95',
            '--output-video-preset', 'medium',
        ],
    },
}

DEFAULT_PRESET = 'kaliteli'


def build_args(preset_name: str, source_paths, target_path: str, output_path: str,
               frame_number: int, face_position: int, temp_path: str, jobs_path: str,
               execution_provider: str = 'cuda', thread_count: int = 4,
               trim_start=None, trim_end=None) -> list:
    preset = PRESETS.get(preset_name) or PRESETS[DEFAULT_PRESET]
    args = ['headless-run']
    args += ['-s'] + list(source_paths)
    args += ['-t', target_path, '-o', output_path]
    args += ['--temp-path', temp_path, '--jobs-path', jobs_path]
    args += DETECTOR_ARGS
    args += [
        '--face-selector-mode', 'reference',
        '--reference-frame-number', str(frame_number),
        '--reference-face-position', str(face_position),
        '--reference-face-distance', '0.45',  # 0.6 çok gevşekti: videodaki HERKESİ eşleştirip değiştiriyordu; 0.3 ise kişiyi kaçırabiliyor
    ]
    if trim_start is not None:
        args += ['--trim-frame-start', str(int(trim_start))]
    if trim_end is not None:
        args += ['--trim-frame-end', str(int(trim_end))]
    args += ['--processors'] + preset['processors']
    args += preset['args']
    args += [
        '--execution-providers', execution_provider,
        '--execution-thread-count', str(thread_count),
        '--video-memory-strategy', 'moderate',
        '--log-level', 'info',
    ]
    return args
