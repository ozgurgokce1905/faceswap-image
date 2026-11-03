"""
warmup.py - Ön ayarlarda kullanılan tüm modelleri önceden indirir (kurulumda bir kez çalışır).
Böylece ilk video işlenirken indirme beklenmez. İndirme ilerlemesi ekranda görünür.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import ff_bridge  # noqa: E402
import presets  # noqa: E402

from facefusion import state_manager  # noqa: E402
from facefusion.processors.core import get_processors_modules  # noqa: E402


def main() -> int:
    provider = os.environ.get('FACESWAP_PROVIDER', 'cuda')
    print('[1/2] Ortak modeller (dedektör, landmark, tanıyıcı, içerik filtresi)...')
    ff_bridge.init(provider, 'info')

    wanted = {}
    for preset in presets.PRESETS.values():
        args = preset['args']
        for i, a in enumerate(args):
            if a in ('--face-swapper-model', '--face-enhancer-model', '--expression-restorer-model'):
                wanted.setdefault(a, set()).add(args[i + 1])

    print('[2/2] İşlemci modelleri...')
    for arg, models in wanted.items():
        module_name = arg[2:-6].replace('-', '_')  # --face-swapper-model -> face_swapper
        state_key = module_name + '_model'
        module = get_processors_modules([module_name])[0]
        for model in sorted(models):
            state_manager.init_item(state_key, model)
            print('   -', module_name, model)
            if not module.pre_check():
                print('HATA: model indirilemedi:', module_name, model)
                return 1
    print('Tüm modeller hazır.')
    return 0


if __name__ == '__main__':
    sys.exit(main())
