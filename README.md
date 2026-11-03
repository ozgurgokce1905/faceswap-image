# faceswap-image

FaceSwap sunucusunun RunPod imajı. Her `main` push'unda GitHub Actions imajı derler ve
`ghcr.io/ozgurgokce1905/faceswap:latest` olarak yayınlar. Pod bu imajla açılınca kurulum yapılmaz (~1-2 dk HAZIR).

- `sunucu/`  : FastAPI sunucu + FaceFusion köprüsü + tüm vücut modu (Wan2.2 Animate)
- `runpod/`  : `image_start.sh` (imaj başlangıcı), `wan_setup.sh` (tüm vücut kurulumu, pod diskine)
- `Dockerfile`: FaceFusion + paketler + modeller gömülü

Kaynak: D:\Dev\faceswap (Ozgur Gokce). created by Ozgur Gokce
