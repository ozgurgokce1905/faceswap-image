# FaceSwap RunPod imajı: FaceFusion + Python paketleri + modeller hazır gömülü.
# Pod bu imajla açılınca kurulum yok; sunucu ~1-2 dk içinde HAZIR olur.
FROM nvidia/cuda:12.8.1-cudnn-runtime-ubuntu24.04

ENV DEBIAN_FRONTEND=noninteractive PYTHONUNBUFFERED=1 PYTHONIOENCODING=utf-8 PIP_NO_CACHE_DIR=1

RUN apt-get update && apt-get install -y --no-install-recommends \
        python3 python3-venv python3-pip git curl ca-certificates xz-utils \
        libgl1 libglib2.0-0 libgomp1 procps \
    && rm -rf /var/lib/apt/lists/*

# ffmpeg (statik)
RUN mkdir -p /opt/bin && cd /tmp \
    && curl -fsSL -o ff.tar.xz https://johnvansickle.com/ffmpeg/releases/ffmpeg-release-amd64-static.tar.xz \
    && tar xf ff.tar.xz && cp ffmpeg-*-static/ffmpeg ffmpeg-*-static/ffprobe /opt/bin/ && rm -rf /tmp/ff*
ENV PATH=/opt/bin:/opt/venv/bin:$PATH

# FaceFusion (sabit sürüm)
ARG FF_COMMIT=decc6c9
RUN git clone -q https://github.com/facefusion/facefusion.git /opt/facefusion \
    && git -C /opt/facefusion checkout -q ${FF_COMMIT}

# Python ortamı
RUN python3 -m venv /opt/venv && /opt/venv/bin/pip install --upgrade pip \
    && grep -v '^onnxruntime' /opt/facefusion/requirements.txt > /tmp/req.txt \
    && /opt/venv/bin/pip install -r /tmp/req.txt "onnxruntime-gpu==1.24.4" \
       nvidia-cuda-runtime-cu12 nvidia-cudnn-cu12 nvidia-cublas-cu12 nvidia-cufft-cu12 \
       nvidia-curand-cu12 nvidia-cuda-nvrtc-cu12 nvidia-nvjitlink-cu12 \
       fastapi uvicorn python-multipart requests \
    && /opt/venv/bin/python -c "import onnxruntime, cv2, numpy, gradio, fastapi; print('paketler OK')"

# Sunucu kodu
ARG VERSION=dev
COPY sunucu /opt/fs/sunucu
COPY runpod /opt/fs/runpod
RUN echo "${VERSION}" > /opt/fs/VERSION && find /opt/fs -name __pycache__ -type d -exec rm -rf {} + ; true
RUN cp /opt/fs/sunucu/sitecustomize.py "$(/opt/venv/bin/python -c 'import site;print(site.getsitepackages()[0])')/sitecustomize.py"

# Modeller (ön ayarlardaki tüm modeller; ~2.5 GB) — derleme sırasında indirilir
RUN FACESWAP_HOME=/opt/fs FACESWAP_FF_DIR=/opt/facefusion FACESWAP_PROVIDER=cpu \
    /opt/venv/bin/python /opt/fs/sunucu/warmup.py \
    && du -sh /opt/facefusion/.assets

# Tüm vücut (Wan2.2 Animate) kodu + Python ortamı (torch cu128, flash-attn, SAM2) — modeller hariç (pod diskine iner)
RUN apt-get update && apt-get install -y --no-install-recommends util-linux build-essential python3-dev && rm -rf /var/lib/apt/lists/* \
    && WAN_HOME=/opt/wan FACESWAP_HOME=/opt/fs SKIP_MODELS=1 bash /opt/fs/runpod/wan_setup.sh \
    && rm -rf /opt/fs/data /opt/fs/logs /root/.cache/pip \
    && du -sh /opt/wan

ENV FACESWAP_FF_DIR=/opt/facefusion FACESWAP_PROVIDER=cuda FACESWAP_HOME=/workspace/fs FACESWAP_THREADS=8 \
    WAN_HOME=/opt/wan WAN_CKPT=/workspace/wan/Wan2.2-Animate-14B
EXPOSE 8765
CMD ["bash", "/opt/fs/runpod/image_start.sh"]
