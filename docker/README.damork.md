# Damork vLLM Docker Image

This image starts from the official `vllm/vllm-openai:latest` runtime and applies the small Damork alias patch on top. That keeps the CUDA/PyTorch stack aligned with the vLLM image that already works on RunPod, while adding support for Damork-branded config and processor names.

## Build

```bash
docker build -f docker/Dockerfile.damork -t damork-vllm:0.1 .
```

The default base image is:

```text
vllm/vllm-openai:latest
```

To pin to a specific official vLLM image:

```bash
docker build \
  -f docker/Dockerfile.damork \
  --build-arg BASE_IMAGE=vllm/vllm-openai:<tag> \
  -t damork-vllm:0.1 .
```

## Run

```bash
docker run --gpus all --ipc=host --shm-size=8g \
  -p 8000:8000 \
  damork-vllm:0.1
```

Default serve settings:

```text
model: PWLabs/Qwen3.5-0.8B
served model name: damork
host: 0.0.0.0
port: 8000
max model len: 2048
gpu memory utilization: 0.85
```

For a private Hugging Face model repo, pass a token:

```bash
docker run --gpus all --ipc=host --shm-size=8g \
  -p 8000:8000 \
  -e HF_TOKEN=hf_xxx \
  damork-vllm:0.1
```

## Override The Model Or Serve Args

Use environment variables for the common defaults:

```bash
docker run --gpus all --ipc=host --shm-size=8g \
  -p 8000:8000 \
  -e DAMORK_MODEL=PWLabs/Damork-Coder-27B-multimodal-v2 \
  -e DAMORK_MAX_MODEL_LEN=4096 \
  -e DAMORK_GPU_MEMORY_UTILIZATION=0.90 \
  damork-vllm:0.1
```

Or pass full `vllm serve` arguments after the image name:

```bash
docker run --gpus all --ipc=host --shm-size=8g \
  -p 8000:8000 \
  damork-vllm:0.1 \
  PWLabs/Damork-Coder-27B-multimodal-v2 \
  --host 0.0.0.0 \
  --port 8000 \
  --served-model-name damork \
  --max-model-len 4096 \
  --gpu-memory-utilization 0.90 \
  --enforce-eager
```

If arguments are passed, the entrypoint preserves them and fills in missing defaults such as `--served-model-name damork`, `--host 0.0.0.0`, and `--enforce-eager`.

## Test

```bash
curl http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model":"damork","messages":[{"role":"user","content":"Hello"}],"max_tokens":64}'
```

Image request:

```bash
curl http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "damork",
    "messages": [
      {
        "role": "user",
        "content": [
          {"type": "text", "text": "Describe this image in detail."},
          {
            "type": "image_url",
            "image_url": {
              "url": "https://fastly.picsum.photos/id/410/200/300.jpg"
            }
          }
        ]
      }
    ],
    "max_tokens": 128
  }'
```

## Push To A Registry

Example for GitHub Container Registry:

```bash
docker tag damork-vllm:0.1 ghcr.io/fares-boutriga/damork-vllm:0.1
docker push ghcr.io/fares-boutriga/damork-vllm:0.1
```

`VLLM_USE_FLASHINFER_SAMPLER=0` is set by default because this was the stable configuration during RunPod testing. Remove or override it only after validating FlashInfer in the target image.

Avoid rebuilding vLLM from source inside this image unless you also control the target NVIDIA driver version. The previous `cu130` build failed on RunPod hosts reporting CUDA driver capability `12.4`.
