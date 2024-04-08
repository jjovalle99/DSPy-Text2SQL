import os
import shlex
import subprocess
from pathlib import Path

from modal import Image, Mount, Secret, Stub, gpu, web_server

MODEL_DIR = "/model"
BASE_MODEL = "Nexusflow/Starling-LM-7B-beta"
GPU_CONFIG = gpu.A100(count=1)


# Download the model
def download_model_to_folder():
    from huggingface_hub import snapshot_download
    from transformers.utils import move_cache

    os.makedirs(MODEL_DIR, exist_ok=True)

    snapshot_download(
        BASE_MODEL,
        local_dir=MODEL_DIR,
        ignore_patterns=["*.pt", "*.gguf", "*.bin"],
    )
    move_cache()


# Image definition
image = (
    Image.from_registry("nvidia/cuda:12.1.1-devel-ubuntu22.04", add_python="3.11")
    .pip_install(
        "vllm==0.3.2",
        "huggingface_hub==0.19.4",
        "hf-transfer==0.1.4",
        "torch==2.1.2",
    )
    .env({"HF_HUB_ENABLE_HF_TRANSFER": "1"})
    .run_function(
        download_model_to_folder,
        timeout=60 * 20,
    )
)

stub = Stub("starling7b-ft", image=image)
mounts_map = {
    "chat_template": {
        "local_path": Path(__file__).parent / "chat_template.jinja",
        "remote_path": "/model/chat_template.jinja",
    },
}


@stub.function(
    gpu=GPU_CONFIG,
    mounts=[Mount.from_local_file(**mounts_map["chat_template"])],
    allow_concurrent_inputs=50,
    timeout=60 * 60,
    keep_warm=1,
)
@web_server(port=8000, startup_timeout=60)
def serve_model():
    base_model = shlex.quote(str(BASE_MODEL))
    cmd = (
        f"python -m vllm.entrypoints.openai.api_server --model {base_model} "
        "--chat-template /model/chat_template.jinja "
    )
    subprocess.Popen(cmd, shell=True)
