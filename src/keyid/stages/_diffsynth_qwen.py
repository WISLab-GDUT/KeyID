from __future__ import annotations

import glob
from pathlib import Path
from typing import Any

from keyid.core.errors import StageError


def match_paths(pattern: str):

    paths = sorted(glob.glob(pattern))
    if not paths:
        raise FileNotFoundError(f"no files matched: {pattern}")
    return paths[0] if len(paths) == 1 else paths


def open_rgb(path: Path) -> Any:

    from PIL import Image

    image = Image.open(path)
    image.load()
    return image.convert("RGB")


def save_image_atomic(image: Any, output_path: Path) -> None:

    output_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = output_path.with_name(f".{output_path.stem}.tmp{output_path.suffix}")
    image.save(tmp_path)
    tmp_path.replace(output_path)


def build_qwen_image_pipeline(
    *,
    edit_root: Path,
    match_paths,
    torch,
    torch_device: str = "cuda",
    torch_dtype=None,
    vram_limit: int | None = None,
):

    from diffsynth.pipelines.qwen_image import ModelConfig, QwenImagePipeline

    torch_dtype = torch_dtype or torch.bfloat16
    vram_config = {
        "offload_dtype": torch_dtype,
        "offload_device": "cpu",
        "onload_dtype": torch_dtype,
        "onload_device": "cpu",
        "preparing_dtype": torch_dtype,
        "preparing_device": torch_device,
        "computation_dtype": torch_dtype,
        "computation_device": torch_device,
    }

    return QwenImagePipeline.from_pretrained(
        torch_dtype=torch_dtype,
        device=torch_device,
        model_configs=[
            ModelConfig(
                path=match_paths(
                    str(edit_root / "transformer" / "diffusion_pytorch_model*.safetensors")
                ),
                **vram_config,
            ),
            ModelConfig(
                path=match_paths(str(edit_root / "text_encoder" / "model*.safetensors")),
                **vram_config,
            ),
            ModelConfig(
                path=match_paths(str(edit_root / "vae" / "diffusion_pytorch_model.safetensors")),
                **vram_config,
            ),
        ],
        processor_config=ModelConfig(path=str(edit_root / "processor")),
        vram_limit=vram_limit,
    )


def load_lora(pipe: Any, lora_path: Path, *, alpha: float = 1.0) -> None:

    from diffsynth.pipelines.qwen_image import FlowMatchScheduler, ModelConfig

    lora = ModelConfig(
        path=str(lora_path),
        model_id="lightx2v/Qwen-Image-Edit-2511-Lightning",
        origin_file_pattern=Path(lora_path).name,
    )
    pipe.load_lora(pipe.dit, lora, alpha=alpha)
    pipe.scheduler = FlowMatchScheduler("Qwen-Image-Lightning")


def require_diffsynth() -> None:

    try:
        import diffsynth
    except ImportError as exc:
        raise StageError(
            "the 'diffsynth' package is not installed in this stage's interpreter; "
            "install it or point stages.<id>.python at an environment that has it"
        ) from exc
