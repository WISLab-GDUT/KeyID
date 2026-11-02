from __future__ import annotations

import gc
import sys
from pathlib import Path

from keyid.core.contract import (
    BatchStage,
    ItemResult,
    StageContext,
    WorkItem,
    guard,
    require_upstream_output,
)
from keyid.core.errors import StageError

DEFAULT_NEGATIVE_PROMPT = "pc game, console game, video game, cartoon, childish, ugly"


class InterpolationStage(BatchStage):
    type = "ltx.interpolation"
    requires_human_reference = True
    needs = ("keyframe_select",)
    description = "render items/<sample_id>/video.mp4 with LTX-2 keyframe interpolation"


    def setup(self, context: StageContext) -> None:
        import torch


        self.upstream = context.require_single_upstream()
        context.logger.info("reading keyframe plan and prompt from %r", self.upstream)

        self._torch = torch
        self._prepare_imports(context)

        from ltx_core.components.guiders import MultiModalGuiderParams
        from ltx_core.loader import LTXV_LORA_COMFY_RENAMING_MAP, LoraPathStrengthAndSDOps
        from ltx_core.model.video_vae import TilingConfig
        from ltx_pipelines.keyframe_interpolation import KeyframeInterpolationPipeline
        from ltx_pipelines.utils.constants import detect_params
        from ltx_pipelines.utils.media_io import encode_video

        self._MultiModalGuiderParams = MultiModalGuiderParams
        self._LoraPathStrengthAndSDOps = LoraPathStrengthAndSDOps
        self._renaming_map = LTXV_LORA_COMFY_RENAMING_MAP
        self._encode_video = encode_video
        self._tiling_config = TilingConfig.default()

        checkpoint_path = Path(context.require_param("checkpoint_path"))
        if not checkpoint_path.exists():
            raise StageError(f"{self.type}: checkpoint not found: {checkpoint_path}")
        distilled_lora_path = Path(context.require_param("distilled_lora_path"))
        if not distilled_lora_path.exists():
            raise StageError(f"{self.type}: distilled LoRA not found: {distilled_lora_path}")
        spatial_upsampler_path = Path(context.require_param("spatial_upsampler_path"))
        if not spatial_upsampler_path.exists():
            raise StageError(f"{self.type}: spatial upsampler not found: {spatial_upsampler_path}")
        gemma_root = Path(context.require_param("gemma_root"))
        if not gemma_root.exists():
            raise StageError(f"{self.type}: gemma_root not found: {gemma_root}")


        detected = detect_params(str(checkpoint_path))
        self.params_detected = detected
        self.height = self._resolve(context.param("height"), detected.stage_2_height)
        self.width = self._resolve(context.param("width"), detected.stage_2_width)
        self.frame_rate = self._resolve(context.param("frame_rate"), detected.frame_rate)
        self.num_inference_steps = self._resolve(
            context.param("num_inference_steps"), detected.num_inference_steps
        )

        self.distilled_lora_strength = float(context.require_param("distilled_lora_strength"))
        self.negative_prompt = str(context.param("negative_prompt", DEFAULT_NEGATIVE_PROMPT))
        self.seed = int(context.param("seed", 42))
        self.image_conditioning_strength = float(context.param("image_conditioning_strength", 1.0))
        self.image_conditioning_crf = int(context.param("image_conditioning_crf", 1))
        self.max_batch_size = int(context.param("max_batch_size", 1))
        self.streaming_prefetch_count = context.param("streaming_prefetch_count")
        self.torch_compile = bool(context.param("compile", False))

        context.logger.info(
            "building LTX pipeline: %dx%d fps=%s steps=%s",
            self.width,
            self.height,
            self.frame_rate,
            self.num_inference_steps,
        )
        self.pipeline = KeyframeInterpolationPipeline(
            checkpoint_path=str(checkpoint_path),
            distilled_lora=[
                self._LoraPathStrengthAndSDOps(
                    str(distilled_lora_path), self.distilled_lora_strength, self._renaming_map
                )
            ],
            spatial_upsampler_path=str(spatial_upsampler_path),
            gemma_root=str(gemma_root),
            loras=(),
            quantization=None,
            torch_compile=self.torch_compile,
        )

    def process_batch(self, items: list[WorkItem], context: StageContext) -> list[ItemResult]:
        return [
            guard(item, lambda item=item: self._process_one(item, context), context.logger)
            for item in items
        ]

    def close(self, context: StageContext) -> None:
        if getattr(self, "pipeline", None) is not None:
            del self.pipeline
        self._free_memory()
        context.logger.info("interpolation closed")


    def _prompt(self, item: WorkItem) -> str:

        return str(
            require_upstream_output(
                item, self.upstream, "prompt", what="the composed generation prompt"
            )
        ).strip()

    @staticmethod
    def _torch_inference_mode():
        import torch

        return torch.inference_mode()

    def _process_one(self, item: WorkItem, context: StageContext) -> ItemResult:
        reason = self.skip_reason(item.raw)
        if reason is not None:
            return ItemResult.skipped(item.sample_id, reason)
        from ltx_pipelines.utils.args import ImageConditioningInput

        plan = item.input(self.upstream).get("keyframe_plan")
        if not plan:
            raise ValueError(f"{self.upstream} produced no keyframe_plan")

        images = []
        for keyframe in sorted(plan, key=lambda entry: int(entry["target_idx"])):


            image_path = context.input_path(self.upstream, str(keyframe["image"]))
            if not image_path.exists():
                raise FileNotFoundError(f"keyframe image is missing: {image_path}")
            images.append(
                ImageConditioningInput(
                    path=str(image_path),
                    frame_idx=int(keyframe["target_idx"]),
                    strength=self.image_conditioning_strength,
                    crf=self.image_conditioning_crf,
                )
            )

        num_frames = int(item.input(self.upstream).get("num_frames") or 0)
        if num_frames <= 0:
            raise ValueError(f"{self.upstream} did not report num_frames")

        from ltx_core.model.video_vae import get_video_chunks_number

        output_path = item.item_dir / "video.mp4"
        output_path.parent.mkdir(parents=True, exist_ok=True)
        video_chunks_number = get_video_chunks_number(num_frames, self._tiling_config)


        prompt = self._prompt(item)

        try:
            with self._torch_inference_mode():
                video, audio = self.pipeline(
                    prompt=prompt,
                    negative_prompt=self.negative_prompt,
                    seed=self.seed,
                    height=self.height,
                    width=self.width,
                    num_frames=num_frames,
                    frame_rate=self.frame_rate,
                    num_inference_steps=self.num_inference_steps,
                    video_guider_params=self._guider_params(self.params_detected.video_guider_params),
                    audio_guider_params=self._guider_params(self.params_detected.audio_guider_params),
                    images=images,
                    tiling_config=self._tiling_config,
                    streaming_prefetch_count=self.streaming_prefetch_count,
                    max_batch_size=self.max_batch_size,
                )
                self._encode_video(
                    video=video,
                    fps=self.frame_rate,
                    audio=audio,
                    output_path=str(output_path),
                    video_chunks_number=video_chunks_number,
                )
            del video, audio
        finally:
            self._free_memory()

        if not output_path.exists():
            raise RuntimeError(f"LTX reported success but wrote no video at {output_path}")

        return ItemResult.succeeded(
            item.sample_id,
            video=context.relative(output_path),
            seed=self.seed,
            num_frames=num_frames,
            height=self.height,
            width=self.width,
            frame_rate=self.frame_rate,
            num_inference_steps=self.num_inference_steps,


            keyframe_names=[Path(str(entry["image"])).name for entry in plan],
        )


    def _guider_params(self, source):
        return self._MultiModalGuiderParams(
            cfg_scale=source.cfg_scale,
            stg_scale=source.stg_scale,
            rescale_scale=source.rescale_scale,
            modality_scale=source.modality_scale,
            skip_step=source.skip_step,
            stg_blocks=source.stg_blocks,
        )

    @staticmethod
    def _resolve(configured, detected):
        return configured if configured is not None else detected

    def _free_memory(self) -> None:
        gc.collect()
        torch = getattr(self, "_torch", None)
        if torch is not None and torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()

    def _prepare_imports(self, context: StageContext) -> None:

        root = Path(context.require_param("ltx_path")).resolve()
        sources = [root / "packages" / "ltx-core" / "src", root / "packages" / "ltx-pipelines" / "src"]
        for source in sources:
            if not source.is_dir():
                raise StageError(f"{self.type}: LTX package source missing: {source}")
            if str(source) not in sys.path:
                sys.path.insert(0, str(source))
        context.logger.info("LTX packages imported from %s", root)
