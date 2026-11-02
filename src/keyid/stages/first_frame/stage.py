from __future__ import annotations

import random

from keyid.core.contract import (
    BatchStage,
    ItemResult,
    StageContext,
    WorkItem,
    guard,
    require_upstream_output,
)
from keyid.core.errors import StageError
from keyid.core.seeding import resolve_seed

SUPPORTED_DTYPES = ("bfloat16", "float16", "float32")


class FirstFrameStage(BatchStage):
    type = "ernie.first_frame"
    needs = ("prompt_compose",)
    description = "generate items/<sample_id>/image.png with ERNIE-Image"


    def setup(self, context: StageContext) -> None:
        import torch
        from diffusers import ErnieImagePipeline

        self._torch = torch
        model_path = str(context.require_param("model_path"))
        dtype_name = str(context.param("torch_dtype", "bfloat16"))
        if dtype_name not in SUPPORTED_DTYPES:
            raise StageError(
                f"{self.type}: torch_dtype must be one of {SUPPORTED_DTYPES}, got {dtype_name!r}"
            )


        self.device = str(context.device or "cuda")
        self.width = int(context.param("width", 1264))
        self.height = int(context.param("height", 848))
        self.steps = int(context.param("steps", 8))
        self.guidance_scale = float(context.param("guidance_scale", 1.0))
        self.use_pe = bool(context.param("use_pe", False))

        requested_seed = context.param("seed")
        if requested_seed is None:
            self.run_seed = random.randint(0, 2**31 - 1)
            context.logger.info("no seed in YAML; drew run seed %d", self.run_seed)
        else:
            self.run_seed = int(requested_seed)

        context.logger.info(
            "loading ERNIE pipeline for first frames: %s (dtype=%s device=%s)",
            model_path,
            dtype_name,
            self.device,
        )
        pipeline = ErnieImagePipeline.from_pretrained(
            model_path,
            torch_dtype=getattr(torch, dtype_name),
        )
        self.pipe = pipeline.to(self.device)

    def process_batch(self, items: list[WorkItem], context: StageContext) -> list[ItemResult]:
        return [
            guard(item, lambda item=item: self._process_one(item, context), context.logger)
            for item in items
        ]

    def close(self, context: StageContext) -> None:
        if getattr(self, "pipe", None) is not None:
            del self.pipe
        torch = getattr(self, "_torch", None)
        if torch is not None and torch.cuda.is_available():
            torch.cuda.empty_cache()
        context.logger.info("first_frame closed")


    def _process_one(self, item: WorkItem, context: StageContext) -> ItemResult:
        prompt = self._prompt(item)
        seed = self._seed_for(item.sample_id)

        generator = self._torch.Generator(device=self.device).manual_seed(seed)
        output = self.pipe(
            prompt=prompt,
            height=self.height,
            width=self.width,
            num_inference_steps=self.steps,
            guidance_scale=self.guidance_scale,
            generator=generator,
            use_pe=self.use_pe,
        )

        image_path = item.item_dir / "image.png"
        output.images[0].save(image_path)

        return ItemResult.succeeded(
            item.sample_id,
            image=context.relative(image_path),
            seed=seed,
            prompt=prompt,
            steps=self.steps,
            guidance_scale=self.guidance_scale,
            height=self.height,
            width=self.width,
        )

    @staticmethod
    def _prompt(item: WorkItem) -> str:
        return str(
            require_upstream_output(
                item, "prompt_compose", "prompt", what="the composed generation prompt"
            )
        ).strip()

    def _seed_for(self, sample_id: str) -> int:

        return resolve_seed(self.run_seed, stage=self.type, sample_id=sample_id)
