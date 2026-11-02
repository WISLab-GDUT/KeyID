from __future__ import annotations

from typing import Any

from keyid.core.contract import BatchStage, ItemResult, StageContext, WorkItem, guard
from keyid.core.errors import StageError
from keyid.core.seeding import resolve_seed

from .._dataset_prompt import resolve_original_prompt, validate_prompt_fields


PE_METHOD_NAME = "_enhance_prompt_with_pe"

SUPPORTED_DTYPES = ("bfloat16", "float16", "float32")


def normalize_pe_prompt(value: Any) -> str:

    if isinstance(value, (list, tuple)):
        value = value[0] if value else ""
    return str(value).strip().strip('"')


class PromptPeStage(BatchStage):
    type = "ernie.prompt_pe"
    needs = ()
    description = "ERNIE-Image prompt enhancer: produces pe_prompt for each sample"


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
        self.temperature = float(context.param("temperature", 0.6))
        self.top_p = float(context.param("top_p", 0.95))
        self.seed = int(context.param("seed", 0))
        self.seed_mode = str(context.param("seed_mode", "per_item"))


        self.prompt_fields = validate_prompt_fields(
            context.param("prompt_fields", ["prompt"]), where=f"{self.type}.prompt_fields"
        )
        if self.seed_mode not in ("fixed", "per_item"):
            raise StageError(f"{self.type}: seed_mode must be 'fixed' or 'per_item'")

        context.logger.info(
            "loading ERNIE pipeline for prompt enhancement: %s (dtype=%s device=%s)",
            model_path,
            dtype_name,
            self.device,
        )
        pipeline = ErnieImagePipeline.from_pretrained(
            model_path,
            torch_dtype=getattr(torch, dtype_name),
        )
        self.pipe = pipeline.to(self.device)


        if not callable(getattr(self.pipe, PE_METHOD_NAME, None)):
            raise StageError(
                f"{self.type}: {type(self.pipe).__name__}.{PE_METHOD_NAME} is missing. "
                "This stage depends on a private diffusers API; re-check the diffusers "
                "version pinned by the stage interpreter and adjust this adapter."
            )

    def process_batch(self, items: list[WorkItem], context: StageContext) -> list[ItemResult]:
        return [guard(item, lambda item=item: self._process_one(item, context), context.logger)
                for item in items]

    def close(self, context: StageContext) -> None:
        pipe = getattr(self, "pipe", None)
        if pipe is not None:
            del self.pipe
        torch = getattr(self, "_torch", None)
        if torch is not None and torch.cuda.is_available():
            torch.cuda.empty_cache()
        context.logger.info("prompt_pe closed")


    def _process_one(self, item: WorkItem, context: StageContext) -> ItemResult:
        prompt, prompt_field = resolve_original_prompt(
            dict(item.raw.get("fields") or {}), self.prompt_fields, sample_id=item.sample_id
        )

        seed = self._seed_for(item.sample_id)
        self._seed_everything(seed)

        raw_pe = getattr(self.pipe, PE_METHOD_NAME)(
            prompt,
            device=self.device,
            width=self.width,
            height=self.height,
            temperature=self.temperature,
            top_p=self.top_p,
        )
        pe_prompt = normalize_pe_prompt(raw_pe)
        if not pe_prompt:
            raise ValueError("prompt enhancer returned an empty prompt")

        return ItemResult.succeeded(
            item.sample_id,
            pe_prompt=pe_prompt,
            prompt=prompt,
            prompt_field=prompt_field,
            seed=seed,
            seed_mode=self.seed_mode,
            temperature=self.temperature,
            top_p=self.top_p,
        )

    def _seed_for(self, sample_id: str) -> int:


        return resolve_seed(self.seed, stage=self.type, sample_id=sample_id, mode=self.seed_mode)

    def _seed_everything(self, seed: int) -> None:

        torch = self._torch
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
