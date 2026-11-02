from __future__ import annotations

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

from .assets import stage_assets
from .facefusion_adapter import (
    DEFAULT_VIDEO_MEMORY_STRATEGY,
    FaceSwapPipeline,
    reject_strategy_in_extra_args,
    resolve_execution_providers,
    validate_video_memory_strategy,
)


ASSETS_RELATIVE = Path(".assets") / "models"


class FaceSwapStage(BatchStage):
    type = "facefusion.video_swap"
    requires_human_reference = True
    needs = ("draft_video",)
    description = "swap the reference face into items/<sample_id>/video.mp4 (FaceFusion)"


    def setup(self, context: StageContext) -> None:


        self.processors = [str(processor) for processor in context.require_param("processors")]
        self.extra_args = [str(item) for item in (context.param("extra_args") or [])]
        strategy = validate_video_memory_strategy(
            context.param("video_memory_strategy", DEFAULT_VIDEO_MEMORY_STRATEGY)
        )
        reject_strategy_in_extra_args(self.extra_args)

        self.facefusion_root = self._prepare_imports(context)
        self._stage_assets(context)

        providers = resolve_execution_providers(
            [str(provider) for provider in context.require_param("execution_providers")],
            allow_cpu_fallback=bool(context.param("allow_cpu_fallback", False)),
        )


        self.source_index = int(context.param("source_image_index", 0))
        if self.source_index < 0:
            raise StageError("source_image_index must be a non-negative human-reference ordinal")

        context.logger.info("facefusion execution providers: %s", providers)
        context.logger.info("facefusion processors: %s", self.processors)
        context.logger.info("facefusion video_memory_strategy: %s", strategy)

        self.pipeline = FaceSwapPipeline(
            processors=self.processors,
            execution_providers=providers,
            extra_args=self.extra_args,
            video_memory_strategy=strategy,
            facefusion_root=self.facefusion_root,
            work_dir=context.stage_dir,
        )

    def process_batch(self, items: list[WorkItem], context: StageContext) -> list[ItemResult]:
        if not items:
            return []

        if any(self.skip_reason(item.raw) is None for item in items) and not self.pipeline.environment_ready:


            self.pipeline.prepare_environment()
            context.logger.info("facefusion environment ready; replaying across %d item(s)", len(items))

        return [
            guard(item, lambda item=item: self._process_one(item, context), context.logger)
            for item in items
        ]

    def close(self, context: StageContext) -> None:
        pipeline = getattr(self, "pipeline", None)
        if pipeline is not None:
            try:
                pipeline.close()
            except Exception:
                context.logger.debug("facefusion teardown failed", exc_info=True)
        context.logger.info("face_swap closed")


    @staticmethod
    def _prompt(item: WorkItem) -> str:

        return str(
            require_upstream_output(
                item, "draft_video", "prompt", what="the composed generation prompt"
            )
        ).strip()

    def _process_one(self, item: WorkItem, context: StageContext) -> ItemResult:


        reason = self.skip_reason(item.raw)
        if reason is not None:
            return ItemResult.skipped(item.sample_id, reason)
        source_image = self._source_image(item, context)
        target_video = context.input_media(item, "draft_video", "video")
        output_path = item.item_dir / "video.mp4"
        prompt = self._prompt(item)

        self.pipeline.run(
            source_path=str(source_image),
            target_path=str(target_video),
            output_path=str(output_path),
        )

        return ItemResult.succeeded(
            item.sample_id,
            video=context.relative(output_path),
            source_image=source_image.name,
            source_image_index=self._source_indices(item.raw)[self.source_index],
            human_reference_index=self.source_index,
            processors=list(self.processors),
            prompt=prompt,
        )

    def _source_image(self, item: WorkItem, context: StageContext) -> Path:

        return context.resolve_human_image(item, self.source_index)

    @staticmethod
    def _source_indices(raw) -> list[int]:
        from keyid.core.references import human_reference_indices

        return human_reference_indices(raw.get("img_paths") or ())


    def _prepare_imports(self, context: StageContext) -> Path:

        candidate = Path(context.require_param("facefusion_path")).resolve()
        if not (candidate / "facefusion" / "__init__.py").is_file():
            raise StageError(
                f"{self.type}: no facefusion package at {candidate}. Point "
                "params.facefusion_path at your external FaceFusion checkout root."
            )
        if str(candidate) not in sys.path:
            sys.path.insert(0, str(candidate))
        return candidate

    def _stage_assets(self, context: StageContext) -> None:
        assets_dir = Path(
            context.param("assets_dir")
            or (self.facefusion_root / ASSETS_RELATIVE)
        )
        target = self.facefusion_root / ASSETS_RELATIVE
        if assets_dir.resolve() == target.resolve():
            context.logger.info("facefusion assets already in place: %s", target)
            return
        if not assets_dir.is_dir():
            raise StageError(
                f"{self.type}: assets_dir does not exist: {assets_dir}. Point it at the "
                "directory holding FaceFusion's .onnx models (kept outside the repository)."
            )
        report = stage_assets(assets_dir, target)
        context.logger.info("facefusion assets staged into %s (%s)", target, report.describe())
