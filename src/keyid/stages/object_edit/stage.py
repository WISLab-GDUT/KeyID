from __future__ import annotations

from pathlib import Path

from keyid.core.contract import BatchStage, ItemResult, StageContext, WorkItem, guard
from keyid.core.errors import StageError

from .edit_plan import build_steps

PLAN_FILE = "edit_steps.json"


class ObjectEditStage(BatchStage):
    type = "qwen.object_edit"
    needs = ("first_frame",)
    description = "edit non-human subjects of the first frame with Qwen-Image-Edit (Track2)"


    def setup(self, context: StageContext) -> None:

        self.edit_heads = bool(context.param("edit_heads", False))
        self.reference_root = context.param("reference_root")
        self.seed = int(context.param("seed", 42))
        self.num_inference_steps = int(context.param("num_inference_steps", 4))
        self.cfg_scale = float(context.param("cfg_scale", 1.0))
        self.lora_strength = float(context.param("lora_strength", 1.0))
        self.save_plan = bool(context.param("save_plan", True))
        self.source_index = int(context.param("source_image_index", 0))

        if self.reference_root is None:
            raise StageError(
                f"{self.type}: reference_root is required; it is the directory that "
                "img_paths are relative to (the task's reference images)."
            )
        self.reference_root = Path(self.reference_root)

        self._models_loaded = False
        self._pipe = None
        self._edit_root = None
        self._lora_path = None
        self._torch = None

        context.logger.info(
            "object_edit ready: edit_heads=%s steps=%d cfg=%s seed=%d",
            self.edit_heads,
            self.num_inference_steps,
            self.cfg_scale,
            self.seed,
        )

    def process_batch(self, items: list[WorkItem], context: StageContext) -> list[ItemResult]:
        return [
            guard(item, lambda item=item: self._process_one(item, context), context.logger)
            for item in items
        ]

    def close(self, context: StageContext) -> None:
        if self._pipe is not None:
            del self._pipe
            self._pipe = None
        if self._torch is not None and self._torch.cuda.is_available():
            self._torch.cuda.empty_cache()
        context.logger.info("object_edit closed")


    def _process_one(self, item: WorkItem, context: StageContext) -> ItemResult:
        plan = build_steps(
            item.sample_id,
            dict(item.raw.get("fields") or {}),
            edit_heads=self.edit_heads,
            reference_root=self.reference_root,
            task_dir=item.item_dir,
        )

        plan_path = item.item_dir / PLAN_FILE
        if self.save_plan:
            from keyid.core.artifacts import atomic_write_json

            atomic_write_json(plan_path, plan)

        base_frame = context.input_media(item, "first_frame", "image")
        output_path = item.item_dir / plan["final_image"]

        if not plan["steps"]:


            context.logger.info(
                "%s: no object steps; passing the base first frame through", item.sample_id
            )
            self._copy(base_frame, output_path)
            return ItemResult.succeeded(
                item.sample_id,
                image=context.relative(output_path),
                edit_steps=[],
                step_count=0,
                edited=False,
                mode=plan["mode"],
                plan_file=context.relative(plan_path) if self.save_plan else None,
            )

        self._ensure_models(context)
        self._run_steps(plan, item, base_frame, context)

        if not output_path.exists():
            raise RuntimeError(f"Qwen edit reported success but wrote no image at {output_path}")

        return ItemResult.succeeded(
            item.sample_id,
            image=context.relative(output_path),
            edit_steps=[step["step"] for step in plan["steps"]],
            step_count=len(plan["steps"]),
            edited=True,
            mode=plan["mode"],
            categories=[step["category"] for step in plan["steps"]],
            plan_file=context.relative(plan_path) if self.save_plan else None,
            seed=self.seed,
        )


    def _ensure_models(self, context: StageContext) -> None:

        if self._models_loaded:
            return

        import torch

        from keyid.stages._diffsynth_qwen import (
            build_qwen_image_pipeline,
            load_lora,
            match_paths,
        )

        self._torch = torch
        edit_root = Path(context.require_param("edit_root"))
        self._lora_path = context.param("lora_path")
        if not edit_root.is_dir():
            raise StageError(f"{self.type}: edit_root is not a directory: {edit_root}")

        context.logger.info("loading Qwen-Image-Edit pipeline: %s", edit_root)
        pipe = build_qwen_image_pipeline(
            edit_root=edit_root,
            match_paths=match_paths,
            torch=torch,
            torch_device=context.device or "cuda",
            torch_dtype=torch.bfloat16,
        )
        if self._lora_path:
            lora_path = Path(self._lora_path)
            if not lora_path.is_file():
                raise StageError(f"{self.type}: lora_path not found: {lora_path}")
            load_lora(pipe, lora_path, alpha=self.lora_strength)
        self._pipe = pipe
        self._models_loaded = True

    def _run_steps(
        self, plan: dict, item: WorkItem, base_frame: Path, context: StageContext
    ) -> None:

        current = base_frame
        total = len(plan["steps"])
        for index, step in enumerate(plan["steps"], start=1):
            reference = (self.reference_root / step["reference_source"]).resolve()
            if not reference.is_file():
                raise FileNotFoundError(
                    f"{item.sample_id}: reference image for step {index} not found: {reference}"
                )
            output = item.item_dir / step["output_image"]
            context.logger.info(
                "%s: edit step %d/%d [%s] -> %s",
                item.sample_id,
                index,
                total,
                step["category"],
                output.name,
            )
            self._edit_one(current, reference, output, step["prompt"], context)
            current = output

    def _edit_one(self, current: Path, reference: Path, output: Path, prompt: str, context) -> None:

        from keyid.stages._diffsynth_qwen import open_rgb, save_image_atomic

        current_image = open_rgb(current)
        reference_image = open_rgb(reference)
        with self._torch.inference_mode():
            image = self._pipe(
                prompt,
                edit_image=[current_image, reference_image],
                seed=self.seed,
                num_inference_steps=self.num_inference_steps,
                height=current_image.height,
                width=current_image.width,
                edit_image_auto_resize=True,
                zero_cond_t=True,
                cfg_scale=self.cfg_scale,
            )
        save_image_atomic(image, output)

    @staticmethod
    def _copy(source: Path, destination: Path) -> None:
        import shutil

        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, destination)
