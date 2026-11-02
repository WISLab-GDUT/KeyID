from __future__ import annotations

import argparse
import os
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
from keyid.core.seeding import resolve_seed

from .._dataset_prompt import DEFAULT_RELAY_CHUNK_SIZE


DEFAULT_INFER_ARGS: dict[str, object] = {
    "model_cls": "wan2.2_moe_distill",
    "task": "i2v",
    "support_tasks": [],
    "sf_model_path": None,
    "use_prompt_enhancer": False,
    "action_ckpt": None,
    "subfolder": None,
    "disable_heads": None,
    "enable_bf16": False,
    "save_rendered": False,
    "render_interp_per_pair": None,
    "render_depth": False,
    "wm_config_path": None,
    "wm_ckpt_path": None,
    "num_iterations": None,
}


def resolve_checkpoint_paths(config, config_json: str | Path) -> None:

    base_dir = Path(config_json).resolve().parent
    for key in ("high_noise_original_ckpt", "low_noise_original_ckpt"):
        value = config.get(key)
        if isinstance(value, str) and value.strip():
            path = Path(value).expanduser()
            config[key] = str(path if path.is_absolute() else (base_dir / path).resolve())


class DraftVideoStage(BatchStage):


    type = "lightx2v.draft_video"
    needs = ("first_frame", "draft_prompt")
    default_first_frame_sources = ("first_frame.image",)
    description = "generate items/<sample_id>/video.mp4 with Wan2.2 I2V (LightX2V, distilled 4-step)"


    def setup(self, context: StageContext) -> None:
        self.first_frame_sources = tuple(
            str(source) for source in (context.param("first_frame_sources") or ())
        ) or self.default_first_frame_sources
        self._prepare_imports(context)
        self._after_imports(context)

        from lightx2v.infer import init_runner
        from lightx2v.utils.set_config import print_config, set_config, set_parallel_config
        from lightx2v.utils.utils import validate_config_paths
        from lightx2v_platform.registry_factory import PLATFORM_DEVICE_REGISTER

        self.platform = str(context.param("platform", "cuda"))


        os.environ["PLATFORM"] = self.platform

        infer_args = argparse.Namespace(**self._infer_args(context))
        config = set_config(infer_args)
        resolve_checkpoint_paths(config, infer_args.config_json)
        if config.get("parallel"):
            platform_device = PLATFORM_DEVICE_REGISTER.get(os.environ.get("PLATFORM", "cuda"), None)
            if platform_device is None:
                raise StageError(f"{self.type}: no platform device registered for {self.platform!r}")
            platform_device.init_parallel_env()
            set_parallel_config(config)
        print_config(config)
        validate_config_paths(config)

        context.logger.info(
            "initialising LightX2V runner: model=%s config=%s",
            infer_args.model_path,
            infer_args.config_json,
        )

        self.runner = init_runner(config)
        self.seed = int(context.param("seed", 42))
        self.negative_prompt = str(context.require_param("negative_prompt"))
        context.logger.info("LightX2V runner ready (seed=%d)", self.seed)

    def process_batch(self, items: list[WorkItem], context: StageContext) -> list[ItemResult]:
        return [
            guard(item, lambda item=item: self._process_one(item, context), context.logger)
            for item in items
        ]

    def close(self, context: StageContext) -> None:
        if getattr(self, "runner", None) is not None:
            del self.runner
        try:
            import torch.distributed as dist

            if dist.is_initialized():
                dist.destroy_process_group()
                context.logger.info("distributed process group destroyed")
        except Exception:
            context.logger.debug("distributed teardown failed", exc_info=True)
        context.logger.info("draft_video closed")


    @staticmethod
    def _prompt(item: WorkItem) -> str:

        return str(
            require_upstream_output(
                item, "draft_prompt", "prompt", what="the dataset prompt for the draft"
            )
        ).strip()

    def _process_one(self, item: WorkItem, context: StageContext) -> ItemResult:
        from lightx2v.utils.input_info import init_empty_input_info, update_input_info_from_dict
        from lightx2v.utils.utils import seed_all

        self._selected_first_frame = None
        image_path = self._first_frame_path(item, context)
        prompt = self._prompt(item)

        output_path = item.item_dir / "video.mp4"
        seed = self._seed_for(item.sample_id)
        seed_all(seed)

        input_info = init_empty_input_info("i2v", [])
        update_input_info_from_dict(
            input_info,
            {
                "seed": seed,
                "prompt": prompt,
                "negative_prompt": self.negative_prompt,
                "image_path": str(image_path),
                "save_result_path": str(output_path),
                "return_result_tensor": False,
            },
        )
        self._apply_relay(input_info, item, context)
        self.runner.run_pipeline(input_info)

        if not output_path.exists():
            raise RuntimeError(f"LightX2V reported success but wrote no video at {output_path}")

        return ItemResult.succeeded(
            item.sample_id,
            video=context.relative(output_path),
            seed=seed,
            prompt=prompt,


            first_frame=image_path.name,
            first_frame_source=self._selected_first_frame,
        )


    def _first_frame_path(self, item: WorkItem, context: StageContext):

        tried = []
        for reference in self.first_frame_sources:
            namespace, _, key = str(reference).partition(".")
            value = (item.inputs.get(namespace) or {}).get(key)
            if not isinstance(value, str) or not value.strip():
                tried.append(f"{reference} (absent)")
                continue
            path = context.input_path(namespace, value)
            if not path.exists():
                tried.append(f"{reference} (missing on disk)")
                continue
            self._selected_first_frame = reference
            return path
        raise ValueError(
            f"{item.sample_id}: no first frame available from {list(self.first_frame_sources)}; "
            f"tried {tried}"
        )

    def _apply_relay(self, input_info, item: WorkItem, context: StageContext) -> None:
        pass

    def _relay_spec(self, item: WorkItem, context: StageContext):

        outputs = item.input("draft_prompt")
        spec = outputs.get("prompt_relay")
        if not isinstance(spec, dict):
            raise ValueError(
                f"{item.sample_id}: {type(self).__name__} needs a prompt_relay from "
                "draft_prompt, but none was produced (is temporal_caption_field configured?)"
            )
        return spec

    def _seed_for(self, sample_id: str) -> int:


        return resolve_seed(self.seed, stage=self.type, sample_id=sample_id)


    def _after_imports(self, context: StageContext) -> None:
        pass

    def _infer_args(self, context: StageContext) -> dict[str, object]:
        infer_args = dict(DEFAULT_INFER_ARGS)
        infer_args.update(context.param("infer_args", {}) or {})
        infer_args["model_path"] = str(context.require_param("model_path"))
        infer_args["config_json"] = str(context.require_param("config_json"))
        return infer_args

    def _prepare_imports(self, context: StageContext) -> None:

        import sys

        lightx2v_root = Path(context.require_param("lightx2v_path"))
        if not (lightx2v_root / "lightx2v" / "__init__.py").is_file():
            raise StageError(
                f"{self.type}: no lightx2v package at {lightx2v_root}. Point "
                "params.lightx2v_path at your external LightX2V checkout root."
            )
        for name in ("", "lightx2v_kernel"):
            candidate = lightx2v_root if name == "" else lightx2v_root / name
            if candidate.is_dir() and str(candidate) not in sys.path:
                sys.path.insert(0, str(candidate))
        context.logger.info("lightx2v imported from %s", lightx2v_root)


class DraftVideoRelayStage(DraftVideoStage):


    type = "lightx2v.draft_video_relay"
    needs = ("first_frame", "draft_prompt", "object_edit")
    default_first_frame_sources = ("object_edit.image", "first_frame.image")
    description = (
        "generate items/<sample_id>/video.mp4 with Wan2.2 I2V + Prompt-Relay "
        "(Track2 temporal captions)"
    )

    def setup(self, context: StageContext) -> None:

        self.relay_chunk_size = int(
            context.param("chunk_size", DEFAULT_RELAY_CHUNK_SIZE)
        )
        super().setup(context)


    def _after_imports(self, context: StageContext) -> None:
        from .prompt_relay import install_prompt_relay_patch

        install_prompt_relay_patch(chunk_size=self.relay_chunk_size)
        context.logger.info(
            "prompt-relay runtime patch installed (chunk_size=%d)", self.relay_chunk_size
        )

    def _apply_relay(self, input_info, item: WorkItem, context: StageContext) -> None:
        spec = self._relay_spec(item, context)

        input_info.prompt_relay = spec
        context.logger.info(
            "prompt-relay attached: %d local prompt(s), global %.40s...",
            len(spec.get("local_prompts") or []),
            spec.get("global_prompt", ""),
        )

    def _process_one(self, item: WorkItem, context: StageContext):
        result = super()._process_one(item, context)
        result.outputs["relay_chunk_size"] = self.relay_chunk_size
        return result
