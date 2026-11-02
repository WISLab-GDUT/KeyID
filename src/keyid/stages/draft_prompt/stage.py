from __future__ import annotations

from keyid.core.contract import BatchStage, ItemResult, StageContext, WorkItem, guard

from .._dataset_prompt import (
    DEFAULT_RELAY_MIN_WINDOW,
    DEFAULT_RELAY_SIGMA,
    DEFAULT_RELAY_WINDOW_OFFSET,
    build_relay_spec,
    full_prompt_from_relay_spec,
    resolve_captions_for_sample,
    resolve_original_prompt,
    temporal_prompt_text,
    validate_caption_field,
    validate_prompt_fields,
)


class DraftPromptStage(BatchStage):
    type = "dataset.draft_prompt"
    needs = ()
    description = "resolve the dataset prompt used by draft_video (and the relay spec for Track2)"

    def setup(self, context: StageContext) -> None:


        self.prompt_fields = validate_prompt_fields(
            context.param("prompt_fields", ["prompt"]),
            where=f"{self.type}.prompt_fields",
        )
        self.caption_field = validate_caption_field(
            context.param("temporal_caption_field"), where=f"{self.type}.temporal_caption_field"
        )
        self.sigma = float(context.param("sigma", DEFAULT_RELAY_SIGMA))
        self.window_offset = int(context.param("window_offset", DEFAULT_RELAY_WINDOW_OFFSET))
        self.min_window = int(context.param("min_window", DEFAULT_RELAY_MIN_WINDOW))

        context.logger.info(
            "draft_prompt ready: prompt_fields=%s temporal_caption_field=%s",
            list(self.prompt_fields),
            self.caption_field or "<none>",
        )

    def process_batch(self, items: list[WorkItem], context: StageContext) -> list[ItemResult]:
        return [
            guard(item, lambda item=item: self._process_one(item, context), context.logger)
            for item in items
        ]

    def _process_one(self, item: WorkItem, context: StageContext) -> ItemResult:
        fields = dict(item.raw.get("fields") or {})
        prompt, prompt_field = resolve_original_prompt(
            fields, self.prompt_fields, sample_id=item.sample_id
        )

        outputs: dict[str, object] = {
            "prompt": prompt,
            "prompt_field": prompt_field,
            "prompt_fields": list(self.prompt_fields),
        }

        captions = resolve_captions_for_sample(
            fields, self.caption_field, sample_id=item.sample_id
        )
        if not captions:

            return ItemResult.succeeded(item.sample_id, **outputs)

        relay_spec = build_relay_spec(
            prompt,
            captions,
            prompt_field=prompt_field,
            sigma=self.sigma,
            window_offset=self.window_offset,
            min_window=self.min_window,
        )
        outputs["prompt_relay"] = relay_spec
        outputs["temporal_prompt"] = temporal_prompt_text(captions)


        outputs["prompt"] = full_prompt_from_relay_spec(relay_spec)
        return ItemResult.succeeded(item.sample_id, **outputs)
