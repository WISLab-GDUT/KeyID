from __future__ import annotations

from pathlib import Path

from keyid.core.contract import (
    BatchStage,
    ItemResult,
    StageContext,
    WorkItem,
    guard,
    map_concurrently,
)

from .._dataset_prompt import (
    resolve_captions_for_sample,
    resolve_original_prompt,
    validate_caption_field,
    validate_prompt_fields,
    with_temporal_captions,
)
from .._openai_client import build_client, read_text_resource, require_text, response_summary

PACKAGED_PROMPT = Path(__file__).resolve().parent / "prompts" / "prompt_compose.txt"

ORIGINAL_PROMPT_PLACEHOLDER = "{原提示词}"
REFERENCE_PROMPT_PLACEHOLDER = "{模型自带的、更符合模型风格的增强提示词}"
SUBJECT_PROMPT_PLACEHOLDER = "{主体描述；若没有则写“无”}"


class PromptComposeStage(BatchStage):
    type = "openai.prompt_compose"
    needs = ("prompt_pe", "head_extract")
    description = "compose the final generation prompt from the PE prompt and the head prompt"

    def setup(self, context: StageContext) -> None:
        self.model_name = str(context.require_param("model_name"))
        self.reasoning_effort = str(context.param("reasoning_effort", "low"))
        self.concurrency = max(1, int(context.param("concurrency", context.concurrency)))
        self.rule_prompt = read_text_resource(context.param("rule_prompt_path"), PACKAGED_PROMPT)
        self.prompt_fields = validate_prompt_fields(
            context.param("prompt_fields", ["prompt"]), where=f"{self.type}.prompt_fields"
        )
        self.caption_field = validate_caption_field(
            context.param("temporal_caption_field"), where=f"{self.type}.temporal_caption_field"
        )
        self.client = build_client(
            context.credentials.get("openai", {}),
            context.endpoints.get("openai", {}),
            timeout=context.param("timeout"),
            params=context.params,
        )

        for placeholder in (
            ORIGINAL_PROMPT_PLACEHOLDER,
            REFERENCE_PROMPT_PLACEHOLDER,
            SUBJECT_PROMPT_PLACEHOLDER,
        ):
            if placeholder not in self.rule_prompt:
                raise ValueError(f"prompt template is missing the placeholder {placeholder}")

        context.logger.info(
            "prompt_compose ready: model=%s concurrency=%d", self.model_name, self.concurrency
        )

    def process_batch(self, items: list[WorkItem], context: StageContext) -> list[ItemResult]:
        return map_concurrently(
            items,
            lambda item: guard(item, lambda: self._process_one(item, context), context.logger),
            concurrency=self.concurrency,
            logger=context.logger,
        )

    def close(self, context: StageContext) -> None:
        client = getattr(self, "client", None)
        if client is not None:
            try:
                client.close()
            except Exception:
                context.logger.debug("closing the api client failed", exc_info=True)
        context.logger.info("prompt_compose closed")


    def _process_one(self, item: WorkItem, context: StageContext) -> ItemResult:
        fields = dict(item.raw.get("fields") or {})
        original_prompt, prompt_field = resolve_original_prompt(
            fields, self.prompt_fields, sample_id=item.sample_id
        )
        captions = resolve_captions_for_sample(
            fields, self.caption_field, sample_id=item.sample_id
        )
        if captions:


            original_prompt = with_temporal_captions(original_prompt, captions)
        reference_prompt = self._reference_prompt(item)
        subject_prompt = self._subject_prompt(item)

        input_text = (
            self.rule_prompt.replace(ORIGINAL_PROMPT_PLACEHOLDER, original_prompt)
            .replace(REFERENCE_PROMPT_PLACEHOLDER, reference_prompt)
            .replace(SUBJECT_PROMPT_PLACEHOLDER, subject_prompt)
        )

        response = self.client.responses.create(
            model=self.model_name,
            input=[{"role": "user", "content": [{"type": "input_text", "text": input_text}]}],
            reasoning={"effort": self.reasoning_effort},
        )
        prompt = require_text(response.output_text, what=f"{item.sample_id}: api response")
        if not prompt:
            raise ValueError("api returned an empty composed prompt")

        return ItemResult.succeeded(
            item.sample_id,
            prompt=prompt,
            original_prompt=original_prompt,
            prompt_field=prompt_field,
            temporal_captions=list(captions),
            reference_prompt=reference_prompt,
            subject_prompt=subject_prompt,
            model=self.model_name,
            api=response_summary(response),
        )

    @staticmethod
    def _reference_prompt(item: WorkItem) -> str:

        pe_outputs = item.input("prompt_pe")
        for key in ("pe_prompt", "reference_prompt", "prompt"):
            value = pe_outputs.get(key)
            if isinstance(value, str) and value.strip():
                return value.strip()
        raise KeyError(f"{item.sample_id}: prompt_pe produced no usable prompt")

    @staticmethod
    def _subject_prompt(item: WorkItem) -> str:

        head_prompt = item.input("head_extract").get("head_prompt")
        if isinstance(head_prompt, str) and head_prompt.strip():
            return head_prompt.strip()

        labels = (item.raw.get("fields") or {}).get("class_label") or []
        label_text = ", ".join(str(label) for label in labels if label)
        return f"class_label: {label_text}" if label_text else "无"
