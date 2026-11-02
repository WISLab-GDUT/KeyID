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
from keyid.core.errors import StageError

from .._dataset_prompt import HUMAN_CATEGORIES, image_category
from .._openai_client import (
    build_client,
    extract_after_marker,
    image_to_data_url,
    read_text_resource,
    require_text,
    response_summary,
)

PROMPTS_DIR = Path(__file__).resolve().parent / "prompts"


TEMPLATE_VARIANTS = {
    "single": PROMPTS_DIR / "head_extract.txt",
    "multi": PROMPTS_DIR / "head_extract_multi.txt",
}
DEFAULT_TEMPLATE = "single"

PACKAGED_PROMPT = TEMPLATE_VARIANTS[DEFAULT_TEMPLATE]


class HeadExtractStage(BatchStage):
    type = "openai.head_extract"
    needs = ()
    description = "extract a conservative head prompt from the reference image via the Responses API"

    def setup(self, context: StageContext) -> None:
        self.model_name = str(context.require_param("model_name"))
        self.send_images = bool(context.param("send_images_to_api", True))
        if not self.send_images:
            raise StageError(
                f"{self.type}: send_images_to_api is false, so this stage cannot produce anything. "
                "Set it to true once you have confirmed the endpoint may receive local images, "
                "or disable the stage in YAML."
            )
        self.detail = str(context.param("image_detail", "high"))
        self.reasoning_effort = str(context.param("reasoning_effort", "none"))
        self.image_index = int(context.param("image_index", 0))


        raw_categories = context.param("reference_categories")
        if raw_categories is None:
            self.reference_categories: tuple[str, ...] | None = None
        elif isinstance(raw_categories, (list, tuple)):
            self.reference_categories = tuple(
                str(category).strip().lower() for category in raw_categories
            )
            if not self.reference_categories or any(not c for c in self.reference_categories):
                raise StageError(
                    f"{self.type}: reference_categories must list at least one category"
                )
        else:
            raise StageError(
                f"{self.type}: reference_categories must be a list of categories, "
                f"got {raw_categories!r}"
            )
        self.concurrency = max(1, int(context.param("concurrency", context.concurrency)))
        self.marker = str(context.param("final_marker", "最终提示词："))

        template_name = str(context.param("template", DEFAULT_TEMPLATE)).strip()
        if template_name not in TEMPLATE_VARIANTS:
            raise StageError(
                f"{self.type}: template must be one of {sorted(TEMPLATE_VARIANTS)}, "
                f"got {template_name!r}"
            )
        self.template_name = template_name
        self.rule_prompt = read_text_resource(
            context.param("rule_prompt_path"), TEMPLATE_VARIANTS[template_name]
        )
        self.client = build_client(
            context.credentials.get("openai", {}),
            context.endpoints.get("openai", {}),
            timeout=context.param("timeout"),
            params=context.params,
        )

        context.logger.info(
            "head_extract ready: model=%s template=%s concurrency=%d images_sent=yes "
            "reference_categories=%s",
            self.model_name,
            self.template_name,
            self.concurrency,
            list(self.reference_categories) if self.reference_categories else "<single>",
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
        context.logger.info("head_extract closed")


    def _process_one(self, item: WorkItem, context: StageContext) -> ItemResult:
        image_paths = self._reference_images(item, context)

        if not image_paths:


            context.logger.info(
                "no human reference image for %s; head prompt left empty", item.sample_id
            )
            return ItemResult.succeeded(
                item.sample_id,
                head_prompt="",
                head_prompts=[],
                model=self.model_name,
                template=self.template_name,
                reference_images=[],
                skipped_reason="no human reference image",
                api={},
            )

        for image_path in image_paths:
            with open(image_path, "rb") as handle:
                handle.read(1)

        response = self.client.responses.create(
            model=self.model_name,
            input=self._build_input(item, image_paths),
            reasoning={"effort": self.reasoning_effort},
        )

        text = require_text(response.output_text, what="api response")
        if len(image_paths) == 1:
            head_prompt = extract_after_marker(text, self.marker)
            if not head_prompt:
                raise ValueError("api returned an empty head prompt")
            head_prompts = [head_prompt]
        else:
            head_prompts = self._parse_ordered_prompts(text, len(image_paths))
            if not head_prompts:
                raise ValueError(
                    f"api returned no per-subject head prompt for {len(image_paths)} reference image(s)"
                )
            head_prompt = self._format_ordered_prompts(head_prompts)

        return ItemResult.succeeded(
            item.sample_id,
            head_prompt=head_prompt,
            head_prompts=head_prompts,
            model=self.model_name,
            template=self.template_name,
            reference_images=[path.name for path in image_paths],
            api=response_summary(response),
        )


    def _build_input(self, item: WorkItem, image_paths: list[Path]) -> list[dict]:
        if len(image_paths) == 1:

            return [
                {
                    "role": "developer",
                    "content": [{"type": "input_text", "text": self.rule_prompt}],
                },
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "input_text",
                            "text": f"请根据这张参考图提取头部提示词。样本ID：{item.sample_id}",
                        },
                        {
                            "type": "input_image",
                            "image_url": image_to_data_url(image_paths[0]),
                            "detail": self.detail,
                        },
                    ],
                },
            ]

        count = len(image_paths)
        content: list[dict] = [
            {
                "type": "input_text",
                "text": (
                    f"请根据下面按顺序提供的{count}张人类参考图提取头部提示词。"
                    f"样本ID：{item.sample_id}。每张图对应一个主体，"
                    "请严格按输入图片顺序输出第1张人脸、第2张人脸……"
                ),
            }
        ]


        for index, image_path in enumerate(image_paths, start=1):
            content.append(
                {
                    "type": "input_text",
                    "text": f"第{index}张人类参考图：只描述这张图中人类主体的头部特征。",
                }
            )
            content.append(
                {
                    "type": "input_image",
                    "image_url": image_to_data_url(image_path),
                    "detail": self.detail,
                }
            )
        return [
            {"role": "developer", "content": [{"type": "input_text", "text": self.rule_prompt}]},
            {"role": "user", "content": content},
        ]

    @staticmethod
    def _parse_ordered_prompts(text: str, image_count: int) -> list[str]:

        final_text = extract_after_marker(text, "最终提示词：")
        prompts: list[str] = []
        for raw_line in final_text.splitlines():
            line = raw_line.strip()
            if not line or line == "最终提示词：":
                continue
            for separator in ("：", ":"):
                if separator not in line:
                    continue
                prefix, candidate = line.split(separator, 1)
                prefix = prefix.strip()
                if (prefix.startswith("第") and "张" in prefix) or prefix.rstrip(".、").isdigit():
                    if candidate.strip():
                        prompts.append(candidate.strip())
                    break
        if len(prompts) == image_count:
            return prompts
        if image_count == 1 and final_text.strip():
            return [final_text.strip()]
        return []

    @staticmethod
    def _format_ordered_prompts(prompts: list[str]) -> str:

        return "\n".join(
            f"第{index}张人脸：{head_prompt}"
            for index, head_prompt in enumerate(prompts, start=1)
        )

    def _reference_images(self, item: WorkItem, context: StageContext) -> list[Path]:

        if self.reference_categories is None:


            return [context.resolve_image(item, self.image_index)]

        raw_paths = [str(value) for value in (item.raw.get("img_paths") or ())]
        selected = [
            value
            for value in raw_paths
            if image_category(value) in self.reference_categories
        ]
        if not selected:
            return []
        paths = [context.resolve_image(item, raw_paths.index(value)) for value in selected]
        return paths
