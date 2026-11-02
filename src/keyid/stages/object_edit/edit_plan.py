from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Mapping

from .._dataset_prompt import HUMAN_CATEGORIES, image_category

LAYOUT_CAMERA_SUFFIX = "图片布局不变，图片镜头不变"

CATEGORY_TARGETS = {
    "ball": "画面中的球",
    "bird": "画面中的鸟",
    "box": "画面中的木箱",
    "cake": "画面中的蛋糕",
    "camera": "画面中的相机",
    "cat": "画面中的猫",
    "clothes": "画面中人物的服装",
    "cup": "画面中的杯子",
    "dog": "画面中的狗",
    "glasses": "画面中人物的眼镜",
    "hairdryer": "画面中的吹风机",
    "hammer": "画面中的法槌",
    "headphone": "画面中人物的耳机",
    "juice": "画面中的饮料",
    "mouse": "画面中的小动物",
    "pie": "画面中的派",
    "pot": "画面中的碗",
    "robot": "画面中的机器人",
    "shoes": "画面中人物的鞋子",
    "teddybear": "画面中的泰迪熊",

    "horse": "画面中的马",
    "rabbit": "画面中的兔子",
    "red_panda": "画面中的小熊猫",

    "bag": "画面中的包",
    "beer": "画面中的啤酒",
    "bike": "画面中的自行车",
    "boat": "画面中的船",
    "bowl": "画面中的碗",
    "clock": "画面中的时钟",
    "earphone": "画面中人物的耳机",
    "guitar": "画面中的吉他",
    "hat": "画面中人物的帽子",
    "lamp": "画面中的路灯",
    "motor": "画面中的摩托车",
    "ring": "画面中人物的戒指",
    "sofa": "画面中的沙发",
    "toy": "画面中的玩具",
    "tv": "画面中的电视",
    "watch": "画面中人物的手表",
}

CATEGORY_REFERENCE_NAMES = {
    "ball": "球",
    "bird": "鸟",
    "box": "木箱",
    "cake": "蛋糕",
    "camera": "相机",
    "cat": "猫",
    "cup": "杯子",
    "dog": "狗",
    "hammer": "法槌",
    "juice": "饮料",
    "mouse": "小动物",
    "pie": "派",
    "pot": "碗",
    "robot": "机器人",
    "teddybear": "泰迪熊",

    "horse": "马",
    "rabbit": "兔子",
    "red_panda": "小熊猫",

    "bag": "包",
    "beer": "啤酒",
    "bike": "自行车",
    "boat": "船",
    "bowl": "碗",
    "clock": "时钟",
    "earphone": "耳机",
    "guitar": "吉他",
    "hat": "帽子",
    "lamp": "路灯",
    "motor": "摩托车",
    "ring": "戒指",
    "sofa": "沙发",
    "toy": "玩具",
    "tv": "电视",
    "watch": "手表",
}

HEAD_CATEGORIES = HUMAN_CATEGORIES


def relative_to_task(task_dir: Path, target_path: Path) -> str:

    return Path(os.path.relpath(target_path.resolve(), task_dir.resolve())).as_posix()


def category_name(category: str) -> str:

    return CATEGORY_TARGETS.get(category, f"画面中的{category}")


def human_target(item_key: str, human_index: int, category: str) -> str:

    if category == "man":
        return "画面中的男性"
    if category == "woman":
        return "画面中的女性"
    return "画面中的人物"


def build_prompt(category: str, target: str) -> str:

    if category in HUMAN_CATEGORIES:
        action = f"将{target}换成第2张图的头"
    elif category == "clothes":
        action = f"将{target}换成第2张图的服装"
    elif category == "glasses":
        action = f"将{target}换成第2张图的眼镜"
    elif category == "headphone":
        action = f"将{target}换成第2张图的耳机"
    elif category == "hairdryer":
        action = f"将{target}换成第2张图的吹风机"
    elif category == "shoes":
        action = f"将{target}换成第2张图的鞋子"
    else:
        reference_name = CATEGORY_REFERENCE_NAMES.get(category, "对应物体")
        action = f"将{target}换成第2张图的{reference_name}"
    return f"{action}，{LAYOUT_CAMERA_SUFFIX}"


def build_steps(
    item_key: str,
    fields_source: Mapping[str, Any],
    *,
    edit_heads: bool,
    reference_root: Path | None = None,
    task_dir: Path | None = None,
) -> dict[str, Any]:

    img_paths = [str(path_value) for path_value in fields_source.get("img_paths", [])]
    categories = [image_category(path_value) for path_value in img_paths]
    category_totals = {category: categories.count(category) for category in set(categories)}
    category_seen: dict[str, int] = {}
    human_seen = 0

    planned: list[dict[str, Any]] = []
    for step_index, path_value in enumerate(img_paths, start=1):
        category = image_category(path_value)
        category_seen[category] = category_seen.get(category, 0) + 1
        category_index = category_seen[category]

        if category in HUMAN_CATEGORIES:
            human_seen += 1
            target = human_target(item_key, human_seen, category)
        elif category == "clothes" and category_totals.get(category) == 2:
            target = "画面中人物的上衣" if category_index == 1 else "画面中人物的下装"
        else:
            target = category_name(category)

        planned.append(
            {
                "source_step": step_index,
                "reference_source": path_value,
                "category": category,
                "target": target,
                "prompt": build_prompt(category, target),
            }
        )

    selected = [
        step for step in planned if edit_heads or step["category"] not in HEAD_CATEGORIES
    ]

    steps: list[dict[str, Any]] = []
    total = len(selected)
    for index, step in enumerate(selected, start=1):
        step = dict(step)
        step["step"] = index
        step["output_image"] = _output_name(index, total, edit_heads=edit_heads)
        if reference_root is not None:
            reference_path = Path(reference_root) / step["reference_source"]
            step["reference_image"] = (
                relative_to_task(task_dir, reference_path)
                if task_dir is not None
                else str(reference_path)
            )
        steps.append(step)

    return {
        "schema_version": 1,
        "item_id": item_key,
        "mode": "full" if edit_heads else "objects_only",
        "base_image": "prompt1_pe_gpt.png",
        "source_img_paths": img_paths,
        "final_image": _final_name(edit_heads=edit_heads),
        "steps": steps,
    }


def _final_name(*, edit_heads: bool) -> str:

    return "prompt1_pe_gpt_edited.png" if edit_heads else "prompt1_pe_gpt_object_edited.png"


def _output_name(index: int, total: int, *, edit_heads: bool) -> str:

    if index == total:
        return _final_name(edit_heads=edit_heads)
    if edit_heads:
        return f"edit_step_{index:02d}.png"
    return f"object_edit_step_{index:02d}.png"
