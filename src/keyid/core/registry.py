from __future__ import annotations

import importlib
import inspect
import pkgutil
from pathlib import Path
from typing import Iterable, Mapping

from .contract import BatchStage
from .errors import RegistryError

DEFAULT_STAGES_PACKAGE = "keyid.stages"


class StageRegistry:


    def __init__(self) -> None:
        self._types: dict[str, type[BatchStage]] = {}
        self._sources: dict[str, str] = {}


    def register(self, stage_cls: type[BatchStage], *, source: str | None = None) -> None:
        if not (inspect.isclass(stage_cls) and issubclass(stage_cls, BatchStage)):
            raise RegistryError(f"{stage_cls!r} is not a BatchStage subclass")
        stage_type = getattr(stage_cls, "type", "")
        if not isinstance(stage_type, str) or not stage_type.strip():
            raise RegistryError(f"{stage_cls.__name__} does not declare a non-empty `type`")
        if inspect.isabstract(stage_cls):
            raise RegistryError(f"{stage_cls.__name__} is abstract and cannot be registered")
        if stage_type in self._types:
            previous = self._sources.get(stage_type, "?")
            raise RegistryError(
                f"duplicate stage type {stage_type!r}: defined by {previous} and "
                f"{source or stage_cls.__module__}"
            )
        self._types[stage_type] = stage_cls
        self._sources[stage_type] = source or stage_cls.__module__


    def get(self, stage_type: str) -> type[BatchStage]:
        try:
            return self._types[stage_type]
        except KeyError:
            raise RegistryError(
                f"unknown stage type {stage_type!r}; discovered: {sorted(self._types)}"
            ) from None

    def __contains__(self, stage_type: object) -> bool:
        return stage_type in self._types

    def types(self) -> list[str]:
        return sorted(self._types)

    def source_of(self, stage_type: str) -> str:
        return self._sources.get(stage_type, "?")

    def default_needs(self, stage_type: str) -> tuple[str, ...]:
        return tuple(self.get(stage_type).default_needs())

    def describe(self) -> list[dict[str, object]]:
        return [
            {
                "type": stage_type,
                "class": self._types[stage_type].__name__,
                "needs": list(self._types[stage_type].default_needs()),
                "description": self._types[stage_type].description,
                "module": self._sources[stage_type],
            }
            for stage_type in self.types()
        ]


    def discover(
        self,
        package: str = DEFAULT_STAGES_PACKAGE,
        *,
        search_path: Iterable[Path] | None = None,
    ) -> int:

        try:
            root = importlib.import_module(package)
        except ImportError as exc:
            raise RegistryError(f"cannot import stage package {package!r}: {exc}") from exc

        if search_path is not None:
            paths = [str(Path(entry)) for entry in search_path]
        else:
            paths = list(getattr(root, "__path__", []))

        before = len(self._types)
        for module_info in sorted(pkgutil.iter_modules(paths), key=lambda info: info.name):
            if not module_info.ispkg:
                continue
            module_name = f"{package}.{module_info.name}"
            try:
                module = importlib.import_module(module_name)
            except Exception as exc:
                raise RegistryError(f"cannot import stage package {module_name!r}: {exc}") from exc
            self._register_module(module, module_name)
        return len(self._types) - before

    def _register_module(self, module: object, module_name: str) -> None:

        prefix = f"{module_name}."
        for _, obj in sorted(vars(module).items()):
            if not (inspect.isclass(obj) and issubclass(obj, BatchStage)):
                continue
            if obj is BatchStage:
                continue
            origin = getattr(obj, "__module__", "")
            if origin != module_name and not origin.startswith(prefix):
                continue
            if inspect.isabstract(obj):

                continue
            self.register(obj, source=module_name)


def build_default_registry(packages: Iterable[str] = (DEFAULT_STAGES_PACKAGE,)) -> StageRegistry:
    registry = StageRegistry()
    for package in packages:
        registry.discover(package)
    return registry


def package_parent_dirs(packages: Iterable[str]) -> list[Path]:

    parents: list[Path] = []
    for package in packages:
        try:
            module = importlib.import_module(package)
        except ImportError as exc:
            raise RegistryError(f"cannot import stage package {package!r}: {exc}") from exc
        module_file = getattr(module, "__file__", None)
        if not module_file:
            continue
        parent = Path(module_file).resolve().parent.parent
        if parent not in parents:
            parents.append(parent)
    return parents


def stage_types_from(config_stages: Mapping[str, object]) -> dict[str, str]:

    mapping: dict[str, str] = {}
    for stage_id, stage_config in config_stages.items():
        mapping[str(stage_id)] = str(getattr(stage_config, "type", ""))
    return mapping
