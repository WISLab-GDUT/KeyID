from __future__ import annotations

import copy
import os
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Iterator, Mapping

import yaml

from . import secrets as secrets_module
from .devices import check_device_environment_conflict
from .errors import ConfigError

SUPPORTED_SCHEMA_VERSIONS = (1,)


_PATH_KEY_PATTERNS = (
    re.compile(r".*_path$"),
    re.compile(r".*_paths$"),
    re.compile(r".*_root$"),
    re.compile(r".*_dir$"),
    re.compile(r".*_file$"),
    re.compile(r"^path$"),
    re.compile(r"^root$"),
    re.compile(r"^dir$"),
    re.compile(r"^file$"),
    re.compile(r"^manifest$"),
    re.compile(r"^metadata$"),
    re.compile(r"^python$"),
    re.compile(r"^config_json$"),
)

_INTERPOLATION_PATTERN = re.compile(r"\$\{[^}]*\}")

REDACTED_PLACEHOLDER = secrets_module.REDACTED


class _StrictLoader(yaml.SafeLoader):
    pass


def _construct_mapping(loader: _StrictLoader, node: yaml.Node, deep: bool = False) -> dict:
    mapping: dict[Any, Any] = {}
    for key_node, value_node in node.value:
        key = loader.construct_object(key_node, deep=deep)
        if key in mapping:
            raise ConfigError(
                f"duplicate key {key!r} at line {key_node.start_mark.line + 1} "
                f"of {getattr(loader, '_source', '<yaml>')}"
            )
        mapping[key] = loader.construct_object(value_node, deep=deep)
    return mapping


_StrictLoader.add_constructor(
    yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG,
    _construct_mapping,
)


def is_path_key(key: str) -> bool:
    return any(pattern.match(key) for pattern in _PATH_KEY_PATTERNS)


def is_secret_key(key: str) -> bool:

    return secrets_module.is_secret_key(key)


@dataclass(frozen=True)
class DatasetConfig:
    format: str
    root: Path
    metadata: Path


@dataclass(frozen=True)
class RuntimeConfig:
    output_root: Path


    work_dir: Path | None = None
    continue_on_item_error: bool = True
    log_level: str = "INFO"
    run_id: str | None = None


    stage_packages: tuple[str, ...] = ("keyid.stages",)


@dataclass(frozen=True)
class BatchConfig:
    size: int = 1
    checkpoint_every: int = 1

    def __post_init__(self) -> None:
        if self.size < 1:
            raise ConfigError(f"batch.size must be >= 1, got {self.size}")
        if self.checkpoint_every < 1:
            raise ConfigError(f"batch.checkpoint_every must be >= 1, got {self.checkpoint_every}")


@dataclass(frozen=True)
class StageConfig:
    stage_id: str
    type: str
    enabled: bool
    needs: tuple[str, ...]
    python: Path
    device: str | None
    batch: BatchConfig
    concurrency: int
    environment: dict[str, str]
    params: dict[str, Any]
    order: int


    needs_declared: bool = False


@dataclass(frozen=True)
class PipelineConfig:
    schema_version: int
    path: Path
    base_dir: Path
    dataset: DatasetConfig
    runtime: RuntimeConfig
    credentials: dict[str, dict[str, Any]]
    stages: dict[str, StageConfig]
    environment: dict[str, str]
    raw: dict[str, Any] = field(repr=False, default_factory=dict)


    endpoints: dict[str, dict[str, Any]] = field(default_factory=dict)


    def stage(self, stage_id: str) -> StageConfig:
        try:
            return self.stages[stage_id]
        except KeyError:
            raise ConfigError(f"unknown stage {stage_id!r}; known: {sorted(self.stages)}") from None

    def enabled_stages(self) -> dict[str, StageConfig]:
        return {sid: stage for sid, stage in self.stages.items() if stage.enabled}

    def credential(self, namespace: str) -> dict[str, Any]:
        return dict(self.credentials.get(namespace, {}))

    def endpoint(self, namespace: str) -> dict[str, Any]:

        return dict(self.endpoints.get(namespace, {}))

    def subprocess_environment(self, stage: StageConfig) -> dict[str, str]:

        merged = dict(self.environment)
        merged.update(stage.environment)
        return merged

    def redacted_snapshot(self) -> dict[str, Any]:

        return redact_secrets(copy.deepcopy(self.raw))


def redact_secrets(node: Any, key: str = "") -> Any:

    return secrets_module.redact(node, key)


def scrub_text(text: str, secrets: Iterable[str]) -> str:

    for secret in secrets:
        if secret and secret in text:
            text = text.replace(secret, REDACTED_PLACEHOLDER)
    return text


def resolve_path(value: Any, base_dir: Path, *, where: str) -> Path:

    if not isinstance(value, (str, Path)) or str(value).strip() == "":
        raise ConfigError(f"{where} must be a non-empty path string, got {value!r}")
    path = Path(str(value)).expanduser()
    resolved = path if path.is_absolute() else (base_dir / path)
    return Path(os.path.normpath(resolved))


def _resolve_node(node: Any, base_dir: Path, where: str) -> Any:

    if isinstance(node, dict):
        resolved: dict[str, Any] = {}
        for key, value in node.items():
            child_where = f"{where}.{key}" if where else str(key)
            if isinstance(value, str) and is_path_key(str(key)):
                resolved[key] = str(resolve_path(value, base_dir, where=child_where))
            else:
                resolved[key] = _resolve_node(value, base_dir, child_where)
        return resolved
    if isinstance(node, list):
        return [_resolve_node(item, base_dir, where) for item in node]
    return node


def _check_no_interpolation(node: Any, where: str = "") -> None:
    if isinstance(node, dict):
        for key, value in node.items():
            _check_no_interpolation(value, f"{where}.{key}" if where else str(key))
    elif isinstance(node, list):
        for index, value in enumerate(node):
            _check_no_interpolation(value, f"{where}[{index}]")
    elif isinstance(node, str) and _INTERPOLATION_PATTERN.search(node):
        raise ConfigError(
            f"{where} uses environment-variable interpolation ({node!r}); "
            "KeyID requires all values to be written literally in YAML"
        )


def _require_mapping(node: Any, where: str) -> Mapping[str, Any]:
    if node is None:
        return {}
    if not isinstance(node, dict):
        raise ConfigError(f"{where} must be a mapping, got {type(node).__name__}")
    return node


def _parse_stage(stage_id: str, node: Any, order: int, base_dir: Path) -> StageConfig:
    where = f"stages.{stage_id}"
    mapping = _require_mapping(node, where)

    stage_type = mapping.get("uses")
    if not isinstance(stage_type, str) or not stage_type.strip():
        raise ConfigError(f"{where}.uses is required and must be a stage type string")

    needs_declared = "needs" in mapping
    needs = mapping.get("needs", None)
    if needs is None or needs == []:
        needs_tuple: tuple[str, ...] = ()
    elif isinstance(needs, list) and all(isinstance(item, str) for item in needs):
        needs_tuple = tuple(needs)
    else:
        raise ConfigError(f"{where}.needs must be a list of stage ids")

    python = mapping.get("python")
    if python is None:
        raise ConfigError(f"{where}.python is required (interpreter for the stage subprocess)")

    batch_node = _require_mapping(mapping.get("batch", {}), f"{where}.batch")
    batch = BatchConfig(
        size=int(batch_node.get("size", 1)),
        checkpoint_every=int(batch_node.get("checkpoint_every", 1)),
    )

    concurrency = int(mapping.get("concurrency", 1))
    if concurrency < 1:
        raise ConfigError(f"{where}.concurrency must be >= 1")

    environment = _require_mapping(mapping.get("environment", {}), f"{where}.environment")
    environment = {str(key): "" if value is None else str(value) for key, value in environment.items()}

    params = _require_mapping(mapping.get("params", {}), f"{where}.params")
    return StageConfig(
        stage_id=stage_id,
        type=stage_type.strip(),
        enabled=bool(mapping.get("enabled", True)),
        needs=needs_tuple,
        python=resolve_path(python, base_dir, where=f"{where}.python"),
        device=(str(mapping["device"]) if mapping.get("device") is not None else None),
        batch=batch,
        concurrency=concurrency,
        environment=environment,
        params=_resolve_node(dict(params), base_dir, f"{where}.params"),
        order=order,
        needs_declared=needs_declared,
    )


def load_config(path: str | Path) -> PipelineConfig:

    config_path = Path(path).expanduser().resolve()
    if not config_path.is_file():
        raise ConfigError(f"config file not found: {config_path}")
    try:
        with open(config_path, "r", encoding="utf-8") as handle:
            loader = _StrictLoader(handle)
            loader._source = str(config_path)
            try:
                raw = loader.get_single_data()
            finally:
                loader.dispose()
    except ConfigError:
        raise
    except yaml.YAMLError as exc:
        raise ConfigError(f"invalid YAML in {config_path}: {exc}") from exc

    if raw is None:
        raise ConfigError(f"config file is empty: {config_path}")
    if not isinstance(raw, dict):
        raise ConfigError(f"config root must be a mapping, got {type(raw).__name__}")

    _check_no_interpolation(raw)

    schema_version = raw.get("schema_version")
    if schema_version not in SUPPORTED_SCHEMA_VERSIONS:
        raise ConfigError(
            f"unsupported schema_version {schema_version!r}; "
            f"expected one of {list(SUPPORTED_SCHEMA_VERSIONS)}"
        )

    base_dir = config_path.parent

    dataset_node = _require_mapping(raw.get("dataset"), "dataset")
    if "root" not in dataset_node:
        raise ConfigError("dataset.root is required")
    metadata = dataset_node.get("metadata")
    if metadata is None:
        raise ConfigError("dataset.metadata is required")
    dataset = DatasetConfig(
        format=str(dataset_node.get("format", "opens2v")),
        root=resolve_path(dataset_node["root"], base_dir, where="dataset.root"),
        metadata=resolve_path(metadata, base_dir, where="dataset.metadata"),
    )

    runtime_node = _require_mapping(raw.get("runtime"), "runtime")
    if "output_root" not in runtime_node:
        raise ConfigError("runtime.output_root is required")
    log_level = str(runtime_node.get("log_level", "INFO")).upper()
    run_id = runtime_node.get("run_id")
    stage_packages = runtime_node.get("stage_packages", ["keyid.stages"])
    if not isinstance(stage_packages, list) or not all(
        isinstance(entry, str) and entry.strip() for entry in stage_packages
    ):
        raise ConfigError("runtime.stage_packages must be a list of importable package names")
    runtime = RuntimeConfig(
        output_root=resolve_path(runtime_node["output_root"], base_dir, where="runtime.output_root"),
        work_dir=(
            resolve_path(runtime_node["work_dir"], base_dir, where="runtime.work_dir")
            if runtime_node.get("work_dir")
            else None
        ),
        continue_on_item_error=bool(runtime_node.get("continue_on_item_error", True)),
        log_level=log_level,
        run_id=(str(run_id) if run_id else None),
        stage_packages=tuple(entry.strip() for entry in stage_packages),
    )

    credentials: dict[str, dict[str, Any]] = {}
    for namespace, values in _require_mapping(raw.get("credentials"), "credentials").items():
        namespace_map = _require_mapping(values, f"credentials.{namespace}")


        misplaced = [key for key in namespace_map if not secrets_module.is_secret_key(str(key))]
        if misplaced:
            raise ConfigError(
                f"credentials.{namespace} holds non-secret field(s) {sorted(misplaced)}. "
                "Everything under `credentials` is redacted in config.snapshot.yaml and "
                "invocation.json, so connection details would be lost to reviewers. "
                f"Move them to `endpoints.{namespace}` and keep only secrets here."
            )
        credentials[str(namespace)] = {
            str(key): _resolve_node(value, base_dir, f"credentials.{namespace}.{key}")
            for key, value in namespace_map.items()
        }

    endpoints: dict[str, dict[str, Any]] = {}
    for namespace, values in _require_mapping(raw.get("endpoints"), "endpoints").items():
        endpoints[str(namespace)] = {
            str(key): _resolve_node(value, base_dir, f"endpoints.{namespace}.{key}")
            for key, value in _require_mapping(values, f"endpoints.{namespace}").items()
        }

    environment = _require_mapping(raw.get("environment", {}), "environment")
    environment = {str(key): "" if value is None else str(value) for key, value in environment.items()}

    stages: dict[str, StageConfig] = {}
    for order, (stage_id, node) in enumerate(
        _require_mapping(raw.get("stages"), "stages").items()
    ):
        stage_id = str(stage_id)
        if stage_id in stages:
            raise ConfigError(f"duplicate stage id {stage_id!r}")
        stages[stage_id] = _parse_stage(stage_id, node, order, base_dir)
    if not stages:
        raise ConfigError("config declares no stages")


    for stage in stages.values():
        merged_environment = {**environment, **stage.environment}
        problem = check_device_environment_conflict(
            stage.device, merged_environment, where=f"stages.{stage.stage_id}"
        )
        if problem:
            raise ConfigError(problem)

    return PipelineConfig(
        schema_version=int(schema_version),
        path=config_path,
        base_dir=base_dir,
        dataset=dataset,
        runtime=runtime,
        credentials=credentials,
        stages=stages,
        environment=environment,
        raw=raw,
        endpoints=endpoints,
    )


def iter_secrets(config: PipelineConfig) -> Iterator[str]:

    yield from secrets_module.iter_config_secrets(
        credentials=config.credentials,
        params=(stage.params for stage in config.stages.values()),
        environment=config.environment,
    )
    for stage in config.stages.values():
        yield from secrets_module.iter_environment_secrets(stage.environment)
