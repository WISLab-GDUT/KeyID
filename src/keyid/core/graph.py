from __future__ import annotations

from dataclasses import dataclass, field
from typing import Mapping, Sequence

from .errors import GraphError


@dataclass(frozen=True)
class StageNode:
    stage_id: str
    type: str
    needs: tuple[str, ...]
    enabled: bool
    order: int


@dataclass
class ExecutionPlan:


    order: list[str]
    nodes: dict[str, StageNode]
    upstream: dict[str, tuple[str, ...]] = field(default_factory=dict)
    scope_reason: str = "full"

    def __iter__(self):
        return iter(self.order)

    def __len__(self) -> int:
        return len(self.order)

    def __contains__(self, stage_id: object) -> bool:
        return stage_id in self.order

    def describe(self) -> list[dict[str, object]]:
        return [
            {
                "stage_id": stage_id,
                "type": self.nodes[stage_id].type,
                "enabled": self.nodes[stage_id].enabled,
                "needs": list(self.nodes[stage_id].needs),
                "upstream_in_run": list(self.upstream.get(stage_id, ())),
            }
            for stage_id in self.order
        ]


def build_nodes(
    stages: Mapping[str, object],
    *,
    default_needs: Mapping[str, Sequence[str]] | None = None,
) -> dict[str, StageNode]:

    default_needs = default_needs or {}
    nodes: dict[str, StageNode] = {}

    for stage_id, stage_config in stages.items():
        stage_type = str(getattr(stage_config, "type", ""))
        if not stage_type:
            raise GraphError(f"stage {stage_id!r} has no type")
        needs = _effective_needs(stage_config, stage_type, default_needs)
        nodes[str(stage_id)] = StageNode(
            stage_id=str(stage_id),
            type=stage_type,
            needs=needs,
            enabled=bool(getattr(stage_config, "enabled", True)),
            order=int(getattr(stage_config, "order", 0)),
        )


    for node in nodes.values():
        for dependency in node.needs:
            if dependency not in nodes:
                raise GraphError(
                    f"stage {node.stage_id!r} needs {dependency!r}, which is not declared in "
                    f"`stages:` (known: {sorted(nodes)})"
                )
        if node.stage_id in node.needs:
            raise GraphError(f"stage {node.stage_id!r} depends on itself")

    detect_cycles(nodes)
    return nodes


def _effective_needs(
    stage_config: object,
    stage_type: str,
    default_needs: Mapping[str, Sequence[str]],
) -> tuple[str, ...]:

    declared = tuple(getattr(stage_config, "needs", ()) or ())
    explicit = getattr(stage_config, "needs_declared", None)
    if explicit is True:
        return declared
    if explicit is False:
        return tuple(default_needs.get(stage_type, ()))
    return declared or tuple(default_needs.get(stage_type, ()))


def detect_cycles(nodes: Mapping[str, StageNode]) -> None:

    WHITE, GREY, BLACK = 0, 1, 2
    colour: dict[str, int] = {stage_id: WHITE for stage_id in nodes}
    stack: list[str] = []

    def visit(stage_id: str) -> None:
        colour[stage_id] = GREY
        stack.append(stage_id)
        for dependency in nodes[stage_id].needs:
            if colour[dependency] == GREY:
                cycle = stack[stack.index(dependency) :] + [dependency]
                raise GraphError("cycle in stage DAG: " + " -> ".join(cycle))
            if colour[dependency] == WHITE:
                visit(dependency)
        stack.pop()
        colour[stage_id] = BLACK

    for stage_id in sorted(nodes):
        if colour[stage_id] == WHITE:
            visit(stage_id)


def topological_order(nodes: Mapping[str, StageNode]) -> list[str]:

    incoming = {stage_id: set(node.needs) for stage_id, node in nodes.items()}
    remaining = dict(incoming)

    def sort_key(stage_id: str) -> tuple[int, str]:
        return (nodes[stage_id].order, stage_id)

    order: list[str] = []
    ready = sorted((sid for sid, deps in remaining.items() if not deps), key=sort_key)
    while ready:
        stage_id = ready.pop(0)
        order.append(stage_id)
        remaining.pop(stage_id, None)
        newly_ready = []
        for other_id, deps in remaining.items():
            if stage_id in deps:
                deps.discard(stage_id)
                if not deps:
                    newly_ready.append(other_id)
        if newly_ready:
            ready = sorted(ready + newly_ready, key=sort_key)

    if len(order) != len(nodes):
        unresolved = sorted(set(nodes) - set(order))
        raise GraphError(f"stage DAG could not be ordered; unresolved: {unresolved}")
    return order


def transitive_upstream(nodes: Mapping[str, StageNode], stage_id: str) -> list[str]:

    seen: list[str] = []
    frontier = list(nodes[stage_id].needs)
    while frontier:
        current = frontier.pop(0)
        if current in seen:
            continue
        seen.append(current)
        frontier.extend(nodes[current].needs)
    return seen


def resolve_scope(
    nodes: Mapping[str, StageNode],
    *,
    target: str | None = None,
    only: str | None = None,
    include_disabled: bool = False,
) -> ExecutionPlan:

    if target and only:
        raise GraphError("--target and --only are mutually exclusive")
    if target and target not in nodes:
        raise GraphError(f"unknown stage for --target: {target!r} (known: {sorted(nodes)})")
    if only and only not in nodes:
        raise GraphError(f"unknown stage for --only: {only!r} (known: {sorted(nodes)})")

    full_order = topological_order(nodes)

    if only:
        selected = [only]
        reason = f"only:{only}"
    elif target:
        wanted = {target, *transitive_upstream(nodes, target)}
        selected = [stage_id for stage_id in full_order if stage_id in wanted]
        reason = f"target:{target}"
    else:
        selected = list(full_order)
        reason = "full"

    if not include_disabled:


        selected = [sid for sid in selected if nodes[sid].enabled or sid == only]


    upstream = {
        stage_id: tuple(dep for dep in nodes[stage_id].needs) for stage_id in selected
    }
    return ExecutionPlan(order=selected, nodes=dict(nodes), upstream=upstream, scope_reason=reason)


def build_plan(
    stages: Mapping[str, object],
    *,
    default_needs: Mapping[str, Sequence[str]] | None = None,
    target: str | None = None,
    only: str | None = None,
) -> ExecutionPlan:
    nodes = build_nodes(stages, default_needs=default_needs)
    return resolve_scope(nodes, target=target, only=only)
