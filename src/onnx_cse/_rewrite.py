"""A minimal pattern-rewrite driver.

A *rewrite* is a function that looks at one node, and -- if it recognizes a redundant
construct -- reports that some of the node's outputs equal an existing value. The
*driver* applies a list of rewrites to a fixpoint, rewires consumers onto the surviving
values, and drops what died.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable, Iterator, Mapping
from dataclasses import dataclass

import onnx
from onnx import AttributeProto

VarName = str
NormalizedName = str

# Optional inputs/outputs are denoted with an empty string.
_OPTIONAL_NAME = ""

# output-name -> equivalent existing value. Returned by a matching rewrite;
# ``None`` when the rewrite does not apply to this node.
Match = Mapping[VarName, NormalizedName] | None
Rewrite = Callable[[onnx.NodeProto, "Context"], Match]
# A node is identified by the identity of its proto object, which is stable for
# the lifetime of a pass (the objects are pinned by ``producers``).
NodeId = int

# Below this opset the Squeeze/Unsqueeze ``axes`` is an attribute rather than
# a second input, so the matchers here do not apply.
_MIN_OPSET = 13


@dataclass
class Context:
    """What a rewrite sees: node producers, plus the alias map the driver
    accumulates as rewrites fire."""

    producers: dict[VarName, onnx.NodeProto]
    aliases: dict[VarName, NormalizedName]

    depends_on: dict[VarName, set[VarName]]
    """Direct dependencies needed by the node that computes `VarName` (incl captured
    ones)."""

    @classmethod
    def main(cls, g: onnx.GraphProto):
        return cls(
            producers={outp: n for n in g.node for outp in n.output},
            aliases={},
            depends_on={},
        )

    def resolve(self, name: VarName) -> NormalizedName:
        """The canonical value ``name`` stands for.

        One hop: targets are
        canonicalized on insert, and graphs are topologically sorted so an
        alias always points at an already-canonical upstream value.
        """
        return self.aliases.get(name, name)

    def producer(self, name: VarName) -> onnx.NodeProto | None:
        """The node producing ``name`` (resolved through aliases), if any."""
        return self.producers.get(self.resolve(name))

    def producer_id(self, name: VarName) -> NodeId | None:
        """The node-id producing ``name`` (resolved through aliases), if any."""
        if node := self.producers.get(self.resolve(name)):
            return id(node)
        return None

    def branch(self, g: onnx.GraphProto) -> Context:
        produced_in_this_graph = {outp: n for n in g.node for outp in n.output}
        return Context(
            self.producers | produced_in_this_graph,
            self.aliases.copy(),
            self.depends_on.copy(),
        )

    def live_nodes(self, var_names: Iterable[VarName]) -> set[NodeId]:
        """Set of nodes that required to compute the `var_names` (after name-
        normalization)."""
        live = set()
        visited = set()
        stack = list(var_names)
        while stack:
            name = stack.pop(-1)
            if name in visited:
                continue
            visited.add(name)
            if node_id := self.producer_id(name):
                live.add(node_id)

            stack.extend(self.depends_on.get(name, []))

        return live


def rewrite(model: onnx.ModelProto, rewrites: list[Rewrite]) -> None:
    """Apply ``rewrites`` to ``model`` in place after checking the opset."""
    opset = _default_opset(model)
    # `axes` was converted from an attribute to an input of Squeeze/Unsqueeze in opset 13
    if opset < _MIN_OPSET:
        raise ValueError(f"unsupported default opset {opset}; requires >= {_MIN_OPSET}")

    changed = True
    while changed:
        changed, _ = _process_graph_recursive(
            model.graph, rewrites, Context.main(model.graph)
        )


def _default_opset(model: onnx.ModelProto) -> int:
    """The version imported for the default (ai.onnx) domain."""
    for imp in model.opset_import:
        if imp.domain in ("", "ai.onnx"):
            return imp.version
    raise ValueError("model does not import the default (ai.onnx) domain")


def _process_graph_recursive(
    g: onnx.GraphProto, rewrites: list[Rewrite], ctx: Context
) -> tuple[bool, set[VarName]]:
    produced_in_this_scope = set(outp for n in g.node for outp in n.output)
    changed = False
    for n in g.node:
        changed_, node_deps = _process_node_recursive(n, rewrites, ctx)
        changed |= changed_
        for outp in n.output:
            # ``node_deps`` already includes the node's (resolved) direct inputs.
            ctx.depends_on[outp] = node_deps

    live_nodes = ctx.live_nodes(el.name for el in g.output)
    live_deps: set[VarName] = set()
    filtered = []

    for n in g.node:
        if id(n) in live_nodes:
            filtered.append(n)
            # Full dependency set (direct inputs + subgraph captures), so a value
            # captured inside a live node's subgraph is propagated upward too.
            for outp in n.output:
                live_deps |= ctx.depends_on.get(outp, set())

    # Check if outputs have been aliased and if so add an Identity node
    identities = []
    for outp in g.output:
        if outp.name in ctx.aliases:
            # Output was canonicalized and now needs an identity node. Its
            # (resolved) input must stay alive in the enclosing scope.
            resolved = ctx.resolve(outp.name)
            identities.append(
                onnx.helper.make_node(
                    "Identity", inputs=[resolved], outputs=[outp.name]
                )
            )
            live_deps.add(resolved)

    g.ClearField("node")
    g.node.extend(filtered + identities)
    # Only names not produced in this scope are captured from an enclosing one.
    return changed, live_deps - produced_in_this_scope


def _process_node_recursive(
    n: onnx.NodeProto, rewrites: list[Rewrite], ctx: Context
) -> tuple[bool, set[VarName]]:
    # Process subgraphs first so that the node rewrite can already use the updated ones.
    _update_inputs(n, ctx)
    node_dependencies = set(n.input)
    changed = False
    for subg in _subgraphs(n):
        changed_, subgraph_deps = _process_graph_recursive(
            subg, rewrites, ctx.branch(subg)
        )

        changed |= changed_
        node_dependencies.update(subgraph_deps)

    for rw in rewrites:
        match_ = rw(n, ctx) or {}
        for out, value in match_.items():
            if out not in ctx.aliases:
                # Canonicalize
                ctx.aliases[out] = ctx.aliases.get(value, value)
                changed = True

    return changed, node_dependencies


def _update_inputs(n: onnx.NodeProto, ctx: Context):
    new_inputs = [ctx.resolve(i) for i in n.input]
    n.ClearField("input")
    n.input.extend(new_inputs)


def _subgraphs(n: onnx.NodeProto) -> Iterator[onnx.GraphProto]:
    for attr in n.attribute:
        if attr.type == AttributeProto.GRAPH:
            yield attr.g
        elif attr.type == AttributeProto.GRAPHS:
            yield from attr.graphs


def squeeze_unsqueeze(n: onnx.NodeProto, ctx: Context) -> Match:
    """``unsqueeze(squeeze(x, [-1]), [-1]) -> x``."""
    if not _is_op_in_default_domain(n, "Unsqueeze") or len(n.input) < 2:
        return None
    if not _axes_is_neg1(n.input[1], ctx):
        return None
    src = ctx.producer(n.input[0])
    if (
        src is None
        or not _is_op_in_default_domain(src, "Squeeze")
        or len(src.input) < 2
    ):
        return None
    if not _axes_is_neg1(src.input[1], ctx):
        return None
    return {n.output[0]: src.input[0]}


def _is_op_in_default_domain(n: onnx.NodeProto, op_type: str) -> bool:
    return n.op_type == op_type and _is_default_domain(n)


def _axes_is_neg1(name: VarName, ctx: Context) -> bool:
    return _is_const_neg1(ctx.producer(name))


def _is_const_neg1(n: onnx.NodeProto | None) -> bool:
    """True if `n` is a `Constant`-node with value `[-1]`."""
    if n is None:
        return False
    if not (n.op_type == "Constant" and _is_default_domain(n)):
        return False

    for attr in n.attribute:
        if attr.name == "value" and attr.type == AttributeProto.TENSOR:
            return _is_tensor_neg1(attr.t)
        if attr.name == "value_ints" and attr.type == AttributeProto.INTS:
            return list(attr.ints) == [-1]

    return False


def _is_tensor_neg1(t: onnx.TensorProto) -> bool:
    """True if tensor `t` holds the single value `[-1]`."""
    arr = onnx.numpy_helper.to_array(t)
    return arr.shape == (1,) and arr.tolist() == [-1]


def _is_default_domain(n: onnx.NodeProto) -> bool:
    return n.domain in ("", "ai.onnx")
