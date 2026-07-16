import onnx
import onnx.parser
import pytest

from onnx_cse._rewrite import rewrite, squeeze_unsqueeze

from .utils import count_nodes


def test_squeeze_unsqueeze_with_extra_node():
    # x's last dim is size 1, so Squeeze([-1]) then Unsqueeze([-1]) is a no-op.
    # `us` aliases `x`, Neg is rewired, and the two ops + Constant are swept.
    m = onnx.parser.parse_model("""
        <ir_version: 9, opset_import: ["": 21]>
        graph (float[N, 1] x) => (float[N, 1] out) {
            axes = Constant<value = int64[1] {-1}>()
            mid = Squeeze(x, axes)
            us = Unsqueeze(mid, axes)
            out = Neg(us)
        }
    """)
    rewrite(m, [squeeze_unsqueeze])
    onnx.checker.check_graph(m.graph)

    assert count_nodes(m.graph) == 1  # only Neg survives
    (neg,) = m.graph.node
    assert list(neg.input) == ["x"]


def test_squeeze_unsqueeze_leaves_graph_empty():
    # x's last dim is size 1, so Squeeze([-1]) then Unsqueeze([-1]) is a no-op.
    # `us` aliases `x`, Neg is rewired, and the two ops + Constant are swept.
    m = onnx.parser.parse_model("""
        <ir_version: 9, opset_import: ["": 21]>
        graph (float[N, 1] x) => (float[N, 1] out) {
            axes = Constant<value = int64[1] {-1}>()
            mid = Squeeze(x, axes)
            out = Unsqueeze(mid, axes)
        }
    """)
    rewrite(m, [squeeze_unsqueeze])
    onnx.checker.check_graph(m.graph)

    assert count_nodes(m.graph) == 1  # Only identity
    (ident,) = m.graph.node
    assert list(ident.input) == ["x"]


@pytest.mark.xfail(reason="Initializers are not taken into account yet.")
def test_squeeze_unsqueeze_axes_from_initializer():
    # Same no-op, but `axes` is a graph initializer rather than a Constant node
    # -- the shape exporters most commonly emit. The match must still fire.
    m = onnx.parser.parse_model("""
        <ir_version: 9, opset_import: ["": 21]>
        graph (float[N, 1] x) => (float[N, 1] out)
            <int64[1] axes = {-1}> {
            mid = Squeeze(x, axes)
            us = Unsqueeze(mid, axes)
            out = Neg(us)
        }
    """)
    rewrite(m, [squeeze_unsqueeze])
    onnx.checker.check_graph(m.graph)

    assert count_nodes(m.graph) == 1  # only Neg survives
    (neg,) = m.graph.node
    assert list(neg.input) == ["x"]


def test_bare_unsqueeze_does_not_match():
    # An Unsqueeze with no `axes` input must not crash the driver nor match.
    m = onnx.parser.parse_model("""
        <ir_version: 9, opset_import: ["": 21]>
        graph (float[N] x) => (float[N, 1] out) {
            out = Unsqueeze(x)
        }
    """)
    rewrite(m, [squeeze_unsqueeze])
    assert count_nodes(m.graph) == 1


def test_alias_flows_into_subgraph():
    # `sq` aliases `x` at the outer level and is captured by both If branches.
    # The alias must flow down so the branches reference `x`, not a dangling
    # `sq` whose producers get pruned.
    m = onnx.parser.parse_model("""
        <ir_version: 9, opset_import: ["": 21]>
        graph (float[N, 1] x, bool cond) => (float[N, 1] out) {
            axes = Constant<value = int64[1] {-1}>()
            mid = Squeeze(x, axes)
            sq = Unsqueeze(mid, axes)
            out = If (cond) <
                then_branch = g1 () => (float[N, 1] t) { t = Identity(sq) },
                else_branch = g2 () => (float[N, 1] e) { e = Neg(sq) }
            >
        }
    """)
    rewrite(m, [squeeze_unsqueeze])
    onnx.checker.check_model(m, True, True)

    # Squeeze/Unsqueeze/Constant gone; only If + its two branch nodes remain.
    assert count_nodes(m.graph) == 3
    (if_node,) = m.graph.node
    for attr in if_node.attribute:
        (body_node,) = attr.g.node
        assert list(body_node.input) == ["x"]


def test_pattern_spans_subgraph_boundary():
    # Squeeze is outer, Unsqueeze inside each If branch, all sharing `axes`.
    # The inner Unsqueeze sees the captured outer Squeeze, so each branch
    # output collapses to `Identity(x)` and the dead outer nodes are pruned.
    m = onnx.parser.parse_model("""
        <ir_version: 9, opset_import: ["": 21]>
        graph (float[N, 1] x, bool cond) => (float[N, 1] out) {
            axes = Constant<value = int64[1] {-1}>()
            mid = Squeeze(x, axes)
            out = If (cond) <
                then_branch = g1 () => (float[N, 1] t) { t = Unsqueeze(mid, axes) },
                else_branch = g2 () => (float[N, 1] e) { e = Unsqueeze(mid, axes) }
            >
        }
    """)
    rewrite(m, [squeeze_unsqueeze])
    onnx.checker.check_model(m, True, True)

    (if_node,) = m.graph.node  # dead outer Squeeze + Constant pruned
    for attr in if_node.attribute:
        (ident,) = attr.g.node
        assert (ident.op_type, list(ident.input)) == ("Identity", ["x"])


def test_captured_value_kept_when_graph_output():
    # `mid` is also an outer graph output, so the Squeeze must stay; the
    # subgraphs are still optimized to `Identity(x)` independently.
    m = onnx.parser.parse_model("""
        <ir_version: 9, opset_import: ["": 21]>
        graph (float[N, 1] x, bool cond) => (float[N, 1] out, float[N] mid) {
            axes = Constant<value = int64[1] {-1}>()
            mid = Squeeze(x, axes)
            out = If (cond) <
                then_branch = g1 () => (float[N, 1] t) { t = Unsqueeze(mid, axes) },
                else_branch = g2 () => (float[N, 1] e) { e = Unsqueeze(mid, axes) }
            >
        }
    """)
    rewrite(m, [squeeze_unsqueeze])
    onnx.checker.check_model(m, True, True)

    assert [n.op_type for n in m.graph.node] == ["Constant", "Squeeze", "If"]
    (if_node,) = [n for n in m.graph.node if n.op_type == "If"]
    for attr in if_node.attribute:
        (ident,) = attr.g.node
        assert (ident.op_type, list(ident.input)) == ("Identity", ["x"])


def test_rejects_old_opset():
    m = onnx.parser.parse_model("""
        <ir_version: 8, opset_import: ["": 12]>
        graph (float[N] x) => (float[N] out) {
            out = Neg(x)
        }
    """)
    with pytest.raises(ValueError, match="opset"):
        rewrite(m, [squeeze_unsqueeze])
