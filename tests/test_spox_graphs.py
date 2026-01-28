import onnx
import pytest
import spox.opset.ai.onnx.v22 as op
from spox import build

from onnx_cse import eliminate_common_subexpressions

from .utils import count_nodes


def redundant_if() -> tuple[onnx.ModelProto, int]:
    cond = op.const(True)
    const = op.const(1)
    outs = {
        f"_{i}": op.if_(
            cond,
            then_branch=lambda: [const],
            else_branch=lambda: [const],
        )[0]
        for i in range(2)
    }

    # Cond, Const, If(Identity, Identity), Identity, Identity
    return build({}, outs), 7


def redundant_constants() -> tuple[onnx.ModelProto, int]:
    a = op.const(1)
    b = op.const(1)

    # Const, Add, Identity
    return build({}, {"out": op.add(a, b)}), 3


@pytest.mark.parametrize("construct", [redundant_if, redundant_constants])
def test_models(construct):
    model, n_exp = construct()
    eliminate_common_subexpressions(model)
    onnx.checker.check_model(model, full_check=True)

    assert count_nodes(model.graph) == n_exp
