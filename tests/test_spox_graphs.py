import numpy as np
import onnx
import onnxruntime as ort
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


def optional_inputs() -> tuple[onnx.ModelProto, int]:
    # ONNX Specs allow optional inputs denoted as `""`.
    # The input mustn't be ignored or else following inputs would lead
    # to an identical hash.
    x = op.const(42)
    res1 = op.clip(x, min=None, max=op.const(40))
    res2 = op.clip(x, min=op.const(40), max=None)

    # Const(42), Const(40), Clip, Clip, Identity, Identity
    return build({}, {"res1": res1, "res2": res2}), 6


@pytest.mark.parametrize(
    "construct", [redundant_if, redundant_constants, optional_inputs]
)
def test_models(construct):
    model, n_exp = construct()
    result_pre_opt = ort.InferenceSession(model.SerializeToString()).run(None, {})

    eliminate_common_subexpressions(model)
    onnx.checker.check_model(model, full_check=True)

    assert count_nodes(model.graph) == n_exp

    result_post_opt = ort.InferenceSession(model.SerializeToString()).run(None, {})
    for pre, post in zip(result_pre_opt, result_post_opt):
        np.testing.assert_array_equal(pre, post)
