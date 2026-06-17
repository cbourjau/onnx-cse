import onnx

from onnx_cse._cse import Scope, cse

from .utils import count_nodes


def redundant_add() -> tuple[onnx.ModelProto, int]:
    # Model with a redundant Add node. The second node's outputs are also outputs of the graph.
    inputs = ["a", "b"]
    n1 = onnx.helper.make_node(
        op_type="Add",
        inputs=inputs,
        outputs=["out1"],
        domain="",
    )
    n2 = onnx.helper.make_node(
        op_type="Add",
        inputs=inputs,
        outputs=["out2"],
        domain="",
    )
    in_infos = [
        onnx.helper.make_tensor_value_info(name, elem_type=11, shape=("N",))
        for name in inputs
    ]
    out_infos = [
        onnx.helper.make_tensor_value_info(name, elem_type=11, shape=("N",))
        for name in ["out1", "out2"]
    ]
    g = onnx.helper.make_graph(
        nodes=[n1, n2], name="graph", inputs=in_infos, outputs=out_infos
    )
    m = onnx.helper.make_model(g)
    m = onnx.shape_inference.infer_shapes(m, True, True, True)
    onnx.checker.check_model(m, True, True)

    # Should reduce to 1 Add and 1 Identity
    return m, 2


def redundant_add_in_subgraphs():
    add_inputs = ["a", "b"]
    inner_add = onnx.helper.make_node(
        op_type="Add",
        inputs=add_inputs,
        outputs=["inner_add"],
        domain="",
    )
    sub_out_infos = [
        onnx.helper.make_tensor_value_info(name, elem_type=11, shape=("N",))
        for name in ["inner_add"]
    ]
    sub_g = onnx.helper.make_graph(
        nodes=[inner_add], name="graph", inputs=[], outputs=sub_out_infos
    )

    if_node = onnx.helper.make_node(
        "If",
        domain="",
        inputs=["cond"],
        outputs=["if_out"],
        then_branch=sub_g,
        else_branch=sub_g,
    )

    outer_add = onnx.helper.make_node(
        op_type="Add",
        inputs=add_inputs,
        outputs=["outer_add"],
        domain="",
    )
    in_infos = [
        onnx.helper.make_tensor_value_info(name, elem_type=11, shape=("N",))
        for name in add_inputs
    ] + [
        onnx.helper.make_tensor_value_info(
            "cond", elem_type=onnx.TensorProto.BOOL, shape=()
        )
    ]
    out_infos = [
        onnx.helper.make_tensor_value_info("outer_add", elem_type=11, shape=("N",)),
        onnx.helper.make_tensor_value_info("if_out", elem_type=11, shape=("N",)),
    ]

    main_g = onnx.helper.make_graph(
        nodes=[outer_add, if_node], name="graph", inputs=in_infos, outputs=out_infos
    )
    m = onnx.helper.make_model(main_g)
    m = onnx.shape_inference.infer_shapes(m, True, True, True)
    onnx.checker.check_model(m, True, True)

    # Add, If, 2x Identity
    return m, 4


def graph_with_initializers_as_constants():
    string_tensor = onnx.helper.make_tensor(
        name="strings",
        data_type=onnx.TensorProto.STRING,
        dims=[2],
        vals=[b"1", b"2"],  # type: ignore
    )

    # Cast node: string_input → int_values (INT64)
    cast_node = onnx.helper.make_node(
        "Cast",
        inputs=["strings"],
        outputs=["int_values"],
        to=onnx.TensorProto.INT64,
    )

    int_values_info = onnx.helper.make_tensor_value_info(
        "int_values", onnx.TensorProto.INT64, [8]
    )

    graph = onnx.helper.make_graph(
        nodes=[cast_node],
        name="string_cast_graph",
        inputs=[],
        outputs=[int_values_info],
        initializer=[string_tensor],
    )

    model = onnx.helper.make_model(
        graph,
        opset_imports=[onnx.helper.make_opsetid("", 21)],
    )

    onnx.checker.check_model(model)

    return model, 1


def test_redundant_add():
    model, n_exp = redundant_add()
    g, _ = cse(model.graph, Scope([el.name for el in model.graph.output]))
    onnx.checker.check_graph(g)

    assert count_nodes(model.graph) == n_exp


def test_redundant_add_in_subgraphs():
    model, n_exp = redundant_add_in_subgraphs()
    g, _ = cse(model.graph, Scope([el.name for el in model.graph.output]))
    onnx.checker.check_graph(g)

    assert count_nodes(model.graph) == n_exp


def test_initializers_as_constants():
    model, n_exp = graph_with_initializers_as_constants()
    g, _ = cse(model.graph, Scope([el.name for el in model.graph.output]))
    onnx.checker.check_graph(g)

    assert count_nodes(model.graph) == n_exp
