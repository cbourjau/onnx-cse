import onnx


def count_nodes(g: onnx.GraphProto) -> int:
    out = 0
    for n in g.node:
        for attr in n.attribute:
            if attr.type == onnx.AttributeProto.GRAPH:
                out += count_nodes(attr.g)
            if attr.type == onnx.AttributeProto.GRAPHS:
                raise NotImplementedError
        out += 1

    return out
