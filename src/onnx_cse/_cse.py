from __future__ import annotations

from collections.abc import Sequence
from copy import copy

import onnx
from onnx import AttributeProto
from typing_extensions import Self
from xxhash import xxh3_128

__all__ = ["eliminate_common_subexpressions"]


def eliminate_common_subexpressions(model: onnx.ModelProto) -> None:
    """Eliminate common subexpressions inplace."""
    scope = Scope([el.name for el in model.graph.output])
    g, _ = cse(model.graph, scope)
    model.graph.CopyFrom(g)


def cse(g: onnx.GraphProto, scope: Scope) -> tuple[onnx.GraphProto, Scope]:
    # add inputs to scope
    for el in g.input:
        assert el.name not in scope.name_hash
        input_digest = xxh3_128(el.name).digest()
        scope.name_hash[el.name] = input_digest
        scope.hash_norm_name[input_digest] = el.name

    # Process nodes
    filtered_nodes = []
    for n in g.node:
        filtered_nodes.extend(scope.process(n))

    # Update protobuf objects
    g = copy(g)
    g.ClearField("node")
    g.node.extend(filtered_nodes)

    # TODO: Update ValueInfo
    # g = scope.update_value_infos(g)

    return g, scope


class Scope:
    name_hash: dict[str, bytes]
    hash_norm_name: dict[bytes, str]

    outputs: tuple[str, ...]
    """Reserved output names."""

    def __init__(self, outputs: Sequence[str]):
        self.name_hash = {}
        self.hash_norm_name = {}
        self.outputs = tuple(outputs)

    def process(self, n: onnx.NodeProto) -> list[onnx.NodeProto]:
        """Add outputs of this node to the scope if they are unique.

        Returns a node with updated input and output names if it
        produced novel output; `None` otherwise.
        """
        # inputs
        h = hash_node_inputs_and_attrs(n, self)

        # Add outputs and their hashes to the scope object
        has_unique_output = False
        for i, name in enumerate(n.output):
            if name == "":
                # Don't bother with unused outputs for now
                # TODO: there is an edge case for optimization where
                # we have a subsequent subexpression that would use
                # this output and is thus not eliminated.
                continue
            h_copy = h.copy()
            # We care about the output slot rather than its name
            h_copy.update(str(i))

            assert name not in self.name_hash  # sanity check
            byte_hash = h_copy.digest()
            self.name_hash[name] = byte_hash
            if byte_hash not in self.hash_norm_name:
                has_unique_output = True
                self.hash_norm_name[byte_hash] = name

        if not has_unique_output:
            # We eliminate this node, but might have to create new
            # identity nodes if some of the node outputs are graph
            # outputs.
            outs = []
            for name in n.output:
                if name in self.outputs:
                    outs.append(
                        onnx.helper.make_node(
                            "Identity",
                            inputs=[self.hash_norm_name[self.name_hash[name]]],
                            outputs=[name],
                        )
                    )
            return outs

        # We may have to update the input names of the node
        updated_names = []
        for name in n.input:
            if name == "":
                updated_names.append("")
                continue
            updated_names.append(self.hash_norm_name[self.name_hash[name]])

        out = copy(n)
        out.ClearField("input")
        out.input.extend(updated_names)
        return [out]

    def update_value_infos(self, infos: onnx.GraphProto) -> onnx.GraphProto:
        raise NotImplementedError

    def subscope(self, g: onnx.GraphProto) -> Self:
        new = type(self)([el.name for el in g.output])
        new.hash_norm_name = self.hash_norm_name.copy()
        new.name_hash = self.name_hash.copy()
        return new


def attr_value_to_bytes(attr: onnx.AttributeProto) -> bytes:
    supported_attrs = {
        AttributeProto.FLOAT,
        AttributeProto.INT,
        AttributeProto.FLOATS,
        AttributeProto.INTS,
        AttributeProto.TENSOR,
        AttributeProto.TYPE_PROTO,
        AttributeProto.TENSORS,
        AttributeProto.TYPE_PROTOS,
        AttributeProto.STRING,
        AttributeProto.STRINGS,
    }
    if attr.type in supported_attrs:
        return attr.SerializeToString()
    raise NotImplementedError


def hash_node_inputs_and_attrs(n: onnx.NodeProto, scope: Scope) -> xxh3_128:
    """Hash the nodes input names, op_type, domain, and metadata prop, and attributes.

    **Subexpressions are updated in place**.
    """
    h = xxh3_128()
    for name in n.input:
        h.update(scope.name_hash[name])
    # node properties
    h.update(n.op_type)
    h.update(n.domain)
    for item in n.metadata_props:
        h.update(item.key)
        h.update(item.value)

    for attr in n.attribute:
        if attr.type == onnx.AttributeProto.GRAPH:
            sub_g, sub_scope = cse(attr.g, scope.subscope(attr.g))
            h.update(attr.name)
            for item in sub_g.output:
                h.update(sub_scope.name_hash[item.name])

            # update subexpression
            attr.g.CopyFrom(sub_g)
        else:
            h.update(attr_value_to_bytes(attr))

    return h
