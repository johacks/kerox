import networkx as nx
import onnx
import spox

import kerox
from kerox.core import SymbolicTensor
from kerox.layer import Layer
from kerox.namespace import get_registered_object


class Model(Layer):
    def __init__(
        self,
        inputs: dict[str, SymbolicTensor] = None,
        outputs: dict[str, SymbolicTensor] = None,
        name: str = None,
        **kwargs,
    ):
        super().__init__(name)
        self._inputs = inputs or {}
        self._outputs = outputs or {}
        self._model_proto = create_model_proto(self._inputs, self._outputs, **kwargs)
        self._layer_graph = build_layer_graph(self._outputs)
        self._layers = [
            node
            for node in nx.topological_sort(self._layer_graph)
            if isinstance(get_registered_object(node), Layer)
        ]

    @property
    def layers(self) -> list[str]:
        return self._layers

    @property
    def model_proto(self) -> onnx.ModelProto:
        return self._model_proto

    @property
    def layer_graph(self) -> nx.DiGraph:
        return self._layer_graph

    def __repr__(self) -> str:
        base = f"Model(inputs={self._inputs}, outputs={self._outputs})\n"
        base += "  Layers:" + "  \n    - ".join([""] + self.layers)
        return base


def build_layer_graph(outputs: dict[str, SymbolicTensor]):
    edge_list = set()

    # Recursively add layer parents to the graph. We assume there are no cycles, as
    # they would be detected when building the model proto.
    def add_layer(layer_name: str) -> dict[str, tuple[str]]:
        layer: Layer = get_registered_object(layer_name)
        if layer.source_layers is None:
            return
        for source_layer in layer.source_layers:
            edge_list.add((source_layer, layer_name))
            add_layer(source_layer)

    for output_name, output in outputs.items():
        # If the output has no source layer, it is a constant
        if output.source_layer is not None:
            edge_list.add((output.source_layer, output_name))

    return nx.from_edgelist(edge_list, create_using=nx.DiGraph)


def create_model_proto(
    inputs: dict[str, SymbolicTensor],
    outputs: dict[str, SymbolicTensor],
    drop_unused_inputs: bool = False,
    doc_string: str = "",
    model_version: int = 1,
    domain: str = "",
) -> onnx.ModelProto:
    spox_inputs = {name: input_.spox_var() for name, input_ in inputs.items()}
    spox_outputs = {name: output.spox_var() for name, output in outputs.items()}
    model_proto = spox.build(
        spox_inputs, spox_outputs, drop_unused_inputs=drop_unused_inputs
    )
    model_proto.producer_name = "kerox"
    model_proto.producer_version = kerox.version()
    model_proto.doc_string = doc_string
    model_proto.model_version = model_version
    model_proto.domain = domain
    return model_proto
