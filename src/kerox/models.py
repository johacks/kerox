import ndonnx
import networkx as nx
import onnx
import spox

import kerox
from kerox.core import EagerTensor, SymbolicTensor
from kerox.layer import Layer
from kerox.namespace import get_registered_object
from kerox.typing import ArrayLike, SupportsSpoxVar


class GraphModel(Layer):
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
        self._inline_model = spox.inline(self._model_proto)
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

    def forward(self, **inputs: dict[str, SupportsSpoxVar]) -> dict[str, spox.Var]:
        return self._inline_model(
            **{name: input_.spox_var() for name, input_ in inputs.items()}
        )

    def __call__(self, **inputs: dict[str, SupportsSpoxVar | ArrayLike]):
        if not inputs:
            raise ValueError("At least one input is required.")
        symbolic_inputs = [type(input_) is SymbolicTensor for input_ in inputs.values()]
        if not all(symbolic_inputs) and (is_symbolic := any(symbolic_inputs)):
            raise ValueError(
                "All inputs must be SymbolicTensor or ArrayLike, not a mix of both."
            )
        self._last_inputs = inputs

        # Called with SymbolicTensor: set source layer and return SymbolicTensor
        if is_symbolic:
            self._source_layers = [
                name
                for name in set(tensor.source_layer for tensor in inputs.values())
                if name is not None
            ]
            outputs = self.forward(**inputs)
            self._last_outputs = {
                name: SymbolicTensor(spox_var=output, source_layer=self._name)
                for name, output in outputs.items()
            }
            return self._last_outputs

        # Called with EagerTensor or convertable to ndonnx.Array: return EagerTensor
        # Ensure all have spox_var() method
        inputs = {
            name: array.value
            if isinstance(array, EagerTensor)
            else ndonnx.asarray(array)
            for name, array in inputs.items()
        }
        # Forward pass
        outputs = self.eager_forward_function(**inputs)

        # Wrap outputs in EagerTensor. Results order is guaranteed by sorted keys
        self._last_outputs = {
            name: EagerTensor(array, eager_source=self._last_inputs)
            for name, array in zip(sorted(self._outputs.keys()), outputs)
        }
        return self._last_outputs

    def __repr__(self) -> str:
        base = f"Model(inputs={self._inputs}, outputs={self._outputs})\n"
        base += "  Layers:" + "  \n    - ".join([""] + self.layers)
        return base


def build_layer_graph(outputs: dict[str, SymbolicTensor]) -> nx.DiGraph:
    edge_list = set()

    # Recursively add layer parents to the graph. We assume there are no cycles, as
    # they would be detected when building the model proto.
    def add_layer(layer_name: str) -> dict[str, tuple[str]]:
        layer: Layer = get_registered_object(layer_name)
        if layer is not None and layer.source_layers:
            for source_layer in layer.source_layers:
                edge_list.add((source_layer, layer_name))
                add_layer(source_layer)

    for output_name, output in outputs.items():
        # If the output has no source layer, it is a constant
        if output.source_layer is not None:
            edge_list.add((output.source_layer, output_name))
            add_layer(output.source_layer)

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
