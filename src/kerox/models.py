import onnx
import spox

import kerox
from kerox.core import SymbolicTensor


def create_model(
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
