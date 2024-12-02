import onnx
import spox

from kerox.core import SymbolicTensor


def create_model(
    inputs: dict[str, SymbolicTensor],
    outputs: dict[str, SymbolicTensor],
    drop_unused_inputs: bool = False,
    name: str = "model",
) -> onnx.ModelProto:
    spox_inputs = {name: input_.spox_var() for name, input_ in inputs.items()}
    spox_outputs = {name: output.spox_var() for name, output in outputs.items()}
    model_proto = spox.build(
        spox_inputs, spox_outputs, drop_unused_inputs=drop_unused_inputs
    )
    model_proto.producer_name = "kerox"

    return model_proto
