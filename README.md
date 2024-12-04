# Kerox: Keras 3 + Spox for easy ONNX export

This repository demonstrates how to integrate [Keras 3](https://github.com/keras-team/keras) + [Spox](https://github.com/Quantco/spox).

It adjusts keras symbolic tensors to be compatible with Spox variables and translates some of Keras ops to Spox ops. Currently just a POC implementing layers:

- `Dense`
- `Dropout`

## Demonstration

### Functional API

```python
import onnx
import spox
from kerox import layers, models, KeroxInput
from keras import datasets


(x_train, y_train), (x_test, y_test) = datasets.boston_housing.load_data()
NUM_FEATURES = x_train.shape[1]

inputs = KeroxInput(shape=(NUM_FEATURES,), dtype="float32")

x = layers.Dense(4, activation="relu")(inputs)
x = layers.Dropout(0.5)(x)
x = layers.Dense(2, activation="relu")(x)
outputs = layers.Dense(1)(x)
model = models.KeroxModel(inputs=inputs, outputs=outputs)
model.summary()
model.compile(optimizer="adam", loss="mse")
model.fit(x_train, y_train, epochs=10, batch_size=32)


inference_outputs = model.onnx_symbolic_call(inputs, training=False)
inference_model: onnx.ModelProto = spox.build(
    inputs={"input": inputs.spox_var()},
    outputs={"output": inference_outputs.spox_var()},
)
print(inference_model)
```

# Sequential API

```python
...
sequential_model = models.KeroxSequential(
    [
        layers.InputLayer(shape=(NUM_FEATURES,), dtype="float32"),
        layers.Dense(4, activation="relu"),
        layers.Dropout(0.5),
        layers.Dense(2, activation="relu"),
        layers.Dense(1),
    ]
)
model.summary()
model.compile(optimizer="adam", loss="mse")
model.fit(x_train, y_train, epochs=10, batch_size=32)
inference_outputs = model.onnx_symbolic_call(inputs, training=False)
inference_model: onnx.ModelProto = spox.build(
    inputs={"input": inputs.spox_var()},
    outputs={"output": inference_outputs.spox_var()},
)
print(inference_model)
```

### Custom Model API

```python
...
class CustomModel(models.KeroxModel):
    def __init__(self):
        super().__init__()
        self.dense_1 = layers.Dense(4, activation="relu")
        self.dropout = layers.Dropout(0.5)
        self.dense_2 = layers.Dense(2, activation="relu")
        self.dense_3 = layers.Dense(1)

    def call(self, inputs):
        x = self.dense_1(inputs)
        x = self.dropout(x)
        x = self.dense_2(x)
        return self.dense_3(x)


model = CustomModel()
model.compile(optimizer="adam", loss="mse")
model.fit(x_train, y_train, epochs=10, batch_size=32)
inference_outputs = model.onnx_symbolic_call(inputs, training=False)
inference_model: onnx.ModelProto = spox.build(
    inputs={"input": inputs.spox_var()},
    outputs={"output": inference_outputs.spox_var()},
)
print(inference_model)
```

## ONNX outputs (print of `inference_model`)

### Functional API

```protobuf
ir_version: 8
producer_name: "spox"
doc_string: ""
graph {
  node {
    input: "input"
    input: "dense/kernel"
    output: "MatMul_0_Y"
    name: "MatMul_0"
    op_type: "MatMul"
    domain: ""
  }
  node {
    input: "MatMul_0_Y"
    input: "dense/bias"
    output: "Add_0_C"
    name: "Add_0"
    op_type: "Add"
    domain: ""
  }
  node {
    input: "Add_0_C"
    output: "Relu_0_Y"
    name: "Relu_0"
    op_type: "Relu"
    domain: ""
  }
  node {
    input: "Relu_0_Y"
    input: "dense_1/kernel"
    output: "MatMul_1_Y"
    name: "MatMul_1"
    op_type: "MatMul"
    domain: ""
  }
  node {
    input: "MatMul_1_Y"
    input: "dense_1/bias"
    output: "Add_1_C"
    name: "Add_1"
    op_type: "Add"
    domain: ""
  }
  node {
    input: "Add_1_C"
    output: "Relu_1_Y"
    name: "Relu_1"
    op_type: "Relu"
    domain: ""
  }
  node {
    input: "Relu_1_Y"
    input: "dense_2/kernel"
    output: "MatMul_2_Y"
    name: "MatMul_2"
    op_type: "MatMul"
    domain: ""
  }
  node {
    input: "MatMul_2_Y"
    input: "dense_2/bias"
    output: "Add_2_C"
    name: "Add_2"
    op_type: "Add"
    domain: ""
  }
  node {
    input: "Add_2_C"
    output: "Identity_0_output"
    name: "Identity_0"
    op_type: "Identity"
    domain: ""
  }
  node {
    input: "Identity_0_output"
    output: "output"
    name: "Introduce_0_id0"
    op_type: "Identity"
  }
  name: "spox_graph"
  initializer {
    dims: 13
    dims: 4
    data_type: 1
    float_data: 0.32057929
    float_data: -0.393380046
    float_data: 0.223172307
    float_data: -0.639093
    float_data: -0.394830257
    float_data: -0.236766204
    float_data: -0.191264808
    float_data: -0.53608036
    float_data: 0.607675791
    float_data: 0.100761041
    float_data: -0.14552173
    float_data: -0.0843543932
    float_data: -0.183419675
    float_data: -0.55561173
    float_data: 0.0369989276
    float_data: 0.0270545818
    float_data: 0.428577781
    float_data: 0.238214388
    float_data: -0.0623987317
    float_data: -0.00605012709
    float_data: 0.175068
    float_data: 0.0386312231
    float_data: -0.157852232
    float_data: -0.151180953
    float_data: -0.392128617
    float_data: 0.179459527
    float_data: 0.153800964
    float_data: -0.005531976
    float_data: -0.285905898
    float_data: -0.281069905
    float_data: -0.155519098
    float_data: -0.480955541
    float_data: 0.0633017346
    float_data: -0.251203924
    float_data: 0.365984738
    float_data: -0.581542253
    float_data: -0.239007041
    float_data: 0.200860649
    float_data: -0.556500673
    float_data: 0.39968279
    float_data: 0.159244075
    float_data: 0.429733932
    float_data: 0.149005711
    float_data: -0.212655
    float_data: 0.405231327
    float_data: -0.0378316715
    float_data: 0.133305013
    float_data: 0.20684813
    float_data: 0.419427812
    float_data: 0.389761478
    float_data: 0.372409046
    float_data: -0.0683852509
    name: "dense/kernel"
  }
  initializer {
    dims: 4
    data_type: 1
    float_data: 0.0636389926
    float_data: 0.0443214849
    float_data: 0
    float_data: -0.0868284628
    name: "dense/bias"
  }
  initializer {
    dims: 4
    dims: 2
    data_type: 1
    float_data: -0.109976307
    float_data: -0.0700701475
    float_data: -0.7950809
    float_data: 0.202961
    float_data: -0.740133286
    float_data: -0.52476716
    float_data: 0.209337384
    float_data: -0.743718147
    name: "dense_1/kernel"
  }
  initializer {
    dims: 2
    data_type: 1
    float_data: -0.0938416198
    float_data: -0.0827707797
    name: "dense_1/bias"
  }
  initializer {
    dims: 2
    dims: 1
    data_type: 1
    float_data: -0.435174137
    float_data: 0.786228597
    name: "dense_2/kernel"
  }
  initializer {
    dims: 1
    data_type: 1
    float_data: 0.109085612
    name: "dense_2/bias"
  }
  input {
    name: "input"
    type {
      tensor_type {
        elem_type: 1
        shape {
          dim {
          }
          dim {
            dim_value: 13
          }
        }
      }
    }
  }
  output {
    name: "output"
    type {
      tensor_type {
        elem_type: 1
        shape {
          dim {
          }
          dim {
            dim_value: 1
          }
        }
      }
    }
  }
}
opset_import {
  domain: ""
  version: 21
}
```

### Sequential API

```protobuf
ir_version: 8
producer_name: "spox"
doc_string: ""
graph {
  node {
    input: "input"
    input: "dense/kernel"
    output: "MatMul_0_Y"
    name: "MatMul_0"
    op_type: "MatMul"
    domain: ""
  }
  node {
    input: "MatMul_0_Y"
    input: "dense/bias"
    output: "Add_0_C"
    name: "Add_0"
    op_type: "Add"
    domain: ""
  }
  node {
    input: "Add_0_C"
    output: "Relu_0_Y"
    name: "Relu_0"
    op_type: "Relu"
    domain: ""
  }
  node {
    input: "Relu_0_Y"
    input: "dense_1/kernel"
    output: "MatMul_1_Y"
    name: "MatMul_1"
    op_type: "MatMul"
    domain: ""
  }
  node {
    input: "MatMul_1_Y"
    input: "dense_1/bias"
    output: "Add_1_C"
    name: "Add_1"
    op_type: "Add"
    domain: ""
  }
  node {
    input: "Add_1_C"
    output: "Relu_1_Y"
    name: "Relu_1"
    op_type: "Relu"
    domain: ""
  }
  node {
    input: "Relu_1_Y"
    input: "dense_2/kernel"
    output: "MatMul_2_Y"
    name: "MatMul_2"
    op_type: "MatMul"
    domain: ""
  }
  node {
    input: "MatMul_2_Y"
    input: "dense_2/bias"
    output: "Add_2_C"
    name: "Add_2"
    op_type: "Add"
    domain: ""
  }
  node {
    input: "Add_2_C"
    output: "Identity_0_output"
    name: "Identity_0"
    op_type: "Identity"
    domain: ""
  }
  node {
    input: "Identity_0_output"
    output: "output"
    name: "Introduce_0_id0"
    op_type: "Identity"
  }
  name: "spox_graph"
  initializer {
    dims: 13
    dims: 4
    data_type: 1
    float_data: 0.384926856
    float_data: -0.455883324
    float_data: 0.223172307
    float_data: -0.711012661
    float_data: -0.348224878
    float_data: -0.174767435
    float_data: -0.191264808
    float_data: -0.600477219
    float_data: 0.6985659
    float_data: 0.0411026031
    float_data: -0.14552173
    float_data: -0.16845347
    float_data: -0.147415459
    float_data: -0.537514806
    float_data: 0.0369989276
    float_data: -0.019154327
    float_data: 0.519494414
    float_data: 0.227582783
    float_data: -0.0623987317
    float_data: -0.0899472684
    float_data: 0.260645479
    float_data: 0.075758405
    float_data: -0.157852232
    float_data: -0.233428285
    float_data: -0.301349699
    float_data: 0.144779131
    float_data: 0.153800964
    float_data: -0.0893597379
    float_data: -0.208175838
    float_data: -0.213603571
    float_data: -0.155519098
    float_data: -0.559892535
    float_data: 0.153604478
    float_data: -0.315029681
    float_data: 0.365984738
    float_data: -0.663761556
    float_data: -0.146884516
    float_data: 0.157279521
    float_data: -0.556500673
    float_data: 0.315637082
    float_data: 0.247859463
    float_data: 0.439517677
    float_data: 0.149005711
    float_data: -0.295498103
    float_data: 0.491352171
    float_data: 0.00962168723
    float_data: 0.133305013
    float_data: 0.125146136
    float_data: 0.511880338
    float_data: 0.326324254
    float_data: 0.372409046
    float_data: -0.148888454
    name: "dense/kernel"
  }
  initializer {
    dims: 4
    data_type: 1
    float_data: 0.151097029
    float_data: 0.0685441419
    float_data: 0
    float_data: -0.16944176
    name: "dense/bias"
  }
  initializer {
    dims: 4
    dims: 2
    data_type: 1
    float_data: -0.190241084
    float_data: 0.0431261435
    float_data: -0.802878737
    float_data: 0.151852816
    float_data: -0.740133286
    float_data: -0.52476716
    float_data: 0.123927698
    float_data: -0.743718147
    name: "dense_1/kernel"
  }
  initializer {
    dims: 2
    data_type: 1
    float_data: -0.186544299
    float_data: -0.0322363526
    name: "dense_1/bias"
  }
  initializer {
    dims: 2
    dims: 1
    data_type: 1
    float_data: -0.358346373
    float_data: 0.735616267
    name: "dense_2/kernel"
  }
  initializer {
    dims: 1
    data_type: 1
    float_data: 0.229875788
    name: "dense_2/bias"
  }
  input {
    name: "input"
    type {
      tensor_type {
        elem_type: 1
        shape {
          dim {
          }
          dim {
            dim_value: 13
          }
        }
      }
    }
  }
  output {
    name: "output"
    type {
      tensor_type {
        elem_type: 1
        shape {
          dim {
          }
          dim {
            dim_value: 1
          }
        }
      }
    }
  }
}
opset_import {
  domain: ""
  version: 21
}
```

### Custom Model API

```protobuf
ir_version: 8
producer_name: "spox"
doc_string: ""
graph {
  node {
    input: "input"
    input: "custom_model/dense_6/kernel"
    output: "MatMul_0_Y"
    name: "MatMul_0"
    op_type: "MatMul"
    domain: ""
  }
  node {
    input: "MatMul_0_Y"
    input: "custom_model/dense_6/bias"
    output: "Add_0_C"
    name: "Add_0"
    op_type: "Add"
    domain: ""
  }
  node {
    input: "Add_0_C"
    output: "Relu_0_Y"
    name: "Relu_0"
    op_type: "Relu"
    domain: ""
  }
  node {
    input: "Relu_0_Y"
    input: "custom_model/dense_7/kernel"
    output: "MatMul_1_Y"
    name: "MatMul_1"
    op_type: "MatMul"
    domain: ""
  }
  node {
    input: "MatMul_1_Y"
    input: "custom_model/dense_7/bias"
    output: "Add_1_C"
    name: "Add_1"
    op_type: "Add"
    domain: ""
  }
  node {
    input: "Add_1_C"
    output: "Relu_1_Y"
    name: "Relu_1"
    op_type: "Relu"
    domain: ""
  }
  node {
    input: "Relu_1_Y"
    input: "custom_model/dense_8/kernel"
    output: "MatMul_2_Y"
    name: "MatMul_2"
    op_type: "MatMul"
    domain: ""
  }
  node {
    input: "MatMul_2_Y"
    input: "custom_model/dense_8/bias"
    output: "Add_2_C"
    name: "Add_2"
    op_type: "Add"
    domain: ""
  }
  node {
    input: "Add_2_C"
    output: "Identity_0_output"
    name: "Identity_0"
    op_type: "Identity"
    domain: ""
  }
  node {
    input: "Identity_0_output"
    output: "output"
    name: "Introduce_0_id0"
    op_type: "Identity"
  }
  name: "spox_graph"
  initializer {
    dims: 13
    dims: 4
    data_type: 1
    float_data: -0.245475456
    float_data: -0.51364994
    float_data: 0.14011021
    float_data: 0.280634344
    float_data: 0.459501386
    float_data: 0.522422075
    float_data: 0.361647874
    float_data: -0.0699077249
    float_data: -0.513671339
    float_data: 0.281712532
    float_data: -0.0128465034
    float_data: 0.0246540308
    float_data: -0.620902061
    float_data: 0.216346025
    float_data: 0.423825979
    float_data: -0.480461717
    float_data: -0.484473228
    float_data: -0.274029195
    float_data: 0.428079039
    float_data: -0.5181247
    float_data: -0.403283209
    float_data: 0.304576337
    float_data: 0.122853398
    float_data: -0.267789304
    float_data: 0.123277158
    float_data: -0.0594285131
    float_data: 0.487624854
    float_data: -0.231153101
    float_data: 0.1251937
    float_data: 0.00508576632
    float_data: 0.597077429
    float_data: -0.54536438
    float_data: 0.232787952
    float_data: 0.572433472
    float_data: -0.335950047
    float_data: 0.155349255
    float_data: 0.424798548
    float_data: -0.304586262
    float_data: -0.277239501
    float_data: -0.199544042
    float_data: -0.152898446
    float_data: -0.241209954
    float_data: 0.0768693388
    float_data: -0.106662184
    float_data: 0.462993801
    float_data: -0.157246143
    float_data: 0.361343443
    float_data: -0.510461688
    float_data: 0.213008225
    float_data: 0.0568366051
    float_data: 0.272384226
    float_data: 0.0342648625
    name: "custom_model/dense_6/kernel"
  }
  initializer {
    dims: 4
    data_type: 1
    float_data: -0.10359475
    float_data: 0
    float_data: 0.140471905
    float_data: 0
    name: "custom_model/dense_6/bias"
  }
  initializer {
    dims: 4
    dims: 2
    data_type: 1
    float_data: 0.598016679
    float_data: -0.186312199
    float_data: -0.0918810368
    float_data: -0.789733887
    float_data: -1.1225704
    float_data: -0.560537577
    float_data: 0.676285744
    float_data: 0.230811119
    name: "custom_model/dense_7/kernel"
  }
  initializer {
    dims: 2
    data_type: 1
    float_data: -0.107218809
    float_data: 0
    name: "custom_model/dense_7/bias"
  }
  initializer {
    dims: 2
    dims: 1
    data_type: 1
    float_data: -0.643168271
    float_data: 0.190358758
    name: "custom_model/dense_8/kernel"
  }
  initializer {
    dims: 1
    data_type: 1
    float_data: 0.110582925
    name: "custom_model/dense_8/bias"
  }
  input {
    name: "input"
    type {
      tensor_type {
        elem_type: 1
        shape {
          dim {
          }
          dim {
            dim_value: 13
          }
        }
      }
    }
  }
  output {
    name: "output"
    type {
      tensor_type {
        elem_type: 1
        shape {
          dim {
          }
          dim {
            dim_value: 1
          }
        }
      }
    }
  }
}
opset_import {
  domain: ""
  version: 21
}
```