# onnx-cse

[![CI](https://github.com/cbourjau/onnx-cse/actions/workflows/ci.yml/badge.svg?branch=main)](https://github.com/cbourjau/onnx-cse/actions/workflows/ci.yml)

`onnx-cse` is a tiny library for [common subexpression elimination](https://en.wikipedia.org/wiki/Common_subexpression_elimination) (CSE) from ONNX models.

## Example usage

The `onnx_cse` package provides a single function:

```python
from onnx_cse import eliminate_common_subexpressions
import onnx

model = onnx.load_model("model.onnx")
# Update model in-place
eliminate_common_subexpressions(model)
```

## Comparisons

It is written in pure Python with minimal dependencies and focuses on being safe, fast, and simple.
It differs from similar tools such as [onnxoptimizer](https://github.com/onnx/optimizer), [onnx-simplifier](https://github.com/daquexian/onnx-simplifier), and [onnxruntime](https://github.com/microsoft/onnxruntime)s own CSE-pass in the following ways:

### Simplicity

`onnx-cse` does one thing (CSE) but does it well.
The entire library is less then a couple of hundred lines of code and easy to understand for anybody interested.

### Performance

From personal experience `onnx-cse` handily outperforms onnxruntime's CSE pass on very large graphs with small weights (~10k nodes with nested subgraphs) while `onnxoptimizer` fails to finish its operation on such graphs at all.

### Optimization across subgraph boundaries

`onnx-cse` eliminates subexpressions in subgraphs if they can be replaced with expressions found in the enclosing scope.

## Installation

### pypi

```
pip install onnx-cse
```

### conda-forge

Using pixi:

```
pixi add onnx-cse
```

or using conda:

```
conda install onnx-cse
```

### Development

You can install the package in development mode using:

```bash
git clone https://github.com/cbourjau/onnx-cse
cd onnx-cse
pixi run pre-commit-install
pixi run postinstall
pixi run test
```
