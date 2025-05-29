## Install fduai

First install the tensor module as it is a dependency:
```sh
cd /path/to/project/tensor && CC=$PWD/cc CXX=$PWD/cc pip install .
```

Then install the autograd/compiler module:
```sh
cd /path/to/project && pip install .
```

## Verify Installation

```py
from fduai.autograd import DataNode
from fduai.compiler import Compiler, generate_mlir

with Compiler() as compiler:
    x = DataNode.zeros([2, 2])
    compiler.add_ret_stmt(None)

print(generate_mlir(compiler))
```
