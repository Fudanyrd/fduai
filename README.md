# FDUAI: 高性能张量与自动微分/编译框架

## 项目简介

FDUAI 是一个面向深度学习和科学计算的高性能张量计算、自动微分与编译框架，支持 CPU 和 CUDA（GPU）后端，具备灵活的 Python 接口、自动微分、MLIR 编译与运行能力。项目采用分层架构，底层为高效 C++/CUDA 实现，向上通过 pybind11 提供 Python 绑定，并集成自动微分、神经网络组件、IR/MLIR 编译、运行时等完整功能。

## 项目架构

```
root
├── tensor/         # 高性能张量库（C++/CUDA/Python绑定）
├── fduai/
│   ├── tensor/     # Python张量接口适配
│   ├── autograd/   # 自动微分与神经网络组件
│   ├── compiler/   # IR/MLIR编译与变量作用域管理
│   ├── runner/     # MLIR编译、链接与运行
│   └── common/     # 操作符、底层库、方言等通用组件
├── examples/       # 示例代码（tensor/autograd/compiler/mixed）
└── ...
```

## 各模块功能详解

### 1. tensor 模块
- **核心**：C++/CUDA 实现的高性能张量库，支持基本张量操作、矩阵运算、设备无缝切换（CPU/CUDA）、NumPy互操作。
- **Python绑定**：通过 pybind11 提供 `tensor_module`，支持 Python 端创建/操作张量。
- **主要API**：`Tensor` 类（创建、加减乘、矩阵乘、转设备、与NumPy互转等），`Device` 枚举。
- **性能特性**：底层采用高效内存布局与并行计算，CUDA 部分支持线程/块/网格优化。
- **构建**：自定义 cc 脚本支持 C++/CUDA 混合编译。

## Tensor Module 实现架构与细节

### 1. 总体设计

- **核心目标**：实现高性能、支持CPU与CUDA的张量库，具备NumPy风格的广播、基础算子、矩阵运算、设备无缝切换、与Python高效绑定。
- **分层结构**：
  - C++/CUDA核心实现（`tensor.h`, `cpu.cc`, `backend.cu`）
  - Python绑定（`tensor.cc`，通过pybind11）
  - 构建系统支持C++/CUDA混合编译（`setup.py`, `cc`脚本）

### 2. 主要类与数据结构

- **Device 枚举**：区分张量所在设备（CPU/CUDA）。
- **Tensor 类**（`tensor.h`）：
  - 属性：`shape`（张量形状）、`num_elements`（元素总数）、`data`（数据指针）、`device`（设备类型）
  - 构造/析构：自动分配/释放CPU或GPU内存
  - 禁止拷贝，支持移动语义，防止内存泄漏
  - 支持Python风格的`__repr__`、`__getitem__`、`__setitem__`、`__len__`
  - 支持与NumPy互操作（`from_numpy`/`to_numpy`）

### 3. 张量操作与算子实现

- **基础算子**：加、减、乘、除、负号、比较、ReLU、转置、矩阵乘（dot）、sum/max等
- **广播机制**：实现与NumPy兼容的广播规则，支持不同shape自动扩展
- **CPU实现**（`cpu.cc`）：
  - 多线程（std::thread/OpenMP）加速大规模运算
  - 支持element-wise、reduce、矩阵运算等
  - 典型实现如`cpu_add`、`cpu_mul`、`cpu_dot`等
- **CUDA实现**（`backend.cu`）：
  - 每个算子对应CUDA kernel（如`addKernel`、`matmulKernel`等）
  - 支持element-wise、矩阵乘、转置、ReLU等
  - 采用block/grid/thread组织，支持高效并行
  - 支持带广播的kernel模板
  - 典型实现如`matmulKernel`（矩阵乘法）、`broadcastOpKernel`（带广播的element-wise）

### 4. Python绑定与接口

- **pybind11绑定**（`tensor.cc`）：
  - 导出`Tensor`类及其所有算子、静态方法、属性
  - 支持Python运算符重载（+、-、*、/、<、负号等）
  - 支持`to_list`、`from_list`、`to_numpy`、`from_numpy`等接口
  - 导出`Device`枚举，支持Python端设备切换
- **模块名**：`tensor_module`，可直接`from tensor_module import Tensor, Device`

### 5. 构建与混合编译

- **setup.py**：自动查找C++源文件，调用pybind11扩展
- **CUDA编译**：通过`nvcc`单独编译`backend.cu`为`backend.so`，主模块链接
- **cc脚本**：自定义编译器wrapper，统一C++/CUDA编译流程
- **依赖**：pybind11、OpenMP、CUDA Toolkit

### 6. 典型用法

```python
from tensor_module import Tensor, Device
a = Tensor((2, 3), Device.CPU)
b = Tensor.ones((2, 3), Device.CUDA)
c = Tensor.dot(a, b.to(Device.CPU))
b.to(Device.CUDA)
import numpy as np
t = Tensor.from_numpy(np.ones((2, 2), dtype=np.float32))
```

### 2. fduai 主模块

#### (1) fduai/tensor
- Python 层张量接口适配。

#### (2) fduai/autograd
- **DataNode**：自动微分核心结构，支持前向/反向传播、梯度累积。
- **nn子模块**：基础神经网络层（如linear、relu）、参数管理、前向/反向接口。
- **编译接口**：支持将神经网络结构编译为IR/MLIR。

#### (3) fduai/compiler
- **Compiler/Variable/Instruction**：IR 构建、变量管理、指令生成。
- **generate_mlir**：自动生成 MLIR 代码。
- **作用域与模块**：支持函数、模块、循环、定时等高级结构。

#### (4) fduai/runner
- **CPURunner**：MLIR 代码编译为目标文件并链接为可执行文件，自动管理临时文件。
- **pipeline**：支持自动化编译-运行流程。

#### (5) fduai/common
- **Operator**：操作符枚举与管理。
- **lib.py**：底层C库、工具函数。
- **dialect.py/op.py**：自定义方言与操作符扩展。

## Autograd Module 实现架构与细节

### 1. 总体设计与目标

- **核心目标**：实现高效的自动微分系统，支持动态图计算、神经网络基础组件、NumPy风格广播、与底层Tensor/Variable无缝协作，并可与编译器/IR集成。
- **主要能力**：
  - 支持前向/反向自动微分，算子重载与计算图构建
  - 支持神经网络层（如线性层、ReLU等）与参数管理
  - 与编译器接口对接，支持IR/MLIR生成与反向编译

### 2. 核心类与数据结构

- **DataNode 类**：
  - 封装底层Tensor/Variable，记录`tensor`、`requires_grad`、`grad`、`inputs`、`op`等属性
  - 静态方法支持`node`、`zeros`、`ones`、`from_list`、`from_numpy`等多种构造
  - 支持Python运算符重载（+、-、*、matmul、neg、relu等），自动构建计算图
  - 维护全局`topological_order`，用于反向传播的拓扑排序
  - 实现`backward`方法，递归式反向传播与梯度累加，支持广播梯度reshape
  - 支持`zero_grad`全局梯度清零

- **Operator 枚举**：定义ADD、SUB、MUL、MATMUL、NEG、RELU等算子类型，驱动自动微分逻辑

### 3. 神经网络组件

- **nn 基类**：抽象神经网络层接口，统一参数管理、前向/反向方法
- **linear 层**：实现全连接层，自动注册权重w和偏置b为可训练参数
- **relu 层**：实现ReLU激活，支持前向传播
- **参数管理**：所有nn子类均可通过`parameters()`方法获取所有可训练参数

### 4. 编译接口

- **compile_nn**：将nn模型结构、输入维度等信息转化为IR/MLIR，支持编译器上下文集成
- **compile_backward**：自动生成反向传播IR，支持loss函数、参数梯度输出、变量管理

### 5. 典型用法与API示例

- **线性回归**（见`examples/autograd/linear_regresssion.ipynb`）：
  利用DataNode构建输入、参数w/b，定义loss，自动反向传播与参数更新。
  ```python
  from autograd import DataNode, Tensor, Device
  import numpy as np
  # 构造数据
  X = DataNode(Tensor.from_numpy(X_np), requires_grad=False)
  Y = DataNode(Tensor.from_numpy(Y_np), requires_grad=False)
  w = DataNode(Tensor.zeros((n_feature, 1), Device.CPU))
  b = DataNode(Tensor.zeros((1,), Device.CPU))
  for _ in range(max_iter):
      l = DataNode.matmul(X, w) + b - Y
      loss = l * l
      item = Tensor.sum_all(loss.tensor) / 2 / len(X.tensor)
      if item < loss_tol: break
      loss.backward()
      w.tensor -= w.grad * lr
      b.tensor -= b.grad * lr
      DataNode.zero_grad()
  ```

- **神经网络拟合sin函数**（见`examples/autograd/sin.ipynb`）：
  自定义Linear层，堆叠多层+ReLU，前向传播、loss、反向传播与参数更新。
  ```python
  class Linear:
      def __init__(self, in_dim, out_dim):
          # ... 权重/偏置初始化 ...
          self.w = DataNode(Tensor.from_numpy(w))
          self.b = DataNode(Tensor.from_numpy(b))
          self.parameters = [self.w, self.b]
      def forward(self, x):
          return DataNode.matmul(x, self.w) + self.b
      def update(self, lr):
          self.w.tensor -= self.w.grad * lr
          self.b.tensor -= self.b.grad * lr
  # 构建多层网络
  lin1 = Linear(1, 64)
  lin2 = Linear(64, 32)
  lin3 = Linear(32, 1)
  for _ in range(max_iter):
      x1 = lin1.forward(X)
      x2 = DataNode.relu(x1)
      x3 = lin2.forward(x2)
      x4 = DataNode.relu(x3)
      out = lin3.forward(x4)
      loss = ((out - Y) * (out - Y)).sum()
      if loss < tol: break
      loss.backward()
      lin2.update(lr)
      lin1.update(lr)
      DataNode.zero_grad()
  ```

- **广播与梯度**（见`examples/autograd/broadcast_grad.ipynb`）：
  DataNode支持NumPy风格广播，自动生成正确的梯度，兼容PyTorch行为。
  ```python
  from autograd import DataNode
  a = DataNode.zeros([2,4], requires_grad=True)
  b = DataNode.ones([1,4], requires_grad=True)
  r = a + b
  r.backward()
  print(a.grad.to_list())  # [[1,1,1,1],[1,1,1,1]]
  print(b.grad.to_list())  # [[2,2,2,2]]
  # 更高维广播
  a = DataNode.zeros([2,1,4], requires_grad=True)
  b = DataNode.ones([3,4], requires_grad=True)
  r = a + b
  r.backward()
  print(a.grad.to_list())  # [[[3,3,3,3]], [[3,3,3,3]]]
  print(b.grad.to_list())  # [[2,2,2,2], ...]
  ```
  与PyTorch对比：
  ```python
  import torch
  a = torch.zeros((2,1,4), requires_grad=True)
  b = torch.ones((3,4), requires_grad=True)
  r = a + b
  l = torch.sum(r)
  l.backward()
  print(a.grad)
  print(b.grad)
  ```

### 6. 关键实现细节

- **算子重载与计算图**：所有基础算子均重载，自动记录操作与输入，动态构建有向无环图
- **反向传播机制**：递归式backward，支持梯度reshape、累加、拓扑排序，兼容复杂图结构
- **广播与梯度处理**：自动处理广播维度的梯度还原，保证与NumPy/PyTorch一致
- **与Tensor/Variable协作**：支持动态图与编译模式切换，底层可选Tensor或Variable
- **NumPy互操作**：支持from_numpy、to_numpy等接口，便于与主流科学计算库集成
- **零梯度与参数管理**：全局zero_grad，nn模块参数统一管理，便于优化器实现

## Compiler Module 实现架构与细节

FDUAI 的 Compiler 模块是连接高层自动微分/神经网络与底层高效执行的桥梁。其核心用途是：以灵活的 Python API 记录高层计算过程为"指令流"，自动管理变量与作用域，并将其一键翻译为高性能、可移植的 MLIR（Multi-Level Intermediate Representation）代码，支持多种硬件后端（如CPU/GPU）和更快的执行。

### 1. 核心概念与设计

- **Compiler/Instruction/Variable**：
  - `Compiler` 负责收集和管理所有变量（`Variable`）、指令（`Instruction`）、输入输出、全局变量、shape等元信息。
  - `Instruction` 记录每一步操作（如加法、乘法、矩阵乘、初始化、循环、条件、打印等），支持自动广播、内存分配、算子融合等。
  - `Variable` 封装张量的shape、dtype、唯一标识，支持自动命名、算子重载、与自动微分/神经网络协作。
- **作用域与结构化编程**：
  - `Module`/`Function`/`Repeat`/`Timer` 等类支持模块化、函数嵌套、循环、定时等结构化编程范式，便于表达复杂模型和优化。
  - 作用域上下文自动管理变量生命周期和嵌套关系。

### 2. 实现方法与MLIR生成

- **指令流记录**：
  - 通过Python上下文（with Compiler/Function/Module）自动捕获每一步操作，形成有序指令流。
  - 支持动态/静态图混合，任意算子组合，灵活表达复杂模型。
- **变量与shape管理**：
  - 所有变量自动分配唯一名称，shape/dtype自动推断，支持广播、reshape、转置等高级操作。
- **MLIR生成**：
  - `generate_mlir`/`compile_function`/`compile_module`等接口将指令流自动翻译为标准MLIR代码，支持memref分配、affine循环、算子映射、内存管理等。
  - 支持多返回值、全局变量、参数初始化、循环、定时、打印等高级特性。
- **灵活性与可扩展性**：
  - 支持自定义算子、嵌套作用域、模块化复用，便于扩展新硬件/新优化。
  - 兼容自动微分、神经网络、张量操作等高层API。

### 3. 典型用法与例子

#### 线性层前向MLIR生成
```python
from fduai.autograd import linear, compile_nn
from fduai.compiler import Compiler, generate_mlir

with Compiler() as c:
    lin = linear(10, 10)
    compiler = compile_nn(lin, [[16, 10]])

ir = generate_mlir(compiler=compiler, is_module=True, funcname='forward')
print(ir)
```
生成的MLIR片段：
```mlir
module {
    func.func @forward(%v0: memref<10x10xf32>,
        %v1: memref<1x10xf32>,
        %v2: memref<16x10xf32>) -> (memref<16x10xf32>) {
        %zero = arith.constant 0 : index
        ... // 矩阵乘、加法、广播、内存分配等
        return %v4 : memref<16x10xf32>
    }
}
```

#### 反向传播MLIR生成
```python
from fduai.autograd import linear, compile_backward
from fduai.compiler import Compiler, generate_mlir

def se_loss(y_pred, y):
    l = y_pred - y
    return l * l

with Compiler():
    lin = linear(10, 1)
    y_shape = [16, 1]
    x_shape = [16, 10]
    compiler = compile_backward(lin, se_loss, y_shape, [x_shape])

ir = generate_mlir(compiler=compiler, is_module=True, funcname='backward')
print(ir)
```

#### 参数初始化MLIR生成
```python
from fduai.autograd import DataNode
from fduai.compiler import Compiler, generate_mlir

with Compiler() as c:
    x = DataNode.from_list([2, 2], [[1, 2], [3, 4]])
    c.add_ret_stmt(None)

ir = generate_mlir(compiler=c)
print(ir)
```

### 4. 典型MLIR片段

- **线性层前向**
```mlir
func.func @forward(%v0: memref<10x10xf32>, %v1: memref<1x10xf32>, %v2: memref<16x10xf32>) -> (memref<16x10xf32>) {
    %zero = arith.constant 0 : index
    ... // 矩阵乘、加法、广播、内存分配等
    return %v4 : memref<16x10xf32>
}
```
- **反向传播**
```mlir
func.func @backward(%v0: memref<10x1xf32>, %v1: memref<1x1xf32>, %v3: memref<16x1xf32>, %v2: memref<16x10xf32>) -> (memref<10x1xf32>, memref<1x1xf32>) {
    %zero = arith.constant 0 : index
    ... // 反向传播链路、梯度累加、内存分配等
}
```
- **参数初始化**
```mlir
func.func @start() {
    %zero = arith.constant 0 : index
    %v0 = memref.alloc() : memref<2x2xf32>
    ... // 元素赋值
    return
}
```

Compiler模块极大提升了模型表达、优化和跨硬件部署的灵活性，是FDUAI高性能与可扩展性的核心基础。

## Runner Module 实现架构与细节

Runner 模块负责将编译器生成的 MLIR 代码自动编译、链接并运行于本地硬件，实现从高层 IR 到高效本地执行的闭环。它是 FDUAI 框架连接"编译-执行"环节的关键。

### 1. 用途与定位
- 负责将 MLIR 代码编译为目标文件（object file），再链接为本地可执行文件，并自动运行。
- 支持自动管理临时文件、动态链接底层 C/C++ 库（如张量打印、同步、计时等）。
- 提供 pipeline 自动化编译-优化-转换流程，便于高效部署。

### 2. 主要功能与实现方法
- **CPURunner 类**（`runner/cpu.py`）：
  - 输入 MLIR IR，自动调用 `mlir-cpu-runner`（或 `mlir-runner`），生成目标文件。
  - 自动链接底层 C/C++ 静态库（如 `printer.c`、`fence.c`、`timer.cpp`），支持张量 JSON 打印、同步、计时等。
  - 支持自定义编译/链接参数，自动管理临时文件（无须手动清理）。
  - 通过环境变量（如 `MLIR_CPU_RUNNER`、`CC`、`LD` 等）灵活配置工具链。
- **PassPipeline**（`runner/pipeline.py`）：
  - 封装 MLIR 优化/转换 pass（如 `auto_dealloc_pass`、`convert_to_llvm_pass`），一键完成内存回收、方言转换、LLVM lower 等。
  - 支持自定义 pass pipeline，便于扩展优化。
- **底层 C/C++ 库**：
  - `printer.c`：支持张量/数组 JSON 格式化打印，便于调试和结果验证。
  - `fence.c`：实现 memory fence，保证内存操作顺序，防止编译器优化带来的问题。
  - `timer.cpp`：高精度计时器，支持性能分析与基准测试。

### 3. 典型用法与端到端流程

- **MLIR 编译-运行流程**：
  ```python
  from fduai.compiler import *
  from fduai.runner.pipeline import convert_to_llvm_pass, auto_dealloc_pass
  from fduai.runner.cpu import CPURunner
  with Module() as m:
      shape = [1024, 1024]
      with Function('main'):
          a = Variable.zeros(shape)
          b = Variable.zeros(shape)
          with Repeat(1000):
              c = a + b
  code = compile_module(m)
  code = auto_dealloc_pass(code)
  code = convert_to_llvm_pass(code)
  runner = CPURunner(code, extra_link_args=['-o', 'add.out'])
  # 运行 add.out 即可执行矩阵加法
  ```
- **张量打印与调试**：
  ```python
  from fduai.compiler import *
  from fduai.runner.pipeline import *
  from fduai.runner.cpu import CPURunner
  with Module() as m:
      with Function('main') as f:
          a = Variable.ones([2, 2, 2])
          _ = a.__repr__()
  ir = compile_module(m)
  ir = convert_to_llvm_pass(ir)
  runner = CPURunner(ir, extra_link_args=['-o', 'p.out'])
  # shell 执行 p.out，输出 [[[1.000000,1.000000],[1.000000,1.000000]],[[1.000000,1.000000],[1.000000,1.000000]]]
  ```
- **与 autograd/编译器协作**：
  ```python
  from fduai.autograd import DataNode
  from fduai.compiler import *
  from fduai.runner.pipeline import *
  from fduai.runner.cpu import CPURunner
  with Module() as m:
      with Function('main') as f:
          a = DataNode.ones([4, 2])
          b = DataNode.ones([1, 2])
          c = a + b
          c.backward()
          grad_a = a.grad
          grad_b = b.grad
          _ = grad_a.__repr__()
          _ = grad_b.__repr__()
  ir = compile_module(m)
  ir = convert_to_llvm_pass(ir)
  runner = CPURunner(ir, extra_link_args=['-o', 'p.out'], extra_compile_args=['--O3'])
  # shell 执行 p.out，输出梯度结果
  ```

Runner 模块极大简化了 MLIR 到本地执行的流程，支持高效调试、性能分析和端到端部署，是 FDUAI 框架高性能与易用性的关键一环。

## 安装与构建

### 1. 安装依赖
```sh
pip install -r tensor/requirements.txt
pip install -r requirements.txt
```

### 2. 编译 tensor 模块
```sh
cd tensor && CC=$PWD/cc CXX=$PWD/cc pip install .
```

### 3. 安装主模块
```sh
cd .. && pip install .
```

### 4. 构建LLVM/MLIR（如需本地编译）
```sh
cmake -G "Unix Makefiles" -DLLVM_ENABLE_PROJECTS=mlir -DLLVM_TARGETS_TO_BUILD="Native;NVPTX" -DCMAKE_BUILD_TYPE=Release ../llvm
```

### 5. 设置环境变量
```sh
export MLIR_CPU_RUNNER=/path/to/llvm/build/bin/mlir-runner
export MLIR_OPT=/path/to/llvm/build/bin/mlir-opt
```

## 典型用法与API示例

### 张量操作
```python
from tensor_module import Tensor, Device

a = Tensor((2, 3), Device.CPU)
b = Tensor.ones((2, 3), Device.CPU)
c = Tensor.zeros((3, 2), Device.CPU)
b[0] = 5.0
d = a + b
e = Tensor.dot(b, c)
b.to(Device.CUDA)
b.to(Device.CPU)
import numpy as np
tensor = Tensor.from_numpy(np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32))
```

### 自动微分与神经网络
```python
from fduai.autograd import DataNode, nn

class MyNet(nn):
    def __init__(self):
        super().__init__()
        self.linear1 = nn.linear(10, 20)
        self.relu = nn.relu()
        self.linear2 = nn.linear(20, 1)
        self.params = self.linear1.parameters() + self.linear2.parameters()
    def forward(self, x):
        x = self.linear1.forward(x)
        x = self.relu.forward(x)
        return self.linear2.forward(x)
```

### 编译与MLIR生成
```python
from fduai.compiler import Compiler, generate_mlir
from fduai.autograd import DataNode, nn, compile_nn

with Compiler() as c:
    net = nn.linear(10, 1)
    compiler = compile_nn(net, [[16, 10]])
    print(generate_mlir(compiler, is_module=True, funcname='forward'))
```

## examples 目录说明

- `examples/tensor/`：张量基础操作、广播、线性/逻辑回归、SVM、softmax等示例。
- `examples/autograd/`：自动微分与神经网络训练（如线性回归、sin函数拟合、广播梯度）示例。
- `examples/compiler/`：IR/MLIR编译、前向/反向自动生成、初始化等示例。
- `examples/mixed/`：端到端混合示例（如线性回归全流程、MLIR生成与运行、Python与MLIR混合调用等）。

每个子目录下均有详细notebook或脚本，适合快速上手和理解各功能模块。

## 贡献与许可

欢迎贡献代码、文档与示例。请在提交前确保代码风格一致、注释清晰。

[请在此处补充项目许可证信息]