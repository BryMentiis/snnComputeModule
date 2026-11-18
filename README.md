# bitsnn-cpu

`bitsnn-cpu` 是一个面向 CPU 的二值脉冲神经网络（Spiking Neural Network，SNN）**推理算子原型**。项目通过 PyTorch C++ 扩展，将二值脉冲和二值权重压缩到 64 位整数中，用位运算完成矩阵乘，并以定点整数推进 LIF（Leaky Integrate-and-Fire）神经元状态。

当前代码覆盖从底层算子到两层脉冲 MLP 的最小链路，重点验证三个问题：

1. 位压缩能否显著降低脉冲与权重的存储和传输开销；
2. `AND + popcount` 与 AVX2 能否加速 CPU 上的二值密集矩阵乘；
3. 优化后的算子能否与整数参考实现保持逐位一致，并嵌入 PyTorch / SpikingJelly 工作流。

本项目目前不是完整的 SNN 训练框架，也不是基于稀疏事件队列的仿真器；其主要技术路线是**密集位并行计算、定点状态更新和 PyTorch 算子集成**。

## 1. 项目背景与目标

SNN 以离散脉冲传递信息。若脉冲和权重都限制为 `0/1`，普通矩阵乘就可以转化为位运算：先对脉冲和权重执行按位与，再统计结果中 `1` 的数量。与逐元素整数乘法或浮点矩阵乘相比，这条路线有机会同时降低内存占用、内存带宽压力和计算成本。

项目围绕这一思路构建可运行、可对拍、可基准测试的 CPU 推理原型：

- 以 `uint64` 的位布局保存数据，在 PyTorch 侧使用 `int64` 张量承载；
- 用 C++20 `std::popcount` 实现可移植的标量内核；
- 在支持的 x86 CPU 上运行时使用 AVX2，运行时不可用时自动退回标量实现；
- 用 Q15 定点格式表示 LIF 衰减系数，避免神经元状态更新依赖浮点运算；
- 保持脉冲在层间处于压缩格式，仅在模型输入和最终输出等边界处打包、解包；
- 通过整数参考实现、单元测试和基准脚本建立正确性与性能验证基础。

## 2. 主要技术范式

### 2.1 位压缩与密集位并行

打包沿张量最后一维进行，每 64 个二值元素保存为一个 64 位字：

```text
[B, N] uint8/bool
        │ pack_spikes
        ▼
[B, ceil(N / 64)] int64
```

第 `w` 个字的第 `b` 位对应原向量第 `w * 64 + b` 个元素。尾字不足 64 位时，其余位保持为零；点积还会使用尾部掩码，避免把填充位计入结果。解包可恢复原始 `uint8` 脉冲。

在不计尾字填充的情况下，与每元素一个字节的 `uint8` 表示相比，理论存储量降为约 **1/64**。这是表示形式带来的理论压缩率，不等同于已经测得的端到端内存收益。

### 2.2 二值矩阵乘：AND + popcount

对于输入脉冲 `s ∈ {0,1}^(B×Nin)` 和权重 `W ∈ {0,1}^(Nout×Nin)`，输出电流为：

```text
I[b, o] = Σn s[b, n] × W[o, n]
        = popcount(packed_s[b] AND packed_W[o])
```

输出使用 `int32`，含义是同时有效的正权重连接数。该数学形式适用于 `{0,1}` 二值点积，**不是**用于 `{-1,+1}` 权重的 XNOR 二值网络。

当前提供三种执行方式：

- `scalar`：逐 64 位字执行 `AND + std::popcount`；
- `avx2`：一次处理 4 个 64 位字，用半字节查找表和 AVX2 指令并行计数；
- `auto`：运行时检测 AVX2，可用则选择 AVX2，否则使用标量实现。

此外保留 `binary_dense_unpacked`，用于未压缩整数路径的正确性对照和性能比较。

### 2.3 Q15 定点 LIF 状态更新

LIF 内核使用 `int32` 输入电流和膜电位，衰减系数 `decay_q ∈ [0, 32768]` 采用 Q15 格式，其中 `32768` 对应浮点衰减系数 `1.0`。单步更新为：

```text
retained = trunc((V[t-1] - V_reset) × decay_q / 32768)
charged  = V_reset + retained + I[t]
bounded  = saturate_int32(charged)
S[t]     = 1, if bounded >= threshold; otherwise 0
V[t]     = V_reset, if S[t] = 1; otherwise bounded
```

关键语义包括：

- 除法向零截断，包括负膜电位情况；
- 充电结果饱和到 `int32` 范围，避免有符号溢出；
- 达到或超过阈值即发放脉冲，并执行硬重置到 `v_reset`；
- 输出的脉冲直接打包，下一层无需先解包；
- `BitLIFNode.from_tau(tau, ...)` 可按 `round((1 - 1 / tau) × 32768)` 生成衰减系数。

膜电位和电流仍为 `int32`，因此位压缩只作用于脉冲与权重，并不表示整个网络都是 1 bit。

### 2.4 压缩格式贯穿网络

当前 `BitMLP` 的数据流为：

```text
[T, B, Nin] uint8/bool
        │ pack_spikes（模型边界）
        ▼
[T, B, ceil(Nin/64)] int64
        │ PackedBitLinear → int32 current
        │ BitLIFNode      → packed spikes + int32 membrane
        │ PackedBitLinear → int32 current
        │ BitLIFNode      → packed spikes + int32 membrane
        ▼
[T, B, ceil(Nout/64)] int64
        │ unpack_spikes（仅在需要普通脉冲张量时）
        ▼
[T, B, Nout] uint8
```

这种设计将打包开销放在边界处，避免每经过一层都恢复成逐元素表示。

## 3. 代码组成

```text
csrc/
  bit_kernels.cpp       位打包、标量点积、定点 LIF
  avx2_kernels.cpp      AVX2 点积与 CPU 能力检测
  torch_ops.cpp         参数检查及 PyTorch CPU 算子注册
src/bitsnn_cpu/
  ops.py                Python 算子封装
  modules.py            PackedBitLinear、BitLIFNode
  networks.py           两层 BitMLP
  spikingjelly.py       SpikingJelly reset_net 适配入口
tests/
  reference.py          NumPy / PyTorch 整数参考实现
  test_bit_ops.py       打包与二值矩阵乘测试
  test_lif.py           LIF 与单步/多步模块测试
  test_spikingjelly.py  MLP 对拍、状态重置和版本测试
benchmarks/
  bench_ops.py          算子及单步链路基准
  bench_mlp.py          多层、多时间步端到端基准
examples/
  minimal_mlp.py        最小可运行示例
```

主要公开接口：

| 接口 | 作用 |
| --- | --- |
| `pack_spikes` / `pack_weights` | 将 `uint8` 或 `bool` 脉冲/权重打包为 `int64` |
| `unpack_spikes` | 将压缩脉冲恢复为 `uint8` |
| `binary_dense` | 压缩二值矩阵乘，支持 `auto`、`scalar`、`avx2` |
| `binary_dense_unpacked` | 未压缩二值矩阵乘，主要用于对照 |
| `lif_step` | 执行一次定点 LIF 更新，返回压缩脉冲和新膜电位 |
| `PackedBitLinear` | 保存压缩权重并产生 `int32` 电流 |
| `BitLIFNode` | 管理膜电位，支持单步 `s` 和多步 `m` 模式 |
| `BitMLP` | 两层二值线性层与 LIF 组成的多时间步网络 |

## 4. 安装与运行

要求 Python 3.10+、PyTorch 2.3.x、NumPy 1.26.x、Ninja，以及支持 C++20 的编译器。Windows 下需要可用的 MSVC 构建环境。

```bash
pip install -e .
```

如需运行 SpikingJelly 集成测试和示例：

```bash
pip install -e ".[spikingjelly]"
```

最小示例：

```python
import torch
from bitsnn_cpu import BitMLP, pack_spikes, unpack_spikes

spikes = torch.randint(0, 2, (4, 2, 65), dtype=torch.uint8)  # [T, B, Nin]
weight1 = torch.randint(0, 2, (67, 65), dtype=torch.uint8)
weight2 = torch.randint(0, 2, (61, 67), dtype=torch.uint8)

model = BitMLP(weight1, weight2, decay_q=16384, threshold=12)
model.reset()
packed_output = model.forward_sequence(pack_spikes(spikes))
output = unpack_spikes(packed_output, n=61)  # [T, B, Nout]
```

也可以直接运行仓库示例：

```bash
python examples/minimal_mlp.py
```

## 5. 当前验证基础

### 5.1 正确性测试

现有测试已经定义了以下检查项：

- 打包/解包往返一致性，以及与 NumPy 位布局参考实现的一致性；
- `1、63、64、65、127、128` 等边界长度、全零、全一、随机数据、尾字掩码；
- `bool` 输入、非连续输入以及非法二值取值；
- 标量、自动、AVX2 和未压缩矩阵乘与整数矩阵乘参考结果一致；
- LIF 多步累积、阈值触发、硬重置、负数截断、饱和及单步/多步一致性；
- `BitMLP` 多时间步输出和两层膜电位与整数参考实现一致；
- SpikingJelly `functional.reset_net(model)` 能清空本项目的神经元状态，重置后重复执行结果一致；
- SpikingJelly 可选依赖版本固定为 `0.0.0.0.14`。

运行完整测试：

```bash
python -m unittest discover -s tests -v
```

需要注意：当前 SpikingJelly 测试验证的是**状态重置兼容性**，示例也是对同一个 `BitMLP` 使用两种重置方式后重复执行；它们尚不能证明本项目 LIF 与 SpikingJelly 原生 LIF 的数值语义已经完全对齐。

### 5.2 性能基准

仓库提供两类基准脚本：

```bash
python -m benchmarks.bench_ops
python -m benchmarks.bench_mlp
```

- `bench_ops` 比较未压缩、标量压缩、自动压缩和 PyTorch 浮点矩阵乘，覆盖不同规模与脉冲密度，并单独测量打包、解包和 LIF；
- `bench_mlp` 比较整数 PyTorch 参考网络、标量/自动压缩网络，以及包含输入打包和输出解包的完整链路；
- 两个脚本都先检查输出正确性，再用 `torch.utils.benchmark.Timer` 报告中位耗时；
- 脚本同时打印理论存储量，但仓库目前没有固定的基准结果、跨平台性能结论或持续性能回归基线。

## 6. 后续验证计划

建议遵循“**先数值正确，再端到端有效，最后讨论性能收益**”的顺序。

### P0：完善数值正确性与边界验证

- 在 AVX2 和非 AVX2 路径上运行同一组随机对拍，增加更多批次、输出维度和非 64 对齐维度；
- 系统覆盖空批次、极端阈值、正负膜电位、不同 `v_reset`、接近 `int32` 上下界及长时间累积；
- 明确并检查非法 dtype、形状、维度和非法后端参数的报错行为；
- 用模糊测试比较 C++ 与 `tests/reference.py`，要求压缩脉冲、输出电流、膜电位和最终脉冲逐元素完全相等；
- 验证重置语义，确保不同样本和不同序列之间不会发生状态串扰。

### P0：接入真实模型与任务级验证

- 加载一个已训练的二值 SNN，而不仅使用随机权重和随机脉冲；
- 固定训练模型的二值化规则、阈值、衰减、重置方式和输入编码，逐层对照电流、膜电位及脉冲；
- 在长序列上检查误差是否随时间累积，并在目标任务上比较准确率、发放率和输出分布；
- 若要求 SpikingJelly 模型迁移，增加与 SpikingJelly 原生 LIF 的独立对拍，而不是仅验证 `reset_net`。

### P1：建立可信的性能与内存结论

- 将“纯内核耗时”“输入/输出边界转换”“权重预打包”分开统计，同时报告包含全部必要转换的端到端结果；
- 扫描 `Nin/Nout`、batch、时间步、脉冲密度、权重密度和线程数，识别真正受益的规模区间；
- 与整数 PyTorch、浮点 PyTorch、未压缩 C++，以及必要时的稀疏实现使用相同数据、相同预热和相同线程设置比较；
- 除中位数外记录波动、重复次数、CPU 型号、操作系统、编译器和指令集，避免用单次运行得出通用加速结论；
- 用实际内存分析工具测量峰值常驻内存和分配次数，区分理论压缩率与真实内存收益；
- 将确认后的结果固化为可复现基线，再设置性能回归阈值。

### P1：验证可移植性与工程稳定性

- 覆盖 Windows/Linux、MSVC/GCC/Clang，以及 AVX2 可用和不可用环境；
- 验证非 x86 平台上的标量回退、安装构建和算子注册；
- 检查重复调用、大模型和长序列下的内存泄漏、异常安全及多实例状态隔离；
- 若未来加入多线程或并行执行，再单独验证线程安全与结果确定性。

## 7. 当前边界

- 仅实现 CPU 推理，不包含训练、反向传播或权重二值化训练算法；
- 权重和脉冲仅支持 `{0,1}`，不包含 `{-1,+1}` XNOR 语义、偏置或多比特权重；
- 当前网络封装是两层 `BitMLP`，尚未覆盖卷积、循环连接、更复杂拓扑和通用模型导出；
- 当前是逐输出神经元的密集循环，没有稀疏事件调度，也没有内部批量并行或多线程内核；
- AVX2 只加速二值点积，打包、解包和 LIF 仍是标量实现；
- SpikingJelly 支持目前集中在可选依赖和重置接口适配，不应视为完整框架兼容；
- 权重预打包和边界转换都有成本，最终收益必须以端到端实测为准。
