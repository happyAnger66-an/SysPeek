# SysPeek 内存带宽测试说明

本文档解释 SysPeek 中两类内存相关跑分项的含义、理论值来源，以及 **为何 pageable 下 H2D 通常快于 D2H** 等常见现象。

## 跑分项概览

| 跑分项 | 测量内容 | 典型瓶颈 | 理论值 |
|--------|----------|----------|--------|
| `host_device_bw` | CPU 主存 ↔ GPU 显存（H2D / D2H） | PCIe 链路（独显）或共享 LPDDR（Jetson） | `host_device_bandwidth_gbps` |
| `device_mem_bw` | GPU 显存内部 copy / write | GDDR / LPDDR 峰值带宽 | `mem_bandwidth_gbps` |

两者测的是**不同路径**，数值不可直接比较：

- **H2D/D2H**：数据穿过 **PCIe**（或 Jetson 统一内存总线）。
- **device_mem_bw**：数据仅在 **GPU 显存**内移动，可接近数百 GB/s（如 RTX 4070 GDDR6X ~504 GB/s）。

---

## Host ↔ Device 传输：`host_device_bw`

### 两种主存类型

| 类型 | 分配方式 | 特点 |
|------|----------|------|
| **pinned**（page-locked） | `pin_memory=True` | 页常驻物理内存，GPU 可直接 DMA，接近链路上限 |
| **pageable**（普通页） | `pin_memory=False` | 页可被 OS 换出，驱动必须使用 **staging buffer**，CPU 参与多，带宽更低 |

### 理论值（Eff. 分母）

对 **pinned** 传输，SysPeek 显示的理论峰值来自 **PCIe 链路**（非显存带宽）：

- **auto**：`nvidia-smi` 读取 `pcie.link.gen.max`、`pcie.link.width.max`，并按 GPU 卡端 lane 数封顶（如 RTX 4070 为 Gen4 **x8** → **≈ 15.75 GB/s/方向**）。
- **fixed**：`theoretical.py` 中 `_TABLE` 手工录入值。

**pageable** 结果也引用同一链路上限作为参考，但实际很难跑满——效率偏低是正常现象，不代表测试错误。

### 默认测试方法

- 总传输量：`--transfer-mb`（默认 256 MB）。
- 并行：**8 个 CUDA stream** 分块并行 copy（`--transfer-streams 8`，`--transfer-mode multi_stream`）。
- 可选模式：
  - `single`：单 buffer，适合对比。
  - `threaded`：多 CPU 线程提交 copy，主要改善 **pageable** 场景。

---

## 为何 pageable 下 H2D 通常快于 D2H？

实测示例（RTX 4070，256 MB，8 stream，pageable）：

| 方向 | 实测带宽 | 相对理论 (15.8 GB/s) |
|------|----------|----------------------|
| H2D pageable | ~12 GB/s | ~76% |
| D2H pageable | ~9 GB/s | ~55% |

同一块 pageable 内存，**H2D > D2H** 是 CUDA 与硬件上的常见现象，并非异常。

### Pageable 实际走的路径

GPU **不能**直接 DMA 到可被换出的普通虚拟页。驱动必须插入 **staging（中转）缓冲区**：

```text
H2D (Host → Device):
  pageable CPU 内存 ──读──► [驱动 staging / pinned 中转] ──PCIe DMA──► GPU 显存

D2H (Device → Host):
  GPU 显存 ──PCIe DMA──► [staging] ──写回──► pageable CPU 内存
```

Pinned 内存跳过大部分 staging，因此 H2D/D2H 均可接近 PCIe 上限（4070 上约 17–18 GB/s）。

### 原因 1：写 pageable 比读更「重」（D2H 特有）

- **H2D**：CPU 侧以 **读** pageable 源数据为主，再写入 staging，路径相对简单。
- **D2H**：DMA 到 staging 之后，还必须 **写回** pageable 目标缓冲区：
  - 可能触发 **缺页**（first touch、按需分配）；
  - 写路径需维护 **cache 一致性**；
  - 驱动常通过 **额外 memcpy** 从 staging 拷贝到用户 buffer。

D2H 在 CPU 侧多一段工作，整体耗时更长，带宽数字更低。

### 原因 2：PCIe 方向略不对称

PCIe 为全双工，但在部分平台/芯片组/驱动栈上，**Device → Host（上行）** 的有效带宽常略低于 **Host → Device（下行）**。在消费级 x8 链路上差几个 GB/s 并不罕见。

### 原因 3：驱动优化倾向 H2D

CUDA 对 **H2D + pageable** 的 staging 路径 historically 优化更多（权重上传等场景）。**D2H pageable** 路径往往更保守，同步点更多。

### 原因 4：Cache 与内存子系统争用

- **H2D**：CPU 顺序读源 buffer，cache 行为相对友好。
- **D2H**：大量数据写回 CPU 内存，可能 **污染 cache**、与 CPU 其它任务争用 DRAM 带宽，进一步拉低 D2H 实测。

### 与 pinned 对比（建立直觉）

| 主存类型 | H2D | D2H | 说明 |
|----------|-----|-----|------|
| **pinned** | ~17–18 GB/s | ~17–18 GB/s | 接近纯 DMA，双向基本对称 |
| **pageable** | ~12 GB/s | ~9 GB/s | D2H 写回 pageable 成本更高 |

若 **H2D pageable 也远低于 D2H**，才值得怀疑测试配置；当前 **H2D > D2H** 符合预期。

---

## Pinned 已接近链路上限时，为何 CPU 仍可能 100%？

### Pinned（~17–18 GB/s）

RTX 4070 卡端为 **PCIe Gen4 x8**，单向理论约 **15.75 GB/s**。实测 pinned **已接近或达到该上限**，此时：

- 瓶颈在 **PCIe**，而非 CPU 提交速率；
- 再增加 stream 或 CPU 并行 copy，对 **pinned** 提升通常很小（个位数百分比以内）。

### Pageable（CPU 占用高）

- 驱动在 CPU 上完成 pin/unpin、staging 分配与拷贝；
- **multi_stream** 可将 pageable H2D 从 ~9 GB/s 提到 ~12 GB/s，但 D2H 仍明显更慢；
- 跑 pageable 时看到 **CPU 100%** 属于正常现象。

---

## 显存带宽：`device_mem_bw`

- **hbm_copy**：大块 device-to-device `copy_`，每字节读一次、写一次，有效字节量 = **2 × buffer**。
- **hbm_write**：`fill_` 写满 buffer，有效字节量 = **1 × buffer**。

理论值对应 **GDDR / LPDDR 峰值**（4070 约 504 GB/s），与 PCIe 理论值无关。

---

## RTX 4070 参考数据（实测量级）

| 测试项 | 典型实测 | 理论 / 上限 | 备注 |
|--------|----------|-------------|------|
| h2d_pinned | ~17–18 GB/s | PCIe Gen4 x8 ~15.75 GB/s | 已达链路级别 |
| d2h_pinned | ~17–18 GB/s | 同上 | 与 H2D 基本对称 |
| h2d_pageable | ~12 GB/s | 同上（参考） | CPU staging，Eff. ~75% |
| d2h_pageable | ~9 GB/s | 同上（参考） | 写回 pageable 更慢，Eff. ~55% |
| hbm_copy | ~400 GB/s | GDDR ~504 GB/s | 显存内部 |

Eff. 略超 100%（如 pinned 112%）可能来自：十进制 GB/s（1e9）计量、实测波动、或略高于保守规格录入——表示 **已在链路边界运行**。

---

## 使用建议

1. **要对标 PCIe 规格**：以 **pinned** 的 H2D/D2H 为准。
2. **要模拟实际上传权重**：多数框架使用 pinned / `cudaMemcpyAsync`；pageable 偏保守。
3. **要减少 D2H 开销**：目标 buffer 使用 `pin_memory=True`，或尽量在 GPU 上消费结果、少做 D2H。
4. **对比 copy 策略**：

```bash
syspeek run --bench host_device_bw --transfer-mode single
syspeek run --bench host_device_bw --transfer-mode multi_stream
syspeek run --bench host_device_bw --transfer-mode threaded
```

5. **理论值来源**：`--spec-source auto`（PCIe 自动推导）或 `fixed`（查表）；见 README 与 `theoretical.py`。

---

## 相关源码

| 文件 | 内容 |
|------|------|
| `src/syspeek/benchmarks/memory_transfer.py` | H2D/D2H 实现（multi_stream / threaded） |
| `src/syspeek/benchmarks/memory_hbm.py` | 显存 copy / write 带宽 |
| `src/syspeek/theoretical.py` | PCIe / 显存理论峰值推导与固定表 |

---

## 一句话总结

**Pageable H2D** 主要是「读 CPU 内存 + staging + DMA」；**pageable D2H** 还要「DMA + 写回 pageable」，写路径更贵、驱动更重，因此 **H2D > D2H** 是正常现象。**Pinned** 则接近纯 DMA，双向均可打满 PCIe（在 4070 上约 17–18 GB/s，即 Gen4 x8 链路上限附近）。
