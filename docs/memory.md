# SysPeek 内存带宽测试说明

本文档解释 SysPeek 中两类内存相关跑分项的含义、理论值来源，以及 **为何 pageable 下 H2D 通常快于 D2H**、**Jetson 统一内存下为何仍有 H2D**、**模型推理主要消耗哪类带宽** 等常见现象。

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

## GPU 本地显存带宽：`device_mem_bw`

> **命名说明**：结果项为 `device_mem_copy` / `device_mem_write`。消费级显卡（如 RTX 4070）为 **GDDR6X**，数据中心卡才可能用 **HBM**；本项统测 **GPU 片上内存**（GDDR / LPDDR / HBM 均适用），与是否 HBM 无关。

- **device_mem_copy**：大块 device-to-device `copy_`，每字节读一次、写一次，有效字节量 = **2 × buffer**。
- **device_mem_write**：`fill_` 写满 buffer，有效字节量 = **1 × buffer**。

理论值对应 **GDDR / LPDDR / HBM 峰值**（4070 GDDR6X 约 504 GB/s；Jetson Thor LPDDR 约 273 GB/s），与 PCIe 理论值无关。

在 Jetson 上，datasheet 宣称的 **273 GB/s** 指的是 **LPDDR 内存子系统在 DRAM 接口上的峰值带宽**，对应 SysPeek 的 **`device_mem_copy` / `device_mem_write`**（GPU 在 device 内存上的访存能力），**不是** H2D/D2H 传输上限，也不是 marketing 里单独写的「D2D copy」术语。

---

## Jetson 统一内存：为何仍有 H2D/D2H？

Jetson（如 Thor）采用 **统一内存（Unified Memory）**：CPU 与 GPU **共享同一块物理 LPDDR**，不经 PCIe。但这 **不等于**「device 可以直接、高效地读任意 host 指针，因此不需要 H2D」。

### 物理共享 ≠ 软件零拷贝

```text
┌─────────────────────────────────────────┐
│           同一块物理 LPDDR               │
│  ┌──────────────┐  ┌──────────────┐     │
│  │ CPU 可见区域  │  │ GPU 可见区域  │     │
│  │ (host alloc) │  │ (cudaMalloc) │     │
│  └──────────────┘  └──────────────┘     │
└─────────────────────────────────────────┘
         ↑                    ↑
    不同虚拟地址空间        不同虚拟地址空间
    不同 cache 策略         不同访问路径
```

- **物理 DRAM 只有一份**，因此 H2D/D2H **不过 PCIe**。
- CUDA 里仍有 **`host` 指针** 与 **`device` 指针** 两套分配；PyTorch 中 `tensor.cpu()` 与 `tensor.cuda()` 通常是两块逻辑内存，中间靠 `copy_` / `.to(device)` 搬运。
- SysPeek 的 `host_device_bw` 测的是这条 **显式 copy 路径**（与 `pin_memory` → `.cuda()` 等实际上传权重流程一致）。

### 为何通常仍要做 H2D？

| 内存类型 | GPU 能否直接高效访问 | 典型做法 |
|----------|----------------------|----------|
| **pageable**（普通页） | 否 / 极慢 | 驱动经 **staging buffer** 再进 device |
| **pinned**（锁页） | 可 DMA，仍常显式 copy | `copy_(non_blocking=True)` |
| **cudaMalloc device** | GPU 原生路径 | `device_mem_copy` 等 |
| **cudaMallocManaged**（真统一内存） | 可以，靠按需迁移 | 首次 touch 有 page fault 开销 |

**真零拷贝**（pinned mapped、`cudaMallocManaged` 等）存在，但编程更复杂、可能有 cache coherency 开销，且 **PyTorch 默认训练/推理路径并不走零拷贝**。SysPeek 故意测最常见的 **显式 H2D/D2H**，而非 zero-copy 极限。

### Jetson 上两类带宽为何差很多？

| 测试 | Thor 典型实测 | 在干什么 |
|------|---------------|----------|
| `device_mem_copy` | ~255 GB/s | GPU buffer ↔ GPU buffer，GPU copy engine 满负荷读+写 |
| `h2d_pinned` | ~127 GB/s | CPU 侧 pinned → device buffer |

同一块 LPDDR，但路径不同：

1. **发起方不同**：H2D 由 CPU/驱动提交；device copy 纯 GPU。
2. **逻辑拷贝**：host 视图 → device 视图（即使物理地址可能很近）。
3. **cache / 一致性**：CPU cache 与 GPU L2 之间的 flush / invalidate。
4. **争用**：copy 时 CPU、其他 master 也在用内存控制器。

**统一内存消掉的是 PCIe 墙，没有消掉 CUDA 双地址空间 + 显式拷贝这套软件模型。**

### Jetson H2D/D2H 理论值为何可能为空？

`device_mem_copy` 查 `mem_bandwidth_gbps`（Thor 表中有 273 GB/s）；`host_device_bw` 查 `host_device_bandwidth_gbps`（Thor 表项可能尚未录入）。auto 推导若因 nvidia-smi memory clock 失败，H2D/D2H 的 Eff. 会显示为 `-`，**不代表测错**，而是工具缺口。即使填入 273 GB/s 作为 H2D 上界，Eff. 也仅作宽松参考——H2D 通常 **低于** LPDDR 峰值。

### 与独显对比

| | RTX 4070（离散） | Jetson Thor（统一） |
|---|------------------|---------------------|
| 物理 | CPU 主存 + GDDR **两块** DRAM | **一块** LPDDR |
| H2D 瓶颈 | **PCIe** ~15.75 GB/s | **内存子系统 + copy 路径**（非 PCIe） |
| device copy 瓶颈 | GDDR ~504 GB/s | LPDDR ~273 GB/s |
| 还要不要 H2D？ | **要**（必须过 PCIe） | **也要**（软件模型仍是 host↔device copy） |

---

## 模型推理时的内存带宽（权重已在 GPU 内存中）

权重、激活等若已 **提前在 GPU device 内存（LPDDR）中分配**，推理过程中计算访问权重，**主要消耗的是 SM ↔ global memory（LPDDR）的带宽**，即 datasheet 的 **273 GB/s** 路径。这 **不是** D2D copy，也 **不是** SMEM / TMEM / cache 之间的片上带宽。

### 「D2D」一词的歧义

| 说法 | 实际含义 | 是不是推理主带宽 |
|------|----------|------------------|
| **D2D copy**（`cudaMemcpyDeviceToDevice`） | 一块 device buffer **拷贝到** 另一块 | ❌ 推理一般不做整块拷贝 |
| **device memory 访问带宽** | SM 从 **global memory（LPDDR）** 读写 | ✅ **这才是推理主消耗** |

SysPeek 的 `device_mem_copy` 用 `dst.copy_(src)` **测量** device 内存带宽能力；推理本身 **不是在做 copy**，而是 SM / Tensor Core **直接 load** 权重，但二者共享同一条 **LPDDR 物理通道**。

### GPU 内存层级与数据流

```text
LPDDR (global / device memory, ~273 GB/s)   ← 权重常驻；主带宽消耗在这一跳
   │
   ▼
L2 cache (片上, TB/s 级)
   │
   ▼
L1 / SMEM (每 SM 片上)
   │
   ▼
寄存器 / TMEM (Blackwell 张量内存)
   │
   ▼
CUDA Core / Tensor Core 计算
```

- **权重** 太大，**常驻 global memory**，放不进 cache / SMEM。
- 每次算到某块权重，SM 需 **从 LPDDR load**（可能命中 L2，但流式访问命中率有限）。
- **SMEM / TMEM / L1 / L2** 带宽极高，**不是瓶颈**；它们用于 **减少对 LPDDR 的访问**（tile 复用、Tensor Core 喂数）。

### Prefill vs Decode

| 阶段 | 计算/访存比 | 瓶颈 | 主带宽 |
|------|-------------|------|--------|
| **Prefill**（处理 prompt） | 高（大 GEMM，权重复用多 token） | 多为 **compute-bound** | Tensor Core 算力为主 |
| **Decode**（逐 token 生成） | 低（每步 1 token，权重读一遍只用一次） | **memory-bound** | **LPDDR ~273 GB/s** |

Decode 经典估算：**每 token 时间 ≈ 模型权重大小 / 内存带宽**。例如 7B FP16（≈14 GB）在 273 GB/s 下带宽天花板约 **~19 token/s**（实际更低）。KV cache 读写也走同一条 LPDDR 带宽。

### 与 SysPeek 跑分项的对应

| 推理场景 | 主要消耗的带宽类型 | 对应 SysPeek 项 |
|----------|-------------------|-----------------|
| 权重已在 device，算子读权重 / KV | SM ↔ LPDDR（global memory） | **`device_mem_bw`**（能力参考） |
| 启动前上传权重、每步喂输入 | host ↔ device copy | **`host_device_bw`** |
| 大块 device buffer 之间搬运 | D2D copy | **`device_mem_copy`**（测能力，非推理常态） |

**一句话**：权重住在 LPDDR；计算时 SM 去 LPDDR 取数，主带宽就是 **~273 GB/s 的 global memory 带宽**；片上 memory 只是用来「少跑几趟 LPDDR」的缓存层。

---

## RTX 4070 参考数据（实测量级）

| 测试项 | 典型实测 | 理论 / 上限 | 备注 |
|--------|----------|-------------|------|
| h2d_pinned | ~17–18 GB/s | PCIe Gen4 x8 ~15.75 GB/s | 已达链路级别 |
| d2h_pinned | ~17–18 GB/s | 同上 | 与 H2D 基本对称 |
| h2d_pageable | ~12 GB/s | 同上（参考） | CPU staging，Eff. ~75% |
| d2h_pageable | ~9 GB/s | 同上（参考） | 写回 pageable 更慢，Eff. ~55% |
| device_mem_copy | ~455 GB/s | GDDR ~504 GB/s | GPU 显存内部 D2D |
| device_mem_write | ~480 GB/s | GDDR ~504 GB/s | GPU 显存纯写 |

Eff. 略超 100%（如 pinned 112%）可能来自：十进制 GB/s（1e9）计量、实测波动、或略高于保守规格录入——表示 **已在链路边界运行**。

---

## Jetson Thor 参考数据（实测量级）

| 测试项 | 典型实测 | 理论 / 上限 | 备注 |
|--------|----------|-------------|------|
| h2d_pinned | ~127 GB/s | 暂无固定表项 | 统一内存 copy 路径，非 LPDDR 峰值 |
| d2h_pinned | ~127 GB/s | 同上 | 与 H2D 基本对称 |
| h2d_pageable | ~112 GB/s | 同上 | pageable staging |
| d2h_pageable | ~113 GB/s | 同上 | |
| device_mem_copy | ~255 GB/s | LPDDR ~273 GB/s | GPU device 读+写，接近 datasheet |
| device_mem_write | ~262 GB/s | LPDDR ~273 GB/s | GPU device 纯写 |

**273 GB/s** 应对标 **`device_mem_bw`**，不应直接作为 H2D/D2H 的 Eff. 分母。跑分前建议 `sudo nvpmodel -m 0 && sudo jetson_clocks`（MAXN）。

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
| `src/syspeek/benchmarks/memory_device.py` | GPU 本地显存 copy / write 带宽 |
| `src/syspeek/theoretical.py` | PCIe / 显存理论峰值推导与固定表 |

---

## 一句话总结

- **独显 / pageable**：**Pageable H2D** 主要是「读 CPU 内存 + staging + DMA」；**pageable D2H** 还要「DMA + 写回 pageable」，写路径更贵，因此 **H2D > D2H** 是正常现象。**Pinned** 则接近纯 DMA，双向均可打满 PCIe（4070 上约 17–18 GB/s）。
- **Jetson 统一内存**：物理 LPDDR 共享，但软件上仍有 host/device copy；**273 GB/s** 是 LPDDR 峰值，对应 **`device_mem_bw`**，不是 H2D 上限。
- **推理**：权重在 device 内存中，主带宽是 **SM ↔ LPDDR**，不是 D2D copy，也不是 SMEM/TMEM 片上带宽；decode 阶段多为 **memory-bound**。
