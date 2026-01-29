# CUTLASS 3.x GEMM Tuning Parameters Explanation

This document explains the tuning parameters used in the GEMM (General Matrix Multiply) energy benchmarking framework. The goal is to find optimal configurations that balance **performance** (execution time) and **energy efficiency** across different power constraints.

## Background: How GPU GEMM Works

Matrix multiplication C = A × B where:
- A is M × K (M rows, K columns)
- B is K × N (K rows, N columns)
- C is M × N (M rows, N columns)

### Matrix Layout Conventions

CUTLASS 3.x uses specific memory layouts for operands:

| Matrix | Layout | Storage | Access Pattern | Leading Dim |
|--------|--------|---------|----------------|-------------|
| A | RowMajor | C order (row-by-row) | A[m,k] = ptr[m*lda + k] | lda = K |
| B | ColumnMajor | Fortran order (column-by-column) | B[k,n] = ptr[k + n*ldb] | ldb = K |
| C | RowMajor | C order (row-by-row) | C[m,n] = ptr[m*ldc + n] | ldc = N |

**Important:** For ColumnMajor B, the leading dimension is the column height (K), not the width (N).

GPUs divide this large computation into smaller **tiles** that fit in shared memory (SMEM), processed by thread blocks. Each thread block computes one tile of the output matrix C.

```
┌─────────────────────────────────────┐
│            Matrix C (M × N)          │
│  ┌──────┐                           │
│  │ Tile │ ← One thread block        │
│  │      │   computes this tile      │
│  └──────┘                           │
│         TileM × TileN               │
└─────────────────────────────────────┘
```

---

## Tuning Parameters Summary

| Parameter | Values | Count | Description |
|-----------|--------|-------|-------------|
| tile_m | 128, 256 | 2 | Tile size in M dimension |
| tile_n | 128, 256 | 2 | Tile size in N dimension |
| tile_k | 16, 32 | 2 | Tile size in K dimension |
| stages | 2, 3, 4 | 3 | Pipeline stages for latency hiding |
| cache_policy | CA, CG | 2 | L1 cache behavior |
| predicated | True, False | 2 | Boundary handling mode |

**Total:** 2 × 2 × 2 × 3 × 2 × 2 = **96 theoretical configurations**

---

## Detailed Parameter Explanations

### 1. tile_m and tile_n (Tile Dimensions)

**Values:** tile_m ∈ {128, 256}, tile_n ∈ {128, 256}

**What it is:**
- tile_m: Height of the output tile (rows of C computed per thread block)
- tile_n: Width of the output tile (columns of C computed per thread block)

**How it affects performance:**
- **Larger tiles** → Fewer thread blocks needed → Better data reuse → Higher arithmetic intensity
- **Smaller tiles** → More thread blocks → Better occupancy on small problems → More parallelism

**How it affects energy:**
- Larger tiles require more shared memory, potentially limiting occupancy
- Better data reuse means fewer global memory accesses (memory is expensive energy-wise)

**Trade-off:**
| Tile Size | Pros | Cons |
|-----------|------|------|
| 128×128 | Lower SMEM usage, higher occupancy | More memory traffic |
| 256×256 | Better data reuse, fewer blocks | High SMEM usage, may limit occupancy |

---

### 2. tile_k (K-dimension Tile Size)

**Values:** tile_k ∈ {16, 32}

**What it is:**
- The "depth" of data loaded into shared memory per iteration
- Each thread block iterates through K in chunks of tile_k

**How it affects performance:**
- **tile_k=32:** Loads more data per iteration → Fewer main loop iterations → Better instruction-level parallelism
- **tile_k=16:** Smaller loads → Lower SMEM usage → Potentially higher occupancy

**How it affects shared memory:**
```
SMEM usage = (tile_m + tile_n) × tile_k × 4 bytes × stages

Example:
  tile_m=128, tile_n=128, tile_k=32, stages=2:
  SMEM = (128 + 128) × 32 × 4 × 2 = 64 KB

  tile_m=128, tile_n=128, tile_k=16, stages=2:
  SMEM = (128 + 128) × 16 × 4 × 2 = 32 KB
```

**Why these values:**
- tile_k must be divisible by 8 (the MMA instruction's K dimension)
- tile_k=16 and tile_k=32 are standard choices that balance SMEM usage and efficiency

---

### 3. stages (Pipeline Stages)

**Values:** stages ∈ {2, 3, 4}

**What it is:**
- Number of buffers used for **software pipelining** of memory loads
- Enables overlapping of memory loads with computation

**How pipelining works:**
```
Without pipelining (1 stage):
  [Load A,B] → [Compute] → [Load A,B] → [Compute] → ...

With 2-stage pipelining:
  [Load₁] → [Load₂ | Compute₁] → [Load₃ | Compute₂] → ...

With 3-stage pipelining:
  [Load₁] → [Load₂] → [Load₃ | Compute₁] → [Load₄ | Compute₂] → ...

With 4-stage pipelining:
  [Load₁] → [Load₂] → [Load₃] → [Load₄ | Compute₁] → [Load₅ | Compute₂] → ...
```

**How it affects performance:**
- **More stages** → Better latency hiding → Higher throughput (especially for high-latency memory)
- **Fewer stages** → Lower SMEM usage → Potentially higher occupancy

**How it affects energy:**
- More stages = more SMEM = potentially lower occupancy = different power characteristics
- Better pipelining = GPU stays busy = higher utilization efficiency

**Trade-off:**
| stages | SMEM Multiplier | Latency Hiding | Best For |
|--------|-----------------|----------------|----------|
| 2 | 2× | Good | SMEM-constrained configs, smaller tiles |
| 3 | 3× | Better | Medium tiles with SMEM headroom |
| 4 | 4× | Best | Small tiles (128×128) where SMEM permits |

---

### 4. cache_policy (L1 Cache Behavior)

**Values:** CA (Cache All), CG (Cache Global)

**What it is:**
- Controls how data loaded from global memory interacts with L1 cache
- **CA (Cache All):** Cache data in both L1 and L2
- **CG (Cache Global):** Cache data only in L2, bypass L1

**How it affects performance:**
```
Memory Hierarchy:
  Registers → L1 Cache (fast, small) → L2 Cache (slower, larger) → Global Memory (slowest)

CA path: Global → L2 → L1 → Registers
CG path: Global → L2 → Registers (bypass L1)
```

**When to use each:**
| Policy | Best For | Reason |
|--------|----------|--------|
| CA | Data reused soon | L1 hit on reuse saves latency |
| CG | Streaming data | Avoids polluting L1 with non-reused data |

**How it affects energy:**
- L1 access uses less energy than L2
- But L1 pollution can cause cache thrashing, hurting other data

**In GEMM context:**
- Matrix data is typically streamed (used once per tile), so CG might be better
- But some access patterns benefit from L1 caching
- **Empirical testing needed** — this is why we benchmark both

---

### 5. predicated (Load Predication)

**Values:** True, False

**What it is:**
- Controls how the kernel handles **boundary conditions** when matrix dimensions don't perfectly divide by tile sizes

**predicated = True:**
```cpp
// Each thread checks if its index is valid
if (row < M && col < N) {
    load_data();  // Only load if within bounds
}
```

**predicated = False (Unpredicated):**
```cpp
// Assumes all indices are valid — no bounds checking
load_data();  // Always load — FASTER but requires alignment
```

**Requirements for Unpredicated (predicated=False):**
- M must be divisible by tile_m
- N must be divisible by tile_n
- K must be divisible by tile_k

**How it affects performance:**
| Mode | Overhead | Requirement |
|------|----------|-------------|
| predicated=True | Branch overhead per load | Works for any M, N, K |
| predicated=False | No overhead | M, N, K must align to tile sizes |

**How it affects energy:**
- Predicated mode has extra instructions (comparisons, branches)
- Unpredicated mode is more efficient but less flexible

**In our benchmark:**
- Problem sizes (8192×8192×8192, etc.) are chosen to be divisible by tile sizes
- Both modes are tested to measure the predication overhead

---

## Configuration Filtering

Not all 96 combinations are valid. We apply these filters:

### Filter 1: Shared Memory Limit
```
SMEM = (tile_m + tile_n) × tile_k × 4 × stages

GPU Limits:
  RTX 3090/4090: 99 KB max per block
  A100: 163 KB max per block
```

**Rejected examples (RTX 3090/4090):**
| Config (tile_m×tile_n×tile_k×stages) | SMEM | Status |
|--------------------------------------|------|--------|
| 256×256×32×2 | 128 KB | ❌ Rejected |
| 256×256×32×3 | 192 KB | ❌ Rejected |
| 256×256×16×4 | 128 KB | ❌ Rejected |
| 256×256×16×3 | 96 KB | ✓ Valid |
| 128×128×32×3 | 96 KB | ✓ Valid |
| 128×128×32×4 | 128 KB | ❌ Rejected |
| 128×128×16×4 | 64 KB | ✓ Valid |

### Filter 2: Warp Tile Divisibility
```
tile_m % 32 == 0  (warp processes 32 rows)
tile_n % 32 == 0  (warp processes 32 cols)
```
All our tile sizes (128, 256) satisfy this.

### Filter 3: MMA K Divisibility
```
tile_k % 8 == 0  (TF32 MMA instruction operates on K=8)
```
Both tile_k=16 and tile_k=32 satisfy this.

### Filter 4: Unpredicated Alignment (Runtime)
```
When predicated=False:
  M % tile_m == 0
  N % tile_n == 0
  K % tile_k == 0
```

---

## Valid Configurations After Filtering

For **RTX 3090/4090** (99KB SMEM limit):
- 96 theoretical → **~60 valid** configurations (varies by problem size)
- Rejected: large tile + high stage combinations exceeding SMEM

For **A100** (163KB SMEM limit):
- 96 theoretical → **~80 valid** configurations (varies by problem size)
- More configurations pass due to higher SMEM limit

---

## configurations.csv File Format

Each case directory contains a `configurations.csv` file listing valid configurations:

```csv
id,tile_m,tile_n,tile_k,stages,cache_policy,predicated
0,128,128,16,2,CA,True
1,128,128,16,2,CA,False
2,128,128,16,2,CG,True
...
```

| Column | Description |
|--------|-------------|
| id | Unique configuration identifier (0-indexed) |
| tile_m | Tile size in M dimension (128 or 256) |
| tile_n | Tile size in N dimension (128 or 256) |
| tile_k | Tile size in K dimension (16 or 32) |
| stages | Pipeline stages (2, 3, or 4) |
| cache_policy | CA (Cache All) or CG (Cache Global) |
| predicated | True (bounds checking) or False (no checking) |

---

## Why These Parameters Matter for Energy Research

1. **Performance-Energy Trade-off:** Faster isn't always more energy-efficient. A slower config might use less total energy.

2. **Power Cap Interaction:** Different configs respond differently to power limits. Some degrade gracefully; others hit cliffs.

3. **Workload Sensitivity:** The optimal config varies by problem shape (M, N, K dimensions).

4. **Hardware Utilization:** Occupancy, memory bandwidth, and compute throughput all affect energy efficiency differently.

By benchmarking all valid configurations across multiple power caps, we can:
- Find Pareto-optimal configurations (best performance at each energy budget)
- Understand which parameters dominate energy consumption
- Provide guidance for energy-aware kernel selection
