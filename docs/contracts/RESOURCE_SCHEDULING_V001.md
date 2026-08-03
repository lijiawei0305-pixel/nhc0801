# NHC0801 服务器资源调度规范 V001

**状态**: 权威口径 — **trial 2026-08-02c**（用户指定：t=10、预留 12 核、10 端点一波）  
**主机**: `nhc614`  
**写根**: 仅 `$WJW/NHC0801`  
**配套**: `RESOURCE_PROFILES_V002.yaml`（`revision: 2026-08-02c`）

试跑后根据墙钟/扩展效率可回调 t=8 或调整预留；改默认须升 `revision`。

---

## 0. 现行默认（02c trial）

| 参数 | 值 |
| --- | --- |
| 线程 \(t\) | **10** 逻辑线程 / endpoint |
| CPU 可调度池 | **0–99**（100 核） |
| CPU 预留 | **100–111**（**12** 核，不绑 auto-fill） |
| 内存 / 端点 | **8 GiB**（实测 HWM~3.5–4.5 GiB，~2× 余量） |
| 主机内存预留 | **40 GiB** |
| 老师队列 | **10 endpoints**（Train3+Val2）×（cation+neutral） |
| 目标并发 | **N=10 一波** → 占用 **100** 核 + 预留 12 |

\[
N_{\mathrm{cpu}} = \left\lfloor \frac{C_{\mathrm{idle\ in\ pool\ 0{-}99}}}{10} \right\rfloor,\quad
N_{\mathrm{mem}} = \left\lfloor \frac{M_{\mathrm{avail}} - 40\,\mathrm{GiB}}{8\,\mathrm{GiB}} \right\rfloor,\quad
N = \max(0, \min(N_{\mathrm{cpu}}, N_{\mathrm{mem}}))
\]

池内 100 核全闲：\(N_{\mathrm{cpu}}=10\) → **正好 10 端点一波**。  
内存：10×8 GiB=80 GiB ≪ ~230 GiB available（当前机）。

---

## 1. 主机事实

| 项 | 值 |
| --- | --- |
| 逻辑 CPU | 112（0–111） |
| 内存 | 251 GiB |
| GPU | 8× V100 32 GB |

---

## 2. 调度单位

- 单位 = **endpoint**（root 的 cation 或 neutral）  
- 同 root 两端点 **可并行**  
- Parent：**CPU-only**；训练 GPU 可共存  
- Final Test：**不自动入队**

---

## 3. 进程环境

```text
OMP_NUM_THREADS=10
MKL_NUM_THREADS=10
OPENBLAS_NUM_THREADS=10
CUDA_VISIBLE_DEVICES=
# 绑核仅来自 0-99 中空闲子集；禁止绑 100-111
# xc=wb97m-d3bj
```

---

## 4. Profile

| ID | 角色 |
| --- | --- |
| `auto_fill_112_t10_r12_v1` | **默认 trial** |
| `auto_fill_112_t8_v1` | 别名 → 同 02c（过渡） |
| V001 single_27 / dual | 历史兼容 |

---

## 5. 修订记录

| 日期 | 内容 |
| --- | --- |
| 2026-08-02 | t=8 池 112；端点可并行 |
| 2026-08-02b | 8 GiB/端点（撤销 32 GiB） |
| **2026-08-02c** | **t=10；预留 12 核（100–111）；10 端点一波 trial** |
