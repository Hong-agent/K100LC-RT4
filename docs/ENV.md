# 环境版本清单（实测跑通的那台机器）

> 由 `scripts/collect_env.sh` 自动生成，采集时间 2026-09-27 21:34:30 CST。
> 目标机器上先跑一遍这个脚本，和本文对不上的地方（尤其内核与驱动）优先排查。

## 操作系统与内核

| 项 | 值 |
|---|---|
| 发行版 | Ubuntu 22.04.5 LTS |
| 内核 | `6.8.0-40-generic` |
| 架构 | `x86_64` |
| 已装内核 | `6.8.0-138-generic 6.8.0-40-generic ` |
| GRUB 默认启动项 | `"Advanced options for Ubuntu>Ubuntu, with Linux 6.8.0-40-generic"` |
| CPU | Intel(R) Core(TM) i7-14650HX |
| CPU 核数 | 24 |
| 内存 | 7.5 GB（跑 27B 模型也够：权重走 mmap + 分块 H2D，宿主常驻 <1GB） |

## 加速卡与驱动

| 项 | 值 |
|---|---|
| 卡 | Hygon K100_LC（`gfx926`，120 CU，64GB HBM，无矩阵核心） |
| 驱动包 | `rock-5.7.1 6.2.35` |
| 内核模块 | `hydcu 6.2.35`，已加载：`hydcu` `hydrm_ttm_helper` `hydcu_sched` `hyttm` `hykcl` `hydrm_buddy` `hy_extra`  |
| 用户态运行时 | `/usr/local/hyhal`（`/opt/hyhal` 软链，322M） |
| 监控进程 | active（`hymgr.service`） |
| 设备节点 | `/dev/kfd` /dev/dri/renderD128 /dev/dri/renderD129 /dev/dri/renderD130   |

```

============================ System Management Interface =============================
======================================================================================
DCU     Temp     AvgPwr     Perf     PwrCap     VRAM%      DCU%      Mode     
0       33.0C    132.0W     auto     450.0W     30%        0.0%      Normal   
======================================================================================
=================================== End of SMI Log ===================================

```

## 容器与工具链

| 项 | 值 |
|---|---|
| docker | `Docker version 28.5.2` |
| DTK 镜像 | `harbor.sourcefind.cn:5443/dcu/admin/base/custom:vllm0.18.1-ubuntu22.04-dtk26.04-py3.10-20260810-qwen3.8` |
| 镜像 ID | `3c65a645b134`（构建于 2026-08-17） |
| 容器内 hipcc | `dcc version: 25.10.0-0` |
| 容器内 python | `Python 3.10.12` |
| 容器内 gcc | `gcc (Ubuntu 11.4.0-1ubuntu1~22.04.3) 11.4.0` |
| 容器内 torch / transformers | `2.10.0 5.5.0` |
| 宿主机 gcc（量化用） | `gcc (Ubuntu 11.4.0-1ubuntu1~22.04.3) 11.4.0` |
| 宿主机 python | `Python 3.10.12` |
| zstd | `v1.4.8` |
| zip / unzip | `unzip 6.0-26ubuntu3.2` `zip 3.0-12build2`  |

## 本仓库

| 项 | 值 |
|---|---|
| 提交 | `8c9d3d4 2026-09-27` |
| 编译目标 | `gfx926` |
| 预编译运行时 | `build/rt`（471 KB） |

## 说明

* **驱动只对「安装时正在运行的内核」编译**：本机装驱动时是 `6.8.0-40-generic`，
  GRUB 默认项也被锁到它（另一个已装内核 `6.8.0-138-generic` 下没有驱动模块与固件）。
  升级内核后必须重装驱动，否则重启会没有 `/dev/kfd`。
* 装驱动用的安装包是**修正版** `rock-5.7.1-6.2.35-V1.6.7-内核68修正.aio.run`
  （原厂包在内核 6.8 上编译失败，原因与 diff 见 `driver/INSTALL.md`）。
* 版本对不上时先看 `REPRODUCE.md` 的复现记录模板，把上面这些值一起贴出来。
