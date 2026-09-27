# DCU 驱动（K100 标准版 / rock-5.7.1，内核 6.8）

这个目录放**文字与配置**部分，方便直接看、diff 和复现；两个大的二进制（安装包 67MB、
用户态运行时 89MB）放在 GitHub Release 里，因为 git 历史不适合塞二进制。

## 从 Release 下载

👉 **<https://github.com/Hong-agent/K100LC-RT4/releases/tag/driver-rock-5.7.1>**

| Release 附件 | 对应本机文件 | 大小 | 校验 |
|---|---|---|---|
| `rock-5.7.1-6.2.35-V1.6.7-kernel68-fix.aio.run` | `rock-5.7.1-6.2.35-V1.6.7-内核68修正.aio.run` | 67MB | md5 `6ce38851cd21478ab669c9ff6b5e45d2` |
| `hyhal-prebuilt-6.8.0-40-generic.tar.gz` | 同左 | 89MB | sha256 `98125070b33824e99acb97b45e7cd138dda8a3ad6164dd057809a79bd60797e2` |
| `K100LC-RT4-src-driver-2026-09-27.zip` | 同左（源码+驱动整包，不含权重） | 153MB | sha256 `958b074f6458042874de47f61b6c4c9d8a83e5ffa4135046a608eec83abb3489` |

```bash
cd ~/下载   # 或任意目录
curl -LO https://github.com/Hong-agent/K100LC-RT4/releases/download/driver-rock-5.7.1/rock-5.7.1-6.2.35-V1.6.7-kernel68-fix.aio.run
md5sum rock-5.7.1-6.2.35-V1.6.7-kernel68-fix.aio.run    # 应为 6ce38851cd21478ab669c9ff6b5e45d2
sudo bash rock-5.7.1-6.2.35-V1.6.7-kernel68-fix.aio.run
sudo reboot
/opt/hyhal/bin/hy-smi                                  # 看得到 DCU 0 才算成功
```

## 本目录内容

| 路径 | 说明 |
|---|---|
| `INSTALL.md` | 完整安装/验证步骤 + 三个坑（必须用修正包、锁内核、升级内核要重装） |
| `manifest.txt` | 打包时的现场快照：内核、包版本、`lsmod`、`hy-smi`、GRUB 默认项、已装内核 |
| `installer/内核68修正-说明.md` | 原厂包在内核 6.8 上编译失败的根因、diff、验证结果 |
| `installer/MD5SUMS.txt` | 修正版安装包的 md5 |
| `system/16-dcu.rules` | udev 规则：`/dev/kfd`、`/dev/dri/renderD*` 的权限与属组 |
| `system/hydcu.conf`、`system/blacklist-hydcu.conf` | modprobe 参数与黑名单 |
| `system/hymgr.service` | 监控守护进程（`/opt/hyhal/bin/hymgr`） |

## 为什么有两个"内核"要小心

驱动**只对安装时正在运行的内核**编译。本机装驱动时是 `6.8.0-40-generic`，安装脚本把
GRUB 默认项锁到了它；另一个已装内核 `6.8.0-138-generic` 下没有驱动模块与固件，
从它启动就没有 `/dev/kfd`。目标机器上同理——**升级内核后必须重装驱动**。
完整步骤见 [INSTALL.md](INSTALL.md)，环境版本对照见 [../docs/ENV.md](../docs/ENV.md)。

## 许可

驱动安装包与用户态运行时是海光/DCU 的第三方内容，遵循其原始许可；这里只做搬运与
版本记录，请在你有权使用的范围内复制与分发（详见仓库 `NOTICE`）。
