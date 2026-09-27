# DCU 驱动安装（K100 标准版 / rock-5.7.1，内核 6.8）

本目录是「能把这台机器跑起来」的驱动现场快照。目标机器按下面顺序做，装完再跑
`bash ../start.sh` 起服务。

## 1. 装驱动（必做）

```bash
sudo bash installer/rock-5.7.1-6.2.35-V1.6.7-内核68修正.aio.run
sudo reboot
/opt/hyhal/bin/hy-smi          # 看得到 DCU 0 就说明驱动 OK
```

三个坑（都是实测踩过的）：

* **必须用「内核68修正」那个包**。原厂包在 Ubuntu 22.04 / 内核 6.8 上会因为
  `enum drm_debug_category` 探测误判（`-Werror=missing-prototypes`）而编译失败；
  修正内容与验证见同目录的《内核68修正-说明.md》，安装包 MD5 见 `installer/MD5SUMS.txt`。
* 驱动**只针对安装时正在运行的内核**编译。本机装的时候是 `6.8.0-40-generic`，安装
  脚本会把 GRUB 默认项改回它（`GRUB_DEFAULT="Advanced options for Ubuntu>Ubuntu,
  with Linux 6.8.0-40-generic"`）；目标机器上如果默认启动的不是装驱动时的内核，
  重启后会没有 `/dev/kfd`，必须把 GRUB 默认项锁到那一个。
* 以后**每次升级内核都要重装一次驱动**。安装器需要 `linux-headers-$(uname -r)`
  和 `gcc`/`make`。

## 2. 用户态运行时（一般不用手工装）

安装器会自己铺 `/usr/local/hyhal`（`/opt/hyhal` 是指向它的软链）。要还原本机这一份：

```bash
sudo tar -C /usr/local -xzf hyhal/hyhal-prebuilt-<内核版本>.tar.gz
```

里面有 `bin/hy-smi`、`bin/hymgr`、`lib/*.so`（HIP/HSA）、`hsa/`、`vbios/` 固件，以及
`dkms/*.ko`（本机编译好的 7 个内核模块）。容器靠 `-v /opt/hyhal:/opt/hyhal:ro` 挂它，
所以这个目录**必须存在**，否则 `start.sh` 起不来。

## 3. 系统侧配置

`system/` 里是这几份文件（装驱动时会自动写好，手工核对/恢复用）：

```bash
sudo cp system/16-dcu.rules /etc/udev/rules.d/
sudo cp system/hydcu.conf system/blacklist-hydcu.conf /etc/modprobe.d/
sudo cp system/hymgr.service /lib/systemd/system/
sudo udevadm control --reload-rules && sudo udevadm trigger
sudo systemctl enable --now hymgr.service
```

## 4. 验证

```bash
ls -l /dev/kfd /dev/dri/renderD*         # 设备节点在
lsmod | grep -E "^(hydcu|hyttm|hykcl)"   # 模块已加载
/opt/hyhal/bin/hy-smi                    # 看得到卡
bash ../status.sh                        # 离线包整体自检
```

`manifest.txt` 是打包时的现场快照（内核、包版本、lsmod、hy-smi、GRUB 默认项），
目标机器对不上时先看它。
