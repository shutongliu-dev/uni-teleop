# data/：DexBench 数据包

代码只从数据根目录读数据。数据根目录默认是这个 `data/`，也可以用环境变量 `DEXBENCH_DATA` 指到别处。Git 只跟踪本说明和 `MANIFEST.json`；数据本体不进 Git，从 GitHub Release 下载。

```text
data/
├── MANIFEST.json      当前代码要求的数据版本：每个文件的路径、SHA256 和大小
├── assets/            房间、机器人装配、物体、材质、天空、字体、任务物体
└── datasets/          冻结的任务发布（每个可选机器人配置一个）及其直接读取的少量输入
```

数据包只放运行任务必需的资产和本仓库测试读取的小型夹具，不放开发过程的证据：原生录像和状态数组、`.jsonl` 流、日志、演示视频都留在开发存储。冻结发布里仍记着这些证据的哈希；本地没有对应文件时，`bench.paths.undistributed` 把它们当作已记录的来源，不要求存在。数据包本身的完整性由 `scripts.data.package check` 逐文件校验。

## 获取与校验

```bash
gh release download data-v6 --repo VVVVME50/dexbench --dir /tmp/dexbench-data   # 或在网页上下载全部分卷
cat /tmp/dexbench-data/dexbench-data-v6.tar.zst.part-* | zstd -d --long=31 | tar -x -C data
.venv/bin/python -m scripts.data.package check     # 逐文件对照 MANIFEST.json 校验
```

压缩用了 zstd 长距离匹配（`--long=31`），解压时必须同样加 `--long=31`（约需 2 GB 内存）。数据包里内容完全相同的文件是同一个文件的硬链接，解压后仍然是；`scripts.data.package refresh` 改写文件时会先断开硬链接，手工编辑时也请用会新建文件的方式（例如先 `cp` 到别处改好再 `mv` 回来），不要原地写，否则会同时改到其他副本。

数据包包含两套机器人构型（原版和 L 型）的全部发布，以及多机器人任务（例如客厅抽积木对抗）事先准备好的机器人实例；运行时不往 `data/` 写任何东西。

数据版本跟着代码走：拉取到新提交后如果 `check` 报告不一致，说明 `MANIFEST.json` 变了，需要下载对应的新数据包。

## 路径约定

- 数据包里没有任何机器路径，解压到哪里都能用。
- USD 文件之间用"相对于所在文件"的路径互相引用。
- JSON 里的文件路径写成 `data://assets/...`、`data://datasets/...`，代码读入时（`bench.paths.load_json`）换成本机的数据根目录。
- Isaac Sim 内置材质只写名字（如 `OmniGlass.mdl`），由各人本机安装的 Isaac Sim 解析，数据和代码都不依赖仿真器的安装位置。

实验输出、渲染媒体等写到 `outputs/` 或 `DEXBENCH_OUTPUT`，不要写进 `data/`。

## 修改数据

可以直接修改 `data/` 里的文件，例如任务初态、规则或资产。改完运行：

```bash
.venv/bin/python -m scripts.data.package refresh
```

它找出内容变了的文件，把它们的新哈希同步到数据包里所有记录它们的清单、源码里记录的哈希（任务定义、测试）以及 `MANIFEST.json`。然后跑测试，把改动的源码和 `data/MANIFEST.json` 一起提交，再把新数据打包发布成新的 Release。

`scripts/data/package.py export` 用于从一个按 `assets/`、`datasets/` 组织的数据存储整体重新生成数据包：收集任务目录和测试用到的全部文件及其依赖，改写成上面的路径约定，重新计算哈希。
