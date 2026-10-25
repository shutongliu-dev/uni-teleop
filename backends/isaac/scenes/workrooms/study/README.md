# 书房与四项整理任务

当前固定环境为 `study_library_20261019_v7`：2.00 × 0.90 m 胡桃木行政大班台、80 mm 厚台面、双落地柜、六抽屉与整幅访客挡板，齐平皮革嵌面配缝线和黄铜细线；后方开放书架、窗帘、植物画、档案盒及阅读角补齐房间细节。固定书桌和四个操作书格保持为空。实际图片与来源见[环境说明](../../../docs/scenes/study_room.md)。

| 顺序 | 任务与规则入口 | 目标 |
|---|---|---|
| Task1 · 易 | [文具分类](tasks/stationery_sorting/README.md) `stationery_sorting` | 铅笔、马克笔、橡皮分别完整放入标注槽位 |
| Task2 · 中 | [按序叠书](tasks/book_stacking/README.md) `book_stacking` | 从底到顶依次为《机器人学基础》《星空与宇宙》《植物观察手册》，前封面向上、书脊同向 |
| Task3 · 较难 | [后架取书](tasks/book_retrieval/README.md) `book_retrieval` | 从后方书架取出《机器人学基础》平放桌面，另外两本保留原书格 |
| Task4 · 难 | [分类上架](tasks/book_shelving/README.md) `book_shelving` | 《机器人学基础》《植物观察手册》归入科学区，《建筑与空间》归入艺术区，直立且书脊朝外 |

难度顺序是任务设计，不代表测得的机器人成功率。四项已实现候选替代原书房三个 planned 槽，`document_placement` 移除；鼠标点击任务不在本轮范围。

## 规则与裁判

每项任务目录都有 `definition.py`、`rules.py`、`evaluator.py` 和说明。共用 `rules.py`、`evaluation.py` 实现配置校验与序列裁判，正式发布冻结各自目标、中文书名、分类和裁判源码。书本资产有独立封面、封底、书脊、纸页及实际几何质量。

全部任务物件的完整包络必须进入目标范围，由指定桌面／层板或紧邻下层书真实承托，完全脱离机器人，并共同稳定至少 300 ms。书本还需满足各任务规定的封面和书脊朝向。裁判检查每个 0.5 ms 原生步的位姿、速度及有符号接触；初态不计分，缺帧、无符号证据或读回错误不产生成功。预算内可以重新抓取和纠正错位；接触地板为任务失败，预算耗尽／取消为未完成，API 或仿真错误单独报告。有限动作预算允许最后一个动作完整执行，下一动作在进入物理前拒绝。

固定房间拥有建筑、家具、空表面、灯光；任务拥有可动物件、文具槽／分类牌、机器人、初态、目标与评分。任务仅复用原 FR3 + 右 SHARPA 和已审查机械支座，基座 `(-0.23,0,0.78)` m，实际支座脚距桌边 0.20 m；独立静态图片预览的支座脚距 0.25 m，二者不混用。没有修改共享控制器或编写任务专用抓取策略。

## 发布与运行

发布固定在 `<开发存储>/datasets/fr3_sharpa_bench/isaacsim/releases/study/<task_id>/<version>`：Task1 文具分类选用 v4，Task2–4 仍为 v3。文具 v4 将分类框中心从 x=0.825 m 移到 x=0.50 m，三件文具排列在 x=0.27 m 的近机器人桌边通道，长轴沿 Y；目标、框轨与印刷标签同步移动。每个 `definition.py` 指定其 manifest 哈希，`catalog.load_task` 和 canonical `task_inputs` 校验同一组文件、资产依赖和父房间。种子标识回合，当前候选物理初态固定，不宣称随机化泛化。

```bash
.venv/bin/python -m scenes.workrooms tasks --family study
.venv/bin/python -m scenes.workrooms task --family study --task book_retrieval
```

普通 `/episodes` 复用公开函数接口；`GET /episodes/{id}/task` 暴露冻结规则和粗粒度裁判原因，逐子步真值留在服务私有证据。原生短链路使用完整冻结源码、原 worker 和公开 hold：

```bash
CUDA_VISIBLE_DEVICES="$STUDY_GPU_UUID" ISAAC_ALLOW_UNVALIDATED_DRIVER=1 \
  ISAAC_BENCH_EXPERIENCE="$PWD/bench/apps/workroom.headless.kit" \
  <开发存储>/interface/.venv/bin/python -m scripts.validation.study_task_release \
  --release <开发存储>/datasets/fr3_sharpa_bench/isaacsim/releases/study/book_retrieval/v3 \
  --task-id book_retrieval --gpu-uuid "$STUDY_GPU_UUID" --output /path/to/new/evidence
```

宿主从完整冻结源码根目录用 CPU 项目解释器启动，由原生 worker 调用冻结的 Isaac 启动脚本。参数以脚本 `--help` 为准，输出必须使用新目录。发布状态保留 `prepared_unvalidated`，初始化与短 hold 证据附在发布之外，不回写旧 manifest，不代表机器人完成取书／上架。v3 冻结发布索引（开发存储 `datasets/fr3_sharpa_bench/development/study_task_v7_migration_20261019/frozen_releases_v3.json`）记录新房间绑定，实际验证范围见[环境说明](../../../docs/scenes/study_room.md)。

Task1 v4 在独立冻结 source03 中通过 300 ms hold、600 子步、601 帧裁判重放及四路 RGB-D 检查，`physics_valid=true`、worker 退出 0；见v4 原生记录（开发存储 `datasets/fr3_sharpa_bench/development/study_stationery_reach_20261020/native03/qualification.json`）和当前文具布局（开发存储 `demo/interface/scenes/study_stationery_reach_20261020/task_initial.png`）。

v7 房间此前的四份 v3 发布均在冻结 source04 中通过 300 ms hold、600 子步与 601 帧裁判重放检查，四路 RGB-D 完整，`physics_valid=true`、worker 退出 0；证据见v3 原生汇总（开发存储 `datasets/fr3_sharpa_bench/development/study_task_v7_migration_20261019/native02/qualification_summary.json`）。上述检查结束均为主动取消，未宣称完整抓放成功。

历史 v6 房间的四项 v2 在冻结 source08 中各完成 300 ms 公开 hold、600 原生子步、601 帧裁判重放和四路 RGB-D 检查，均 `physics_valid=true`、worker 退出 0。见原生汇总（开发存储 `datasets/fr3_sharpa_bench/development/study_task_release_20261019/native04/qualification_summary.json`）。这是初始场景与运行链路的检查，完整抓放尚未验证。历史 v6 任务初态：文具分类（开发存储 `demo/interface/scenes/study_library_20261019_v6/tasks01/stationery_sorting_task.png`）、叠书（开发存储 `demo/interface/scenes/study_library_20261019_v6/tasks01/book_stacking_task.png`）、后架取书（开发存储 `demo/interface/scenes/study_library_20261019_v6/tasks01/book_retrieval_room.png`）、分类上架（开发存储 `demo/interface/scenes/study_library_20261019_v6/tasks01/book_shelving_task.png`）。

固定房间构建和展示从完整冻结源码执行：

```bash
python -m scripts.assets.build_study_room --output /path/to/new/shared/asset/version
python -m scripts.assets.prepare_study_preview --environment /path/to/environment --output /path/to/new/preview
CUDA_VISIBLE_DEVICES="$STUDY_GPU_UUID" ISAAC_ALLOW_UNVALIDATED_DRIVER=1 \
  scripts/isaac_python.sh -m scripts.media.render_study_room \
  --asset /path/to/preview --output /path/to/new/render \
  --cameras room workcell tabletop shelf --samples 128
```

可复用资源位于共享 `assets`，发布及实验位于共享 `datasets`，图片位于 `demo/interface/scenes`。模型、扫描纹理、字体及许可证按来源保留；原六房间诊断、旧书房版本及失败记录均不覆盖。
