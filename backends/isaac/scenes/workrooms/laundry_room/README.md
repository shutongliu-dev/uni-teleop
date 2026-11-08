# 洗衣房任务

五项任务均支持回合开始前选择 `single_arm`、`dual_same_side` 或 `dual_opposed`。Task4 是单件T恤折叠，Task5 是单条长裤折叠；两项都从展开衣物开始。房间仍为 `laundry_residential_20261021_v3`，衣物与机器人安装由任务组合。旧发布、旧叠放任务和历史录像保持各自身份。

2026-11-03 初态与贴图：五任务的三种安装分别使用独立 `laundry_randomized_release.v1`，
每个发布冻结 `seed=0..9` 十个小案例，共150个。父输入是各自原 `completion_inputs_v1`；
seed 0 保留原物理初态和外观，seed 1–9 只将待操作衣物沿世界 XY 平移。
其中13套安装使用每轴最多4 cm的v1方案，先按原声明工作区的方向余量限制幅度并留2 mm边界。
T恤单臂、同侧双臂使用显式v2近侧方案：X幅度最多20 mm，Y只向机器人侧移动；
单臂Y幅度20 mm，同侧受原工作区限制为8.008071 mm，半幅位置分别为10 mm和4.0040355 mm。
每套都有10个不同的XY位置，有限方向表在作者输入中固定。朝向、高度和初始褶皱保持原样。
叠放任务只移动待搬运毛巾，底层毛巾和全部任务目标坐标保持原位。
新外观在原 albedo 图集上叠加条纹、格纹或斜纹，暗纹强度13.2%–22.8%；
原织纹、缝线、口袋结构及法线、粗糙度等其他贴图保留。尺寸、静止网格、材料、接触、
质量、求解器、机器人安装与共享控制器都不变，运行中不写衣物节点。
回合通过既有 `--seed N` 选择，默认安装仍为 `dual_opposed`；公开合同只给所选案例身份，
不公开私有布点或重置坐标。新的冻结规则源码仅调整初态说明的一处文字，数值阈值、
状态机和原生观察器保持原样。新输入位于共享 `laundry_seed_randomization/v1` 资产目录和
`laundry_initial_randomization_v1` 发布目录；旧发布保留。输入发布、离散可达性、原生初态、
HTTP接口与实际机器人折衣分别验证，不继承历史通过结论，也不以被动保持计作整理成功。
最初T恤v1远侧方案有5个案例未通过原抓取姿态检查；固定多起点复核以及正Y缩至20 mm
的诊断均保留，未被计为最终输入通过。v2统一采用近侧范围，不逐seed更换评分或物理门槛。
最终输入选择见150案例清单（开发存储 `datasets/fr3_sharpa_bench/development/laundry_seed_randomization_20261103/case_selection02.json`）。
摊平的被动诊断统一观察1.5秒，其他任务0.8秒，都检查原0.5秒稳定末窗；
这是诊断时长，正式回合的reset及原生初始化预热没有增加。

本轮最终150个案例通过初态保护、实际台面支撑及1230个离散抓取目标检查；每个目标
至少有一只原手的接近、接触、闭合、抬升见证，另一只手保持待机，不代表所有双手组合或连续轨迹可行。
150例原生被动记录共210150帧、210000子步，全部经原解释器和冻结裁判独立回放；
15套安装另在seed 5完成真实GPU Bench／Agent接口、RGB-D、腕部微动、返回、保持及正常退出。
CPU完整Agent流程单独覆盖15套安装（14项显式seed 5、1项默认seed 0），不混入原生统计。
全部测试为0外部模型调用，完整抓取／折衣策略和RSI资格仍独立验证。
查看最终验证记录（开发存储 `datasets/fr3_sharpa_bench/development/laundry_seed_randomization_20261103/qualification.json`）
和150案例／195张原生图集（开发存储 `demo/interface/scenes/laundry_seed_randomization_20261103/index.html`）。
图集直接链接原始PNG，按任务、安装和seed筛选，并逐例显示实际观察时长。

2026-10-25历史发布：当时15个catalog选择指向 `fabric_visual_v1`，同时启用已显式绑定的
台面材质和新版面料外观。15种配置分别完成新原生HTTP／Agent接口与原解释器裁判回放，
验证结果单独保存在本轮开发记录（开发存储 `datasets/fr3_sharpa_bench/development/laundry_publication_20261025/root/qualification.json`），
不写入任务目录，不作为Bench／Agent启动条件或裁判输入。每次任务均根据本次实际仿真状态评分。
使用[当前 Bench／Agent 启动指南](../../../docs/agent_bench/laundry_publication_20261025.md)，
查看15构型真实场景图（开发存储 `demo/interface/scenes/laundry_publication_20261025/index.html`）。

该次2026-10-25发布保持当时的规则、初态、机器人与共享控制器。123个离散工作空间点通过准确的几何身份关系
复绑；三个毛巾目标另有本轮3秒被动仿真和末0.5秒检查，衣物合法目标的原有证据保留各自来源。
Task1单臂、Task4同侧双臂、Task5对侧双臂另有完整Agent自动选型流程的代表检查，
不把这三项代表检查写成另外12项也执行了完整runner。
原生接口和目标可行性不代表完整机器人抓取、折衣策略成功；数据采集与策略验证仍独立进行。
旧 `table_material_v1`、旧发布和全部失败记录保留，冻结manifest的准备期标志不改写。

| 编号 / canonical ID | 目标 | 动态物体 | 当前名义材料厚度 / 质量 |
|---|---|---|---|
| Task1 / `towel_flattening` | 展开局部折回的毛巾、摊平并释放 | 一条毛巾 | 4 mm / 144 g |
| Task2 / `towel_folding` | 沿长边中线对折，两半基本重合并释放 | 一条毛巾 | 4 mm / 144 g |
| Task3 / `towel_stacking` | 把一条已折毛巾叠到另一条上并释放 | 两条毛巾 | 每条 4 mm / 144 g |
| Task4 / `shirt_folding` | 单件展开T恤收袖、折好并释放 | 一件T恤 | 1.6 mm / 240 g |
| Task5 / `trousers_folding` | 对齐两条裤腿，折好单条长裤并释放 | 一条长裤 | 1.8 mm / 390 g |

材料参数是仿真先验。新厚度有独立原生物性记录，不能继承旧3 mm毛巾或1.2 mm衣物的成绩。Task1采用12 cm局部折回初态，3秒无操作沉降后仍保持折回，刚体对齐后的摊平RMS为55.73 mm，不能靠被动等待直接完成。T恤和裤子的展开初态来自对应新材料的真实无控制沉降，再按安装选项做明确记录的刚体摆放变换；不改衣物静止形状、拓扑或运行中布点。

## 三种安装

| 选项 | 机器人基座位置 `(x, y, z)`，单位 m | 朝向与关节 |
|---|---|---|
| `single_arm` | right `(0.75, −1.15, 0.90)` | 面向 +Y；官方右SHARPA与FR3，29 DOF |
| `dual_same_side` | left `(0.45, −1.15, 0.90)`；right `(1.05, −1.15, 0.90)` | 两臂均面向 +Y；官方左/右手，58 DOF |
| `dual_opposed` | left `(0.75, 0.15, 0.90)`；right `(0.75, −1.25, 0.90)` | 左臂面向 −Y、右臂面向 +Y；58 DOF |

每个选项绑定独立冻结release，共15个；默认是 `dual_opposed`。一名Agent通过既有共享控制器操作所选机器人，不在回合中切换安装。单臂允许依次重抓和利用台面整理，裁判不要求必须双手同时操作。形状、支撑、材料层次、操作信用和释放稳定的数值条件保持相同，见 [RULES.md](RULES.md)。

单臂与同侧双臂的毛巾工作区相对原安装向东移10 cm、向南移8 cm；长裤向南移8 cm。T恤单臂采用累计yaw0、向南移4 cm，同侧双臂采用yaw90、向南移3 cm，对置保留原摆放。初态、折叠参考、目标和公开位置说明一致变换；现有台面有效区域保持原边界。T恤单臂和同侧双臂的初态最小有效区域余量分别为50 mm与10.008 mm。

## 发布与验证

台面修复前的实验根为 `<开发存储>/datasets/fr3_sharpa_bench/development/laundry_contact_options_20261023`。Task1、4、5的9个选项位于 `releases/<canonical_id>/<configuration>/contact_v2`，保留 `source_configuration03` 构建与 `source_configuration_runtime04` 接口资格。Task2三个选项位于 `fold_practical_v3_candidates01/towel_folding/<configuration>`，显式采用 `laundry_practical_folding_task.v3`，普通接口在 `source_configuration_runtime07` 独立重验。Task3三个选项位于 `stack_practical_v3_candidates01/towel_stacking/<configuration>`，采用 `laundry_practical_stacking_task.v3` 与 `towel_isometric_u_fold_v1` 初态，普通接口来源为 `source_configuration_runtime06`。两次规则发布都保持对应前版scene、initial、physics_profile、asset_file_hashes四文件逐字节相同。Task1保留 `laundry_practical_task.v2`，单件衣物保留 `laundry_single_garment_folding_task.v4`。旧发布与原结果均保留。

实际夹持空间检查（开发存储 `datasets/fr3_sharpa_bench/development/laundry_contact_options_20261023/configuration_workspace_qualification02.json`）覆盖全部15个配置及123个衣缘/材料角点：真实拇指、食指的10 cm接近、接触和5 cm抬升，以及5个闭合姿态通过IK、环境/安装座/另一臂待机净空和非相邻FR3凸体检查。检查采用实际新初态，包含折回毛巾的四个材料角；原123点见证保留 `contact_v1` 身份，经15组场景、初态、物理参数逐文件相同的明确绑定记录（开发存储 `datasets/fr3_sharpa_bench/development/laundry_contact_options_20261023/public_material_identity_transition02.json`）适用于 `contact_v2`，没有重新声称执行IK。这是离散位姿检查，不等同于连续轨迹、同时双臂整理动作或实际抓住衣物。

上述旧15项均完成真实普通Bench/Agent接口检查：实际腕部微动、返回、保持、四路或五路RGB-D、取消和进程退出均通过。Task2三项新版接口及逐原生帧冻结裁判精确回放见Task2接口资格（开发存储 `datasets/fr3_sharpa_bench/development/laundry_contact_options_20261023/practical_fold_native_interface_qualification01.json`），共6,963个原生帧／177次控制评分精确一致；其12个旧离散目标通过四物理文件逐字节相同明确复绑，没有新跑IK。Task3三项接口资格（开发存储 `datasets/fr3_sharpa_bench/development/laundry_contact_options_20261023/practical_stack_native_interface_qualification01.json`）覆盖6,963个原生帧／177次控制评分，其30个离散点／321次IK同样仅沿用原独立检查。Task1、4、5的九项保留原runtime04资格，模型调用为0。接口微动不是完整整理任务或纯摩擦抓取成功。

原生接口记录、场景图和抓取/折叠演示分别绑定自身证据。普通 `/episodes` 仅写关节目标，不能写布点或增加抓持附件；演示中若使用真实捕获后的局部辅助，会在视频入口明确说明。微动、保持或抓持局部实验不计作完整折叠成功。方法、失败对照与实际捕获证据见[接触修复记录](../../../docs/scenes/laundry_contact_options.md)。

五任务 × 三安装场景图与新演示（开发存储 `demo/interface/scenes/laundry_tasks_20261021/contact_options_20261023/index.html`）

Task3 v3稳定窗口仍为0.5秒，按每原生步与20毫秒前的真实位置差计算每条毛巾节点速度P95≤0.08m/s，并把全节点窗口漂移收紧为8毫米；原每dt速度仍完整保留。折后高度采用静止材料面积加权P99−全节点最低点≤40毫米，同时所有节点绝对高度差≤45毫米。缺失连续帧或20毫秒前史、未完全释放、严重变形/交叉仍失败。

`captured_towel_stack_native08`的真实双指捕获及局部手腕辅助、完整释放已独立记录；同流v3终态复评（开发存储 `datasets/fr3_sharpa_bench/development/laundry_contact_options_20261023/research/stack08_practical_v3_offline79.json`）通过，20毫秒位置速度P95为0.01760m/s、全节点漂移4.529毫米，末帧面积P99高度36.589毫米。原v2失败保留，旧失败轨迹04仍被新规则拒绝。此为原视频的离线终态重判，没有新跑动作，也没有完整普通SequenceEvaluator任务评分；不授普通策略或纯摩擦抓取成功。runtime06的普通接口复验单独绑定。

Task2 v3保留对应RMS≤30毫米，使用静止材料面积加权配对误差P95≤60毫米及全点80毫米绝对上限，其他形状、层次、0.5秒释放及原每dt速度/15毫米漂移要求不变。`captured_towel_fold_native09`的同流终态复评197（开发存储 `datasets/fr3_sharpa_bench/development/laundry_contact_options_20261023/research/towel_fold09_practical_v3_offline197.json`）在1001帧全部通过：末帧RMS10.07毫米、面积P95误差14.29毫米、全点最大63.30毫米、正确层次99.83%，窗口漂移1.273毫米。旧v2因一个自由角超过60毫米的失败保留。该录像同样使用实际夹持后的声明辅助，只有离线终态通过，没有新动作或普通整任务评分；runtime07普通接口是独立资格。

## 显式台面材质候选

历史场景的 `table.friction=0.45` 没有绑定到所引用洗衣房的真实桌面碰撞体。
新候选可显式选择 `laundry_table_friction_045_v1`：只给桌面绑定静/动摩擦
0.45、恢复系数0，以及 friction/restitution 的 `average` combine。
这些是有来源声明的仿真先验，不是实测材质；地板、台面几何和静态 actor 类型保持原样。
默认构造和所有未选择该选项的冻结发布保持原样。

从包含修复的完整冻结源码运行，每个原发布派生到独立的新目录：

```bash
python -m scripts.assets.prepare_laundry_table_material \
  --source-release <开发存储>/datasets/<experiment>/<original-release> \
  --output <开发存储>/datasets/<experiment>/<new-table-candidate>
```

工具支持五任务的三个安装选项，也支持当前Task2/3的practical v3发布；它不修改catalog。
只改候选的 `scene.json`、`physics_profile.json` 和 `source_provenance.json`，
重新绑定manifest与原发布/新源码哈希。任务、规则、重置、冻结裁判、布料物理和共享资产文件
保持字节一致，所有资格保持未验证。`workroom_surface_materials` 同时存在于scene与physics
profile，普通 `task_inputs()` 读取后仍保留；它与 `cloth_physics_options` 分开。

原生初始化后的 `runtime_workroom_material_check` 检查实际桌面绑定路径。
静态桌面的摩擦系数及combine仍标为USD回读，不能当作native数值或接触动力学验证。
候选需要新的原生绑定、接触和普通接口证据后才能决定是否发布，旧资格不继承。

普通接口检查使用另行封存、确实选择这15个候选的源码；`source_runtime01`负责候选准备与
独立物理诊断，其catalog仍保留原版本。`public_http_smoke --configuration`会按冻结catalog
选择发布，仅替换 `--release` 参数不足以验证新候选。最小接口检查包含真实Agent选择、
RGB-D、只读IK、微动/返回、0.6秒保持和原解释器逐帧裁判重放，不计完整衣物操作成功。

## 渲染面料外观与候选来源（只改外观）

旧外观是单色闭合壳加贴边色带，远看像纸片。新外观由布料 scene 的 `visual_profile`
（profile 路径加 sha256）显式选择，`FabricClothVisual`
（`bench/simulation/isaacsim_backend/fabric_visuals.py`）只读跟随原生中面：
固定 Loop 细分一级、原厚度闭合皮肤、OmniSurface（带 sheen 的布料 MDL）。
贴图是按衣片坐标烘焙的 CC0 Poly Haven 扫描（针织、牛仔、毛圈），染成冻结颜色，
并画出成衣结构：T恤罗纹领、绱袖线、双针卷边；牛仔裤腰头、裤袢、门襟J线、
前袋与零钱袋、铆钉、后育克、后袋装饰线、皮牌、猫须与磨白；毛巾毛圈结团、
平织边带与包边。洗后褶皱、缝边圆角、缝线起皱只写在法线贴图里，几何、厚度、
质量、摩擦与求解器参数全部不变。双片缝合衣物用前/后图集，单层毛巾用正/反图集，
两半都沿 u 镜像，保证翻折后可见面的切线手性正确；渲染顶点按图集分片复制，
所有缓冲保持 vertex 插值。profile 绑定布料静止网格哈希与冻结颜色，不匹配即拒绝。

贴图版本由冻结源码烘焙到新的共享资产目录（不可覆盖）：

```bash
python -m scripts.assets.bake_laundry_fabric_textures \
  --output <开发存储>/assets/materials/laundry_fabric_visual/<version> \
  --cloth shirt <shirt_release> 0 --cloth trousers <trousers_release> 0 \
  --cloth towel_blue <towel_release> 0 --cloth towel_ochre <towel_stacking_release> 1
```

再把任一现有发布派生为只改外观的新候选（不改catalog）：

```bash
python -m scripts.assets.prepare_laundry_fabric_visual \
  --source-release <release> --output <fresh_candidate> \
  --profile towel <assets>/<version>/towel_blue/profile.json
```

派生只改 `scene.json`（布料 `visual_profile`）、`asset_file_hashes.json`（加入 profile
与贴图）和 `source_provenance.json`；物理参数、初态、规则、冻结裁判逐字节不变，
`physics_changed=false`，不继承任何资格。原生渲染需单独取证。

2026-10-25 首版：贴图资产
`<开发存储>/assets/materials/laundry_fabric_visual/v1`（T恤、长裤、蓝/赭毛巾四套），
由15个 `table_material_v1` 派生的候选位于
`<开发存储>/datasets/fr3_sharpa_bench/development/laundry_fabric_visual_20261025/releases/<task>/<configuration>/fabric_visual_v1`。
证据汇总（开发存储 `datasets/fr3_sharpa_bench/development/laundry_fabric_visual_20261025/fabric_visual_evidence.json`）：
原报告记录15/15 候选经 canonical env 原生加载、机器人保持0.5秒后实时RTX渲染通过（单臂4路相机、双臂5路相机），全部布料为新外观；
保留的失败尝试（渲染产品空帧、9B显存不足）均已单独记录并重跑。确定性对照中毛巾摊平原/新版布点逐位一致；
双层T恤即使原版两次启动也不逐位一致，衣边局部约5.7 mm弹起在原版1/5、新版2/3出现（Fisher p≈0.46），
这些小样本不足以证明物理等价。对比页面（开发存储 `demo/interface/scenes/laundry_tasks_20261021/fabric_visual_20261025/index.html`）
含原/新折叠视频。这批历史记录只是渲染与发布加载检查，不是抓取、折叠或任务资格；
当时catalog仍指向原 `table_material_v1`，本轮新发布及独立原生验证见本文开头。

合并核验记录（开发存储 `datasets/fr3_sharpa_bench/development/laundry_claude_merge_20261025/root/integration01.json`）
确认候选、贴图及物理/规则/初态/裁判文件的绑定；同时保留历史取证限制：20份 `native02`
报告中17份所记录的运行脚本哈希与当前 `research/native_fabric_check.py` 不符，且原报告未单独绑定进程退出回执。
因此本次核验不将这些报告认证为可完整复现的原生任务资格；历史报告和脚本均不改写。

## 材料与折形可保持性检查

2026-10-24的原生动态证据（开发存储 `datasets/fr3_sharpa_bench/development/laundry_material_repair_20261024/material_dynamics_review01.json`）
包含5个独立进程，每个记录3秒、1501帧。各组采用同一显式台面材质；衣物刚度、厚度、质量、
静止网格、机器人和原v4裁判不变。仅在reset时声明初始折形，运行期间没有布点写入或附件。

| 被动诊断初态 | 原形状与稳定末窗 | 末态自交 | 结论范围 |
|---|---|---:|---|
| 原生展开T恤 | 稳定，折叠目标不通过 | 0 | 展开负例 |
| T恤原解析折形 | 最后251帧通过 | 5对微小交叉 | 每帧均在原v4交叉尺寸容差内，机器人完全分离 |
| 原生展开长裤 | 稳定，折叠目标不通过 | 0 | 展开负例 |
| 长裤原解析折形 | 形状、速度不通过 | 1192对 | 原目标初态有1197对硬交叉，失败原样保留 |
| 长裤无自交折形候选 | 最后251帧通过 | 0 | 原裁判RMS9.50mm、层次96.47%，每帧物理与机器人分离门通过 |

长裤原参考的第二次横折半径12mm小于已有14mm层高，产生负内半径和层间穿插。
诊断候选保留原评分参考，只在初态取消竖向预压缩并让折弯半径覆盖已有层高，
其生成代码、节点数组和独立检查均由哈希绑定。该候选在原生松弛前的面内应变P95仍为26.95%，
不能称等距构造；松弛后降至1.28%。其通过说明原评分容差内存在稳定形状，
不证明原解析目标处处物理合理，也不代表机器人已能从展开状态折到那里。
本次没有改评分门槛、选定发布的任务参考或衣物材料刚度。

长裤初态与原生末态对照图（开发存储 `datasets/fr3_sharpa_bench/development/laundry_material_repair_20261024/research/pants_feasibility_comparison01.png`）
以红色标记发生中心面交叉的三角形；高度显示放大3倍，原始数据与图像来源另有哈希记录。

物理检查使用`source_runtime01`/`source_runtime02`；后者仅增加带来源验证的初态候选准备功能，
原生运行器、后端和控制器字节相同。保留初始化时的4ms原生预热，从随后真实`cloth_q`记录判读；
导出的USD Mesh.points仍可能是作者初态，不能当作末态。5次完整运行均在GPU0的NVIDIA L20X上正常退出。
GPU1两次在Kit启动阶段出现Vulkan设备丢失/exit139，单独保留为运行环境失败。

主动操作仍需单独修复。旧`shirt29`、`pants12`已完成声明的局部辅助抓持与释放，
但最终层叠率约80.58%/76.67%、高度超量约50.18/47.86mm，原规则仍失败；
`pants13`第一折后的弯曲裤脚又使第二阶段平面抓取入口被拒绝。
这些动作轨迹不能由上述被动终态通过替代。毛巾`flap13`另有实测指垫相对wrist运动
导致捕获确认丢失的证据；未合入的中点锚点草稿只有CPU与离线反事实检查，
没有新的原生抬起/放下/释放证据，且搬运期间持续双指接触仍需逐帧检查。

## 原版本 Bench 与 Agent 启动

以下命令保留原发布的复现方式。当前新版外观任务使用[本轮启动指南](../../../docs/agent_bench/laundry_publication_20261025.md)。

两端使用同一冻结源码。下面启动Task2的新v3默认对置发布，服务会公布三个安装选项；Agent可自动选择，也可通过 `--configuration` 固定。Task1、4、5仍使用runtime04和原contact_v2路径。当前Task3两端使用 `source_configuration_runtime06`，服务的 `BENCH_TASK_RELEASE` 为实验根下 `stack_practical_v3_candidates01/towel_stacking/dual_opposed`，Agent用 `--task task3`。旧源码仍绑定各自旧发布，不能替代新规则发布。

```bash
LAUNDRY_DATA=<开发存储>/datasets/fr3_sharpa_bench/development/laundry_contact_options_20261023
LAUNDRY_SOURCE="$LAUNDRY_DATA/source_configuration_runtime07"
LAUNDRY_RUN=<开发存储>/datasets/fr3_sharpa_bench/agent_bench/laundry/$(date +%Y%m%d_%H%M%S)
LAUNDRY_PORT=8765
: "${LAUNDRY_GPU_UUID:?请设置本次空闲 GPU UUID}"
cd "$LAUNDRY_SOURCE"
PYTHONDONTWRITEBYTECODE=1 <开发存储>/interface/.venv/bin/python -B \
  -m scripts.maintenance.freeze_project_sources verify "$LAUNDRY_SOURCE"
env -u ISAAC_PYTHON_LAUNCHER \
  PYTHONPATH="$LAUNDRY_SOURCE" PYTHONDONTWRITEBYTECODE=1 \
  BENCH_TASK_RELEASE="$LAUNDRY_DATA/fold_practical_v3_candidates01/towel_folding/dual_opposed" \
  BENCH_AGENT_DATA_DIR="$LAUNDRY_RUN/episodes" BENCH_DEVICE=cuda:0 \
  CUDA_VISIBLE_DEVICES="$LAUNDRY_GPU_UUID" ISAAC_BENCH_SINGLE_CUDA_DEVICE=1 \
  ISAAC_ALLOW_UNVALIDATED_DRIVER=1 \
  ISAAC_BENCH_EXPERIENCE="$LAUNDRY_SOURCE/bench/apps/workroom.headless.kit" \
  <开发存储>/interface/.venv/bin/python -B -m uvicorn bench.runtime.server:app \
  --host 127.0.0.1 --port "$LAUNDRY_PORT"
```

另一个终端启动Agent，凭据留在外部配置中：

```bash
LAUNDRY_SOURCE=<开发存储>/datasets/fr3_sharpa_bench/development/laundry_contact_options_20261023/source_configuration_runtime07
LAUNDRY_AGENT_RUN=<开发存储>/datasets/fr3_sharpa_bench/agent_runs/laundry/$(date +%Y%m%d_%H%M%S)
LAUNDRY_PORT=8765
cd "$LAUNDRY_SOURCE"
PYTHONPATH="$LAUNDRY_SOURCE" <开发存储>/interface/.venv/bin/python -B \
  -m agent.model_client.function_runner \
  --project-root "$LAUNDRY_SOURCE" --bench-url "http://127.0.0.1:$LAUNDRY_PORT" \
  --credentials <开发存储>/interface/configs/agent_bench/local_gateway_credentials.json \
  --family laundry_room --task task2 --configuration auto --sim-budget none --http-timeout 900 \
  --output "$LAUNDRY_AGENT_RUN"
```

`auto` 在创建回合前让模型读取公开规则和布局，选择一个已公布的安装；`single_arm`、`dual_same_side` 或 `dual_opposed` 可固定实验条件。公开返回值不包含私有发布路径或衣物真值。已创建回合的配置ID及manifest hash保持固定。

## 历史固定对置发布

下列保留早期3 mm毛巾、1.2 mm单件衣物及更早叠放实验的原发布、录像和复现说明；相关资格仅适用于这些历史版本。

### 五项任务的双臂入口与规则

五项任务的机器人接口均采用同一 Agent 在普通 `/episodes` 会话中控制两台 FR3 和官方左、右 SHARPA，共 58 DOF。复用机械 Task10 的 `dual_fr3_sharpa_w01/isaacsim_6_1_v2` 资产及共享双手控制器 `classical_fingertip_bimanual`，手名为 `left/right`，同一动作的 `hand_geometries` 可同时提交两手目标。Task1 本次加入双臂范围；不规定必须按固定双手顺序操作。

两臂安装位置仍为 left `(0.75, 0.15, 0.90)m`、yaw `−π/2`，right `(0.75, −1.25, 0.90)m`、yaw `+π/2`，提供三个场景相机和左右腕相机，共五路 RGB-D。命名布料使用 `author-only` 初始化；普通机器人动作仅写关节目标，不能驱动布点、重写布料位置/速度或增加隐藏抓持附件。

Task1/2保留 `laundry_practical_task.v2` 的原公开规则，Task3当前版本见上文与[RULES.md](RULES.md)；Task4/5 当时候选使用独立的 `laundry_single_garment_folding_task.v4`，允许未压平的自然折层空气；v1/v2/v3发布保持原规则。旧 `shirt_stacking` / `trousers_stacking` 按 canonical ID 保留历史加载，默认列表与编号4/5只指向新折叠任务；新任务已绑定独立候选发布，不继承旧叠放资格。允许轻微褶皱、放置误差与接触抖动：毛巾摊平/对折 RMS ≤ 3 cm，折后放置中心误差 ≤ 5 cm；衣物在台面内刚体对齐后的形状 RMS ≤ 3 cm、中心误差 ≤ 15 cm，整件旋转不受固定朝向限制。服装层次及实际支撑比例均须 ≥ 85%，不能展开、严重拉伸或压扁成近似单层；新衣物profile仅允许完整交线与材料面积审计确认的微小局部穿插，较大异常仍阻断成功。完整数值与恢复条件见 [RULES.md](RULES.md)，以发布内的 [rules.py](rules.py) 与 [evaluation.py](evaluation.py) 为准。

全部五项都要求真实操作信用：至少一只手在同一连续原生碰撞几何接近段内净移动 1 cm，且任务衣物在该段位移至少 2.5 cm。先前被动下落不能由后来的手势追认。全部机器人碰撞形状与全部任务衣物分离后，重新计连续 0.5 s 稳定；Task1/2/4/5逐布逐帧原始/位置差分速度 P95 ≤ 0.08 m/s、窗口内所有节点漂移 ≤ 15 mm；Task3按上文显式v3的20毫秒位置速度与8毫米漂移判定。初态不计完成，重新接触或目标失效会撤销当时成功。

新Task4/5每项只有一个动态衣物（内部兼容ID为 `towel`），从展开状态开始，不能以已折成品替代初态或加入第二件衣物。资产由独立的单件折叠模块提供，目标参考只用于只读裁判，不能写回布点驱动仿真。投影所得的临时参考必须标记 `provisional_projection_not_qualified`，不能授予任务成功；最终须绑定可复核的折叠参考与资产来源。旧 [garments.py](garments.py) 的两件单衣片叠放模型和原证据保持历史身份。

当时Task4候选为 `flat_settled_v4`，Task5候选为 `outer_yaw70_v4`，均位于 `<开发存储>/datasets/fr3_sharpa_bench/development/laundry_single_garment_folding_20261023/releases/`。T恤采用项目自建的四开口平展双面缝合pattern，长裤为三开口缝合shell，材料厚度均为1.2 mm；两件初态均由无控制的真实PhysX沉降状态及原始拓扑重建。官方Newton T恤和失败候选保留，不能用其临时投影参考替代当时真实展开参考。

两个v4候选已接入编号4/5及canonical入口，并通过封存 `source_release15` 的CPU HTTP/Agent接口检查。两件v4又从 `source_runtime16` 完成真实原生HTTP、Agent绑定、双腕微动/返回/保持及五路RGB-D检查；合计1,142个原生帧和116个控制评分与冻结裁判精确回放一致，worker/server均退出0、模型调用0。证据见本轮原生接口资格（开发存储 `datasets/fr3_sharpa_bench/development/laundry_single_garment_folding_20261023/v4_native_interface_qualification01.json`）；它与IK、工程折叠和普通策略资格分开，不继承较早v2候选成绩。v4的scene、initial state与所选旧物理版本字节一致，变更只涉及冻结规则；T恤使用原1.2 mm/.45摩擦资产，长裤使用同厚度/.45摩擦、朝向70°的资产。首个长裤原生HTTP尝试在加载前因跨NumPy的P95汇总差异失败，记录保留；诊断确认真实节点逐点相同，仅派生速度汇总相差3.725×10⁻⁹ m/s。修复只兼容派生汇总舍入，节点、来源哈希和真实物理阈值保持严格。该加载诊断不代表仿真或完整折叠通过。旧v2接口归档为 `v2_native_interface_qualification01.json`，不迁移到新v4。

当时双臂折叠demo（开发存储 `demo/interface/scenes/laundry_tasks_20261021/single_garment_folding_20261023/index.html`）采用T恤 `fold_flat_shirt02` 和长裤 `fold_trousers10`：真实机器人运动配合明确声明的局部附件、局部夹区滚转，长裤另有局部同侧手/夹区碰撞过滤，解除附件并退手后恢复，最终3.66秒全部关闭。两条实际原生轨迹按新冻结v4做独立几何/释放/稳定重评，各自最后251帧全部通过形状、材料网格、交叉尺寸及70个原生机器人碰撞体分离检查，布点位置/速度写入为0。完整来源与资格边界见工程折叠几何证据（开发存储 `datasets/fr3_sharpa_bench/development/laundry_single_garment_folding_20261023/engineered_folding_geometry_v4_qualification01.json`）。

| 工程轨迹 | 末态RMS / P95 | 末窗最低正确折层 | 台面支撑 |
|---|---|---|---|
| T恤 `fold_flat_shirt02` | 15.07 / 34.86 mm | 85.6455% | 98.25% |
| 长裤 `fold_trousers10` | 13.83 / 22.14 mm | 88.8744% | 100% |

T恤本次折层裕量较小，不宣称多次运行鲁棒率。原v2/v3报告保持原结论，真实控制mask完整保留；普通robot_mode禁止这些附件，因此这份工程证据不转计普通机器人回合或模型摩擦抓取成功。两条全原生帧交叉审计均未触发严重异常门限，但中间有可恢复的软阈值超限。长裤首折一个受控腰口节点短时低于台面，最深7.691 mm；一条局部边37帧超过1.5倍、最高1.5309倍，未触及10 mm/2倍硬门限，在最终无控制稳定窗前恢复。不能写成全程软阈值全部通过；最终稳定窗口全部合格。

当时实际可运行版本以目录的冻结 release 指针为准；Bench 与 Agent 必须选择同一版本及源码。CPU 规则反例、原生初始化/接口微动、完整手布操作和材料物理资格分别记录。合成成功 fixture 或接口检查通过不代表机器人已完成抓取、对折或叠放；新五项发布的具体证据应绑定各自记录，不能继承下方历史工程成绩。

### 历史 v4 Bench 与 Agent 启动

两端均使用冻结源码 `source_runtime16`。该快照绑定 Task4 `shirt_folding/flat_settled_v4`、Task5 `trousers_folding/outer_yaw70_v4`，不能与旧 v2/v3 服务混用；前三项仍保留各自 `dual_v1`。下面以 Task4 为例，Task5 同时把服务的 canonical ID/版本改为 `trousers_folding`/`outer_yaw70_v4`，Agent 改为 `task5`。一个服务实例只发布所选任务。

Bench 终端设置本次空闲 GPU 的 UUID 后启动：

```bash
LAUNDRY_DATA=<开发存储>/datasets/fr3_sharpa_bench/development/laundry_single_garment_folding_20261023
LAUNDRY_SOURCE="$LAUNDRY_DATA/source_runtime16"
LAUNDRY_RUN=<开发存储>/datasets/fr3_sharpa_bench/agent_bench/laundry/$(date +%Y%m%d_%H%M%S)
LAUNDRY_PORT=8765
: "${LAUNDRY_GPU_UUID:?请设置本次空闲 GPU UUID}"
cd "$LAUNDRY_SOURCE"
PYTHONDONTWRITEBYTECODE=1 <开发存储>/interface/.venv/bin/python -B \
  -m scripts.maintenance.freeze_project_sources verify "$LAUNDRY_SOURCE"
env -u ISAAC_PYTHON_LAUNCHER \
  PYTHONPATH="$LAUNDRY_SOURCE" PYTHONDONTWRITEBYTECODE=1 \
  BENCH_TASK_RELEASE="$LAUNDRY_DATA/releases/shirt_folding/flat_settled_v4" \
  BENCH_AGENT_DATA_DIR="$LAUNDRY_RUN/episodes" BENCH_DEVICE=cuda:0 \
  CUDA_VISIBLE_DEVICES="$LAUNDRY_GPU_UUID" ISAAC_BENCH_SINGLE_CUDA_DEVICE=1 \
  ISAAC_ALLOW_UNVALIDATED_DRIVER=1 \
  ISAAC_BENCH_EXPERIENCE="$LAUNDRY_SOURCE/bench/apps/workroom.headless.kit" \
  <开发存储>/interface/.venv/bin/python -B -m uvicorn bench.runtime.server:app \
  --host 127.0.0.1 --port "$LAUNDRY_PORT"
```

Host 使用项目 CPU Python，原生 worker 自动调用同一冻结源的 `scripts/isaac_python.sh` 进入共享 Isaac Sim 6.1.0。driver override 沿用本次验证配置。服务收到创建请求才启动物理，HTTP health 不等于原生场景已就绪。

Agent 在另一个终端使用现有外部凭据配置。以下是模型回合入口；本次接口检查没有调用模型，也没有授予普通策略折叠成功：

```bash
LAUNDRY_SOURCE=<开发存储>/datasets/fr3_sharpa_bench/development/laundry_single_garment_folding_20261023/source_runtime16
LAUNDRY_AGENT_RUN=<开发存储>/datasets/fr3_sharpa_bench/agent_runs/laundry/$(date +%Y%m%d_%H%M%S)
LAUNDRY_PORT=8765
cd "$LAUNDRY_SOURCE"
PYTHONPATH="$LAUNDRY_SOURCE" <开发存储>/interface/.venv/bin/python -B \
  -m agent.model_client.function_runner \
  --project-root "$LAUNDRY_SOURCE" --bench-url "http://127.0.0.1:$LAUNDRY_PORT" \
  --credentials <开发存储>/interface/configs/agent_bench/local_gateway_credentials.json \
  --family laundry_room --task task4 --sim-budget none \
  --http-timeout 900 --output "$LAUNDRY_AGENT_RUN"
```

Agent 自动校验目录与服务的 release SHA，使用同一双臂控制器。普通回合只发关节/手部几何目标；演示中的显式局部附件仍属于独立工程入口，不能通过普通 Agent 动作悄悄启用。

### 历史叠放方案的2026-10-23验证结果

以下Task4/5对应已折衣物叠放；不是新单件T恤/长裤折叠的验证结果。

交付记录（开发存储 `datasets/fr3_sharpa_bench/development/laundry_five_tasks_20261023/qualification.json`）绑定最终五个发布、源码、CPU 回归、IK、原生 HTTP／Agent 和逐帧裁判回放。真实效果图（开发存储 `demo/interface/scenes/laundry_tasks_20261021/five_tasks_20261023/index.html`）并排展示每项初始和末帧相机图。

五项各运行 1.06 s／2,120 个原生步，保存 2,121 帧；全部 10,605 帧及 270 个控制周期评分与冻结裁判严格回放一致，五路 RGB-D、双腕微动／返回／半秒保持、幂等和 Agent 检查通过，worker／server 正常退出。原生观察器记录的自交及布间交叉均为零。毛巾使用 `source_runtime02`，衣物使用 `source_runtime03`；前三项的公开静态目标投影另由 CPU HTTP／Agent 检查补验，不伪称旧原生运行已采用后来代码。

上衣、长裤最后半秒的 1,001 帧均满足实用形状、层次、速度、漂移和接触条件。上衣最大形状 RMS 2.904 mm、最低正确层次 99.962%、最大节点漂移 1.491 mm；长裤分别为 3.563 mm、100%、3.704 mm。严格峰值诊断仍完整保留。对折记录有一条平滑法线回退警告，最终图像正常；`source_runtime03` 已修复 USD 同值法线更新通知，后两项未再出现法线警告。

最终离散工作空间检查覆盖 134 组双臂／268 个单臂端点，全部 IK 收敛；最小环境包围盒间距 7.509 mm、最小双机器人包围盒间距 5.496 mm。该检查不代表连续轨迹避障或实际夹持成功。当时资格为场景、裁判、接口、离散可达性和被动稳定性；完整自主抓取／折叠／叠放策略与材料物理标定尚未验收。

### 历史五项工程操作视频（Task4/5为叠放尝试）

播放五项合集与单项视频（开发存储 `demo/interface/scenes/laundry_tasks_20261021/demo_20261023/package01/index.html`）。前三项复用已核验的历史原生毛巾录像；上衣和长裤使用当时衣物资产另建 robot-free 工程发布，执行两处局部多层抓持、搬运、放置和完全释放。新衣物每项模拟 10.5 秒，8.5 秒时解除全部附件，再保留 2 秒被动过程。录像没有自由布点位置/速度写入，不代表双臂机器人策略成功。

单片保留原始帧率和正常速度；24 FPS 合集只重复已有帧，不插值生成动作。持续说明条位于原图外。来源、逐帧时序和哈希见视频目录的 `manifest.json`；新录制源码、失败/中止候选和末窗检查保存在 `<开发存储>/datasets/fr3_sharpa_bench/development/laundry_demo_20261023`。长裤末半秒的形状、对齐、完整释放和漂移通过，但 raw/pose 速度 P95 为 0.1804/0.1943 m/s，高于 0.08 m/s 稳定门槛；如实保留接触余振及失败结果，不标任务成功。末窗检查与完整流碰撞审计分别记录，视频本身不授予任务或物理资格。

上衣片段是完整叠放尝试，未通过折层保持和稳定验收：末帧形状 RMS 34.36 mm，正确层次约 15.99%，末半秒 raw/pose 速度 P95 0.4523/0.5236 m/s，最大漂移 20.72 mm。外轮廓仍可辨认不能代替折层验证；该工程轨迹尚需修复。两项原片、合集字幕、网页和末窗报告保留失败信息，不能把被动接口验收转记为完整搬运成功。

工程录制入口为 `scripts.validation.laundry_cloth_interactions`，衣物必须显式指定 `--garment-profile folded_garment_leading_patches_v1 --initialization author-only`，并选择独立 robot-free release；禁止把双臂 release 中的机器人静默省略。普通五任务 Bench/Agent 发布继续使用上方双臂入口。

### 历史叠放发布与复现

目录选择前三项 `dual_v1`、上衣 `dual_v3`、长裤 `dual_v2`。上衣资产 v3 去除了领口处约 0.034 mm 的极短网格边，保留衣片外轮廓、折痕、名义质量和 1.6 mm 厚度；最短边约为 0.45 mm。上衣取物位置向桌内平移 10 cm，以便两只真实手指到达布面。服装规则 v2 以材料厚度判断自然压实后的层次，旧发布保持原判据。

任务发现沿用通用 Agent 命令：

```bash
python -m agent.model_client.scene_tasks --family laundry_room
```

实际调用 `agent.model_client.function_runner` 时用 `--family laundry_room --task task1` 至 `task5` 选择任务，并使用匹配源码启动的 Bench 服务。五项都提供世界坐标系下的公开静态目标中心；动态布点和原生裁判证据不进入 Agent 观测。

独立的原生接口测试使用 `scripts.validation.public_http_smoke`，从对应完整源码快照启动。示例（输出必须是未存在的新目录）：

```bash
cd <开发存储>/datasets/fr3_sharpa_bench/development/laundry_five_tasks_20261023/source_runtime03
PYTHONPATH=. <开发存储>/interface/.venv/bin/python -B -m scripts.validation.public_http_smoke run \
  --release <开发存储>/datasets/fr3_sharpa_bench/development/laundry_five_tasks_20261023/releases/shirt_stacking/dual_v3 \
  --task-id shirt_stacking --agent-family laundry_room --hold-duration-s .5 \
  --gpu-uuid GPU-0e25a0f7-5cfd-4039-a657-466bda03de44 \
  --output <开发存储>/datasets/fr3_sharpa_bench/development/laundry_reproduction_new \
  --server-startup-timeout-s 600 --http-timeout-s 900 --agent-query-timeout-s 900
```

该脚本验证真实双腕接口微动、返回、静置、幂等、五路 RGB-D 与 Agent 适配，零模型调用。它不执行完整抓取或叠放策略。冻结裁判回放应使用原生 Isaac 所带的 Python/NumPy 环境；跨环境末位浮点差不能通过改分数掩盖。

### 2026-10-21 Task2 / Task3 双臂安装记录（历史）

该轮 `towel_folding` 和 `towel_stacking` 新增 `robot_v1`：一个 Agent 在普通 `/episodes` 会话中控制两台 FR3 和官方左、右 SHARPA，共 58 DOF。当时 Task1 `towel_flattening` 使用 `practical_v2` 工程入口。旧工程发布及其原生操作证据均保持原样，不能把工程抓点成功计为机器人抓取成功。

两臂沿折衣岛南北侧相对安装：left 基座 `(0.75, 0.15, 0.90)m`、yaw `−π/2`；right 基座 `(0.75, −1.25, 0.90)m`、yaw `+π/2`。六个支架碰撞体存放在共享资产 `task_assets/laundry_room/robot_installation/opposed_left_right_v1`。复用机械 Task10 的官方资产、58 关节待机姿态、驱动和控制器；仅调整各臂安装变换，毛巾初态与布料物理配置保持原值。三个场景相机加左右腕相机提供五路 RGB-D。

机器人入口明确采用 `robot_joint_targets`。命名布料使用 `author-only` 初始化，保留原生预热结果，不重写节点位置或速度，不建立隐含抓持附件。冻结 `native_observer.py` 在每个物理步观察真实布节点、速度和原生 cooked robot collision shapes，记录几何接近及完全分离；这些是几何证据，不是接触力测量。操作信用要求至少一只手在同一连续近布时段净移动 1 cm，且该段毛巾位移至少 2.5 cm；不能追认先前被动下落。最终全部机器人碰撞体与全部毛巾分离后才计稳定时间，不强制固定双手动作顺序。

IK 检查以实际中指端点接近布角，覆盖两任务接近、翻折／搬运、放置和撤退的离散双臂姿态；不会把高处腕位可达等同于手已能触及毛巾。报告区分离散 IK、实际形状静态净空、原生接口微动和完整操作资格。完整抓取／对折／叠放策略尚未随本次入口安装验证。

### 2026-10-21 毛巾规则与工程验证记录（历史）

以下几何容差同时用于工程模式和新机器人模式；历史工程记录不迁移机器人操作信用。

该轮公开规则和冻结裁判采用 `laundry_practical_task.v2`。按用户要求，以可靠完成整理、放置和释放为目标，允许软体仿真的轻微褶皱、局部接触抖动与小接触误差。旧发布及严格结果保持原样；无规则版本字段的历史任务仍按旧规则加载。

| 成功判断 | 实用容差 |
|---|---|
| 摊平 | 对应位置 RMS ≤ 3 cm，最差点 ≤ 7 cm；展开面积 ≥ 90%，残余高度范围 ≤ 2.5 cm |
| 对折 | 两半对应 RMS ≤ 3 cm，最差点 ≤ 6 cm；中心偏差 ≤ 5 cm，折后尺寸偏差 ≤ 5 cm；层次正确比例 ≥ 85% |
| 叠放 | 两条均保持折形，中心偏差 ≤ 5 cm；实际布面几何支撑比例 ≥ 85%，不能悬空或压成同一层 |
| 释放与稳定 | 全部抓点解除，连续 0.5 s；逐布、逐帧节点速度 P95 ≤ 8 cm/s，同时检查相邻步位置差；最近 0.5 s 内所有节点相对窗口起点漂移 ≤ 1.5 cm |
| 接触与恢复 | 稳定窗口允许 ≤ 2 mm 厚度包络压入、≤ 1.5 倍局部边长，实际布面交叉须已消失；整理中的小异常恢复后可重新计时 |

这些容差是工程规则先验。明显崩坏仍停止：压入 > 1 cm、局部边长 > 2 倍，或同帧自交/布间交叉对数合计 > 8；证据缺失、隐藏固定点、全布控制不能得分。初态不计完成，仍须有实际操作和至少 2.5 cm 位移。成功后继续观测时，重抓、滑走、折形失效会撤销当时成功；完整记录以实际末窗为准。

严格物理审计仍按旧限值独立报告，包括 1 mm 压入、1.25 倍边长和最大节点速度；它不再替代新版任务成功判断。正式规则见 [rules.py](rules.py)，裁判见 [evaluation.py](evaluation.py)。三项 `practical_v2` 分别沿用已经验证的摊平 v1、对折 v3、真实折形叠放 `rss_native_fold01` 的场景、物理、资产和初态，只修改规则与裁判。

三项已用已有原生记录完整回放新版冻结裁判，均通过：摊平 24,001 帧、对折 31,001 帧、叠放 29,001 帧，总计 84,003 帧。末半秒全部释放，最大节点漂移分别为 0、1.097、0.308 mm；未重新运行物理仿真。432 项相关 CPU 回归及 61 项公开规则检查通过，两组有重叠。新快照 `rules_practical_20261021/source02` 的实际 HTTP 发现接口已核对三项规则、哈希和 Agent 发布选择，仍明确拒绝普通机器人 episode。完整来源和结果见规则验证记录（开发存储 `datasets/fr3_sharpa_bench/development/laundry_tasks_20261021/rules_practical_20261021/qualification.json`）。叠放派生资产会在加载时复验原折叠证据，该项公开读取检查约 45 秒，当时仍有加载开销。

每条为 60 × 40 厘米、108 克，名义面密度 450 克/平方米、物理厚度 3 毫米。PhysX 表面厚度与 3 毫米可见壳分别声明；可见壳不替代物理厚度。法线与粗糙度纹理使用保留 CC0 来源信息的 Poly Haven terry cloth。尺寸、材料与求解器数值都是仿真先验；实际接触、层距和折形需原生证据确认。初态 U 折是离散网格近似，不能宣称精确解析了 3 毫米圆弯。

旧工程发布的能力是 `task_inputs` 和 `engineering_cloth_probe`。本轮按用户授权，脚本可驱动最多两个世界空间直径不超过 75 毫米的局部抓区；每区可同时夹住上下两层，每个相连材料片在静止网格中的直径也不超过 75 毫米，每条布控制点不超过 10%。未驱动部分仍是活布，验收前释放全部约束。裁判要求完整逐子步状态、显式控制掩码、重力与约束审计、自交/布间交叉/厚度穿透证据。目标满足后还需保持 0.5 秒的无控制稳定，初态、单张成功截图或零速度写入不能代替这些证据。成功仅表示这一控制模式下的工程任务完成，`physics_accepted` 和机器人抓取资格保持独立。

旧工程发布未提供机器人安装与手布接触验证；其 `capabilities.robot_episodes=false` 会明确拒绝，不把直接布点驱动伪装成机器人关节动作。任务专属发布不计入普通机器人任务数量。

旧工程发布的发现接口 `/tasks` 返回 `task_kind=engineering`、`robot_episodes=false`、`runnable=false` 和工程入口；选中该任务的 `/interfaces` 明确标记普通关节与函数动作不可用。新 Agent `function_runner` 在 Bench / 模型启动前拒绝，旧 Agent `runner` 在读取公开任务后、控制器发现和创建 episode 前拒绝，提示使用 `scripts.validation.laundry_cloth_interactions`。服务端仍以 HTTP 403 阻止普通 episode；这些公开接口不提供可写布料顶点。实际服务与两个 Agent 的检查见入口边界记录（开发存储 `datasets/fr3_sharpa_bench/development/laundry_tasks_20261021/discovery_boundary_review01.json`）。

从完整项目冻结快照运行以下 Python 接口，生成全新版本，不覆盖历史文件：

```python
from scenes.workrooms.laundry_room.assets import prepare_assets
from scenes.workrooms.laundry_room.release import prepare_task_release

prepare_assets(task_id, '<开发存储>/assets/task_assets/laundry_room/<task>/<version>')
prepare_task_release(task_id,
    '<开发存储>/datasets/fr3_sharpa_bench/development/laundry_tasks_20261021/releases/<task>/<version>',
    asset_manifest='<开发存储>/assets/task_assets/laundry_room/<task>/<version>/manifest.json',
    physics_options=reviewed_native_options)
```

叠放初态可显式取自已完成的原生对折终态。`prepare_assets('towel_stacking', ..., source_fold_native_report=..., source_fold_audit=..., cells_xy=..., physics_options=...)` 默认仍要求原严格审计全部通过。只有额外提供 `source_fold_practical_review=...` 时，才接受 `laundry_released_geometry_p95_speed.v1` 的已接受复核：复核必须绑定同一原生记录、完整逐帧物理审计、发布、资产与源码哈希。生成前重新核对全部帧审计覆盖，并从原始数组复算末 0.5 秒每帧的释放和几何、原始/位置差分速度 P95 ≤ 35 毫米/秒及所有节点漂移 ≤ 6 毫米；原严格峰值和失败结论保留在 `laundry_settled_fold_reset.v2` 来源记录里。

该入口要求网格拓扑、材料和 `physics_options` 与原对折发布完全相同，不自动重采样。只对终态节点作 XY 平移，Z 不变；不旋转、缩放或修形，也不复制大状态档案。新叠放初态没有操作分或物理资格，仍须独立原生运行和完整审计。加载派生资产时会再次验证上述来源与节点，缺少或失效的实用复核不会退回其他初态。

发布包固定场景、任务、初态、公开规则、物性先验、裁判源代码及外部资产哈希。准备成功只表示输入完整，manifest 仍为 `prepared_unvalidated`。随后用 `bench.isaacsim.release.task_inputs` / `load_evaluator` 读取同一版本；工程脚本和独立复核必须使用发布中的裁判。目录指针只在版本生成并核对后写入 `catalog_definition.RELEASES`。

CPU 测试只核对任务契约、反例、连续性和发布防篡改；不把合成顶点数组称为原生布料验证。原生脚本入口为 `scripts.validation.laundry_cloth_interactions`，具体结果及完整记录在共享数据目录中按版本保存。

以下命令从完整冻结的 `source_cloth_task08` 复现 `native04` 的细网格对折配置；该记录已通过实用复核，新的运行仍需自己的完整审计。正式任务必须同时指定相匹配的 `--case` 与 `--release`，并使用全新输出目录。这里显式选择 `--initialization author-only`：保留两步原生预热后的状态，不再次写入节点位置或速度；脚本默认仍为历史 `tensor-reset` 路径。历史 `native03` 使用 `source_cloth_task07` 和圆弧路径，其原始启动记录（开发存储 `datasets/fr3_sharpa_bench/development/laundry_tasks_20261021/native03_towel_folding_receipt.json`）保持不变。

```bash
cd <开发存储>/datasets/fr3_sharpa_bench/development/laundry_tasks_20261021/source_cloth_task08
CUDA_VISIBLE_DEVICES=GPU-f223f9f2-92da-e16a-ed9f-9704eb7e4b37 \
ISAAC_ALLOW_UNVALIDATED_DRIVER=1 ISAAC_BENCH_SINGLE_CUDA_DEVICE=1 \
./scripts/isaac_python.sh -m scripts.validation.laundry_cloth_interactions \
  --case towel_folding \
  --release <开发存储>/datasets/fr3_sharpa_bench/development/laundry_tasks_20261021/releases/towel_folding/v3 \
  --initialization author-only \
  --fold-edge-alignment --fold-grip-span-m .04 .04 \
  --fold-trajectory bezier --fold-peak-height-m .16 \
  --media-fps 2 \
  --output <开发存储>/datasets/fr3_sharpa_bench/development/laundry_tasks_20261021/repro_fold_v3_NEW \
  --media-output <开发存储>/demo/interface/scenes/laundry_tasks_20261021/repro_fold_v3_NEW
```

`--fold-trajectory bezier --fold-peak-height-m .16` 显式选择实验性 Bézier 候选；默认仍为 `arc`。`--passive-after-release-s` 可显式指定释放后记录时长，默认 6 秒。轨迹选择和记录时长均不改变发布的材料、评分门槛或末窗要求，进行中的实验不计通过。

新的完整流审计还独立验收记录末尾连续 0.5 秒：每个原生子步都须目标形状合格、全部释放，所有节点的原始速度与位置差分速度不超过 35 毫米/秒，窗口内位移不超过 6 毫米。旧冻结裁判的早期成功锁存仅保留为操作历史；最终接受同时要求完整物理流通过和真实末窗通过。几何函数与数值门槛仍取自同一冻结裁判。此检查收紧了完整记录验收，没有降低标准；历史冻结源、发布和报告保持原样，新审计写入新目录。

用户随后明确选择“实用稳定”：允许小幅接触抖动和局部工程抓持，要求折形、放置和释放可靠。独立的 `laundry_released_geometry_p95_speed.v1` 实用复核因此只把末窗的两个速度统计量改为**每条毛巾、每一帧分别计算的节点速度 P95**，上限仍为 35 毫米/秒；原始速度和相邻原生步位置差分速度均需满足。连续 0.5 秒的全部释放、每帧冻结目标几何、所有节点不超过 6 毫米的位移，以及完整原生流的穿透、交叉和拉伸审计继续保留。复核绑定原生记录、冻结源码、发布和完整审计的哈希，并验证实际操作历史，不给初态或被动重置诊断操作分。结果单独写为 `practical_operation_accepted`，同时保留严格裁判的失败与最大节点速度；不改旧发布、旧裁判或机器人资格。

这项折中借鉴了 [RoboSimStudio 的公开实现](https://github.com/robosimstudio/robosimstudio/tree/b001cc7e10a83577261aac4471fa45e0ced6d415)对材料、操作路径和静止判据分别处理的做法。其毛巾示例使用 Newton VBD 双表面加体积网格；本项目继续使用已固定版本的 PhysX 表面布料，未安装或切换仿真器。其最终 P95 检查也不等于这里连续半秒、完整几何审计的验收。代码阅读与来源固定在研究记录（开发存储 `datasets/fr3_sharpa_bench/development/laundry_tasks_20261021/failure_review_20261021_local01/robosimstudio_source_review01.json`），未声称在本机复现其演示。

截至 2026-10-21，已完成的证据如下。帧数包含不计分的初态；早期四次运行的发布、源码、原始状态和审计哈希见证据索引（开发存储 `datasets/fr3_sharpa_bench/development/laundry_tasks_20261021/delivery_evidence_table01.json`），新增候选单独链接原始记录。

| 任务 / 发布 | 完整原生记录 | 审计结论 | 已证实的结果 |
|---|---|---|---|
| 摊平 `v1` | `native01`，12 秒 / 24,001 帧 | 通过，含独立末窗（开发存储 `datasets/fr3_sharpa_bench/development/laundry_tasks_20261021/audit02_towel_flattening/report.json`） | 6.54 秒首次达到目标；11.5–12.0 秒全部释放、目标形状与稳定均通过，末态对应位置 RMS 1.13 毫米。仅该工程控制模式下的摊平操作通过。 |
| 对折 `v1` | `native01`，13.5 秒 / 27,001 帧 | 失败（开发存储 `datasets/fr3_sharpa_bench/development/laundry_tasks_20261021/audit01_towel_folding/report.json`） | 6.311 秒首次厚度重叠超限，全流最大 2.472 毫米；末态未保持合格折形与稳定。 |
| 对折 `v2` | `native02`，15.5 秒 / 31,001 帧 | 失败（开发存储 `datasets/fr3_sharpa_bench/development/laundry_tasks_20261021/audit02_towel_folding/report.json`） | 9.4235 秒首次厚度重叠超限，全流最大 2.314 毫米；末态仍有明显卷皱，不能生成合格叠放初态。 |
| 对折 `v3`，60 × 40 网格 | `native03`，15.5 秒 / 31,001 帧 | 全流物理几何通过，末窗失败（开发存储 `datasets/fr3_sharpa_bench/development/laundry_tasks_20261021/audit03_towel_folding/report.json`） | 全部帧已检查，最大厚度重叠 0.848 毫米、最大边拉伸 1.091 倍；末态两半对应位置 RMS 35.97 毫米、高度跨度 61.46 毫米。末 0.5 秒已全部释放，但原始节点速度峰值 1.269 米/秒、位移 7.88 毫米，折形与稳定均失败。 |
| 叠放 `v1` | `native02`，14.5 秒 / 29,001 帧 | 失败（开发存储 `datasets/fr3_sharpa_bench/development/laundry_tasks_20261021/audit02_towel_stacking/report.json`） | 第 1 步最大边拉伸约 2.795 倍，已经违反物理门槛；后续记录仍完整审计，不计任务通过。 |
| 对折 `v3`，Bézier | `native04`，15.5 秒 / 31,001 帧 | 全流物理几何通过、严格速度失败（开发存储 `datasets/fr3_sharpa_bench/development/laundry_tasks_20261021/audit04_towel_folding/report.json`）；实用复核通过（开发存储 `datasets/fr3_sharpa_bench/development/laundry_tasks_20261021/rss_repair01/practical_fold04/report.json`） | 全流最大厚度重叠 0.152 毫米、边拉伸 1.094 倍。末半秒全部释放、每帧折形合格；末态对应位置 RMS 4.35 毫米，所有节点最大漂移 1.10 毫米。逐帧 P95 原始/差分速度峰值为 23.36/22.89 毫米/秒；原始最大节点速度仍为 167.17 毫米/秒，未宣称严格通过。 |
| 叠放 `v1`，`author-only` | `native03`，14.5 秒 / 29,001 帧 | 完整审计失败（开发存储 `datasets/fr3_sharpa_bench/development/laundry_tasks_20261021/audit03_towel_stacking/report.json`） | 1.958 秒抬升期间已出现厚度重叠超限，早于 2 秒开始水平搬运；上层末态对应位置 RMS 53.17 毫米，末半秒最大速度 1.730 米/秒、漂移 11.34 毫米。不计工程任务通过。 |

`native01_towel_stacking` 在启动阶段取消，保留记录但不计完整尝试。对折 `v1/v2` 与叠放 `native02` 的失败报告早于独立末窗门；其完整流物理失败已经足以拒绝，历史报告未被改写。摊平有严格完整通过证据；对折 `native04` 有独立实用通过证据，严格速度门仍失败。对折改进同时涉及路径与时间参数化，不能据这两次记录把收益单独归因于某一个参数。叠放的高抬升失败和新的低位搬运对照分别保留，不据进行中的结果更新发布指针或资格。

历史续接核对见恢复记录（开发存储 `datasets/fr3_sharpa_bench/development/laundry_tasks_20261021/recovery_20261021/handoff_status01.json`）。`native04_towel_folding` 是 `source_cloth_task08` 的 Bézier 候选；原生进程与 `run_audit03_serial.py` 在 agent 停止后仍继续运行，记录中的 PID、进度仅代表核对时刻。这些记录的审计现已收取，结果在上表；历史 PID 不代表当时运行状态。

已找回 `e7128cf` 的显式叠放抓区补丁，相关 266 项 CPU 回归检查通过（开发存储 `datasets/fr3_sharpa_bench/development/laundry_tasks_20261021/recovery_20261021/cpu_checks.json`）。`--stack-grip-span-m .04 .04` 按材料坐标定义两处局部抓区，省略时保留历史节点集合。`--stack-trajectory low-clearance` 在桌面上方低位搬运，显式 `--stack-grip-layout crease-edge` 将两个抓区放在相连折边两端，保持未抓部分和下方毛巾自由。真实折形允许两抓区初始 X 略有不同，两者仍使用相同平移与升降，不整形布料。

| 叠放对照 | 完整审计与实用复核 | 结果 |
|---|---|---|
| `native_stack_low01`，低位、对角抓区、30 × 20 网格 | 完整审计失败（开发存储 `datasets/fr3_sharpa_bench/development/laundry_tasks_20261021/rss_repair01/audit_stack_low01/report.json`） | 5.475 秒首次厚度侵入超限；末态上层对应 RMS 43.10 mm、漂移 15.99 mm，出现堆皱。 |
| `native_stack_crease01`，低位、双折边抓区、30 × 20 网格 | 修正检查器后的完整审计（开发存储 `datasets/fr3_sharpa_bench/development/laundry_tasks_20261021/rss_repair01/audit_stack_crease02/report.json`）；实用复核拒绝（开发存储 `datasets/fr3_sharpa_bench/development/laundry_tasks_20261021/rss_repair01/practical_stack_crease01/report.json`） | 29,001 帧物理几何全过，最大厚度侵入 0.828 mm、拉伸 1.074 倍。末窗完全释放且每帧目标形态通过；上层 RMS 2.16 mm、漂移 0.716 mm。原始/差分逐帧 P95 峰值 52.37/61.96 mm/s，超过 35 mm/s，不能计实用通过。 |
| `native_stack_settled_fine02`，真实折形派生、60 × 40 网格、低位双折边抓区 | 全流物理几何通过（开发存储 `datasets/fr3_sharpa_bench/development/laundry_tasks_20261021/rss_repair01/audit_stack_settled_fine02/report.json`）；实用复核通过（开发存储 `datasets/fr3_sharpa_bench/development/laundry_tasks_20261021/rss_repair01/practical_stack_settled_fine02/report.json`） | 29,001 帧零自交、零布间交叉，最大厚度侵入 0.893 mm、拉伸 1.038 倍。末半秒全部释放且每帧目标形态通过；上层 RMS 3.96 mm、漂移 0.308 mm，原始/差分逐帧 P95 峰值 24.33/26.69 mm/s，低于 35 mm/s。严格最大速度仍为 181.08/194.89 mm/s，原严格失败保留。 |

摊平现有严格通过证据；对折 `native04` 与真实折形细网格叠放 `native_stack_settled_fine02` 有独立实用通过证据。叠放的新组合同时改变初态、网格与抓区节点采样，不能把改善单独归因于细网格。视频：对折（开发存储 `demo/interface/scenes/laundry_tasks_20261021/native04_towel_folding/task.mp4`） · 叠放（开发存储 `demo/interface/scenes/laundry_tasks_20261021/rss_repair01/native_stack_settled_fine02/task.mp4`）。完整来源、失败对照、验收与检查记录见本轮证据索引（开发存储 `datasets/fr3_sharpa_bench/development/laundry_tasks_20261021/rss_repair01/repair_evidence02.json`）。

旧 `audit_stack_crease01` 的两个自交命中均由共享拓扑端点的浮点误差引起。修复只排除非平行线段已知共享端点的重复求交，保留同一三角对其余边及共面分支，不扩大容差或跳过整对三角；20 项 CUDA 回归检查通过。新全流复审写入 `audit_stack_crease02`，旧报告保留。被动接触偏移、求解频次和 settling 参数对照未显示足够收益，本轮未更改默认物性。

任务及修复已从 `agent/laundry-recovery-20261021` 通过快进合入主目录 `main`（代码提交 `194470d`），原开发分支和冻结快照保留。真实折形资产由 `source_rss_repair05` 生成，运行使用 `source_rss_repair06`；两者布料作者、运行与裁判代码相同。`source05/native_stack_settled_fine01` 曾因冻结默认清单遗漏 `workroom.headless.kit` 而在物理启动前失败，该记录保留；新默认清单已包含该配置，116 项冻结/派生回归通过，`source06` 的新进程正常启动。代码接入、输入准备、原生运行和操作验收分别记账。

细网格真实折形叠放使用以下已通过实用验收的显式配置；运行记录为 `rss_repair01/native_stack_settled_fine02`，其完整审计与实用复核见上表。历史 `v1` 发布仅表示工程任务输入，换发布目录不会自动选择复位和抓持参数，因此此处给出完整复现命令。每次复现须使用新的输出目录：

```bash
cd <开发存储>/datasets/fr3_sharpa_bench/development/laundry_tasks_20261021/source_rss_repair06
CUDA_VISIBLE_DEVICES=GPU-f223f9f2-92da-e16a-ed9f-9704eb7e4b37 \
ISAAC_ALLOW_UNVALIDATED_DRIVER=1 ISAAC_BENCH_SINGLE_CUDA_DEVICE=1 \
./scripts/isaac_python.sh -m scripts.validation.laundry_cloth_interactions \
  --case towel_stacking \
  --release <开发存储>/datasets/fr3_sharpa_bench/development/laundry_tasks_20261021/releases/towel_stacking/rss_native_fold01 \
  --initialization author-only \
  --stack-trajectory low-clearance --stack-grip-span-m .04 .04 \
  --stack-grip-layout crease-edge --media-fps 2 \
  --output <开发存储>/datasets/fr3_sharpa_bench/development/laundry_tasks_20261021/repro_stack_native_fold_NEW \
  --media-output <开发存储>/demo/interface/scenes/laundry_tasks_20261021/repro_stack_native_fold_NEW
```

派生输入加载会完整重验原折叠档案与末窗，增加启动耗时。上面的旧源码与旧发布命令用于重现历史严格/实用复核；运行新规则须使用包含 `laundry_practical_task.v2` 的完整新快照和对应 `practical_v2` 发布，仍需显式传入已验证的初始化、路径和抓区参数。`practical_v2` 规则发布不会修改工程脚本的历史动作默认值；普通机器人入口由另行冻结的 `robot_v1` 提供。

另有 `source_cloth_task04` 的无附件复位对照（开发存储 `datasets/fr3_sharpa_bench/development/laundry_tasks_20261021/reset_probe01_comparison_v2.json`）：同一发布和配置下，仅保留原生预热状态的分支在预热及随后 100 步中最大边拉伸不超过 1.051 倍；再次写入初始节点位置、速度的分支首步约 2.795 倍、峰值速度约 58.2 米/秒。两次预热位置逐字节相同，速度最大差约 2.24×10⁻¹³ 米/秒。该对照支持把复位路径作为大瞬态的因素继续调查；它没有证明内部求解器机制，也没有完成叠放、完整物理资格或释放稳定验收。后续材料、复位与轨迹候选仍需各自原生运行和完整审计，进行中的候选不计入通过结果。
