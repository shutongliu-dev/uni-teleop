# 厨房七项任务

七项任务共 **36 件动态任务物品**，覆盖载物端送、果蔬分拣、薄餐具收纳、锅盖配合、瓶罐与包装食品归位、餐后器具整理和实际接触擦拭。柔性海绵是一件物品，内部含28个原生物理部件。每个任务还加载 **12 件固定背景物品**：果酱、蜂蜜、油壶、盐磨、胡椒磨、咖啡罐、可可罐、香草罐、面粉盒、茶盒、折叠餐巾和蛋杯。它们有明确身份及碰撞，公开说明为非评分陈设，不冒充可操作物品。

沿用住宅厨房 `kitchen_galley_20261019_v2` 的柜体、电器、锅具、搁板、推车、植物和原开源扫描陈设。每项独立加载动态物品、必要夹具和附加陈设，不改写父环境。原始 FR3 + 右 SHARPA、组合版本、控制器与物理材料保持原配置。

| 编号 / ID | 动态物品 | 任务固定件 | 工作内容 |
|---|---|---|---|
| [Task1 `breakfast_serving`](tasks/breakfast_serving/README.md) | 5：托盘、餐盘、空杯、两块面包 | 0 | 源区配盘后真实带载端送，目标区放稳 |
| [Task2 `produce_transfer`](tasks/produce_transfer/README.md) | 6：三个番茄、三根短胡萝卜 | 3：砧板、两只分类盘 | 按类别装入浅盘，允许安全台面恢复 |
| [Task3 `utensil_sorting`](tasks/utensil_sorting/README.md) | 6：两勺、两叉、两把钝餐刀 | 2：源垫、三格盘 | 按类别收纳，柄朝向自由，允许同类搭叠 |
| [Task4 `pot_lid_placement`](tasks/pot_lid_placement/README.md) | 2：双耳空锅、独立锅盖 | 1：源垫 | 锅先放稳在冷炉位，再盖独立锅盖 |
| [Task5 `pantry_stocking`](tasks/pantry_stocking/README.md) | 8：油、醋、盐、胡椒、红椒粉、米盒、意面盒、罐头 | 4：源垫、三类收纳盒 | 按瓶装、调味品、干货归位；瓶罐直立，纸盒和罐头可平放 |
| [Task6 `table_clearing`](tasks/table_clearing/README.md) | 8：两盘、两碗、两只带柄杯、勺、叉 | 4：源垫、三类收纳盘 | 盘碗、杯具、餐具归位；真实盘碗堆叠可接受 |
| [Task7 `surface_wiping`](tasks/surface_wiping/README.md) | 1：带握持区弹性海绵 | 1：回收浅盘 | 三片污渍整体清除 80%、每片 60%，海绵真实接触滑动后归位 |

固定件共15件；上表不重复计入每场12件背景物品。编号用于识别和排列，尚未校准为难度等级。

## 历史：2026-10-23 物品尺寸与细节

36件动态物品按类型适度放大：早餐托盘10%，餐具与锅具15%，杯盘、碗、瓶罐和纸盒20%，胡萝卜25%，番茄与面包30%；海绵仅在平面放大20%，成为120×78×45mm、79.2g。质量按原声明有效密度更新，质心与完整惯量由新实体重新计算；网格顶点直接采用米制新尺寸，没有在动态刚体上挂缩放变换。

餐盘直径由140增至168mm、杯高80增至96mm、锅身直径180增至207mm、面包长50增至65mm。杯盘保留真实空腔，补充釉沿与杯唇，餐具增加拉丝细纹，纸盒补折痕。海绵的黄色泡棉与完整绿色擦层使用真实浅凹孔、上部防滑沟和专属材质；孔隙是视觉细节，物理仍为独立平底刚体，不声称泡沫压缩或吸水。

放大同步更新承托高度、容纳边界和目标示例。早餐两块面包的目标中心间距从52增至68mm；锅盖源垫从490增至510mm以容纳完整盖沿；餐具分类中的两把勺子按真实碗底与柄端支撑面刚性转约0.91°，不再从质心悬出支撑区的水平姿态开始。餐具托盘的不可见碰撞底面按20mm固定网格细分，原顶点、边界、闭合实体与外观不变；未改善静置的SDF768对照保留为历史，最终仍用384。调味品与餐后器具的源区内收，餐后刀叉柄朝机器人，目标托盘整体右移60mm，仍位于原工作台内。固定房间、机器人安装、29关节待命、共享控制、接触材料和公开评分阈值保持原配置。旧胡萝卜natural_rest字段保存原始静置构造记录；新几何、支撑点和惯量以dimension_changes及当前对象字段为准。

该轮尺寸修订选用Task1/4/5/6 household_v2、Task2 household_v1、Task3 household_v4、Task7 household_visual_v2；后续果蔬和锅具布局见下节，实际版本以各definition的哈希为准。该轮证据在尺寸与接入检查（开发存储 `datasets/fr3_sharpa_bench/development/kitchen_prop_scale_20261023/qualification.json`），图片见原生图册（开发存储 `demo/interface/scenes/kitchen_prop_scale_20261023/index.html`）。离散IK、实际几何净空、独立目标承托、工程擦拭和真实HTTP/Agent分别记录，不合并为机器人完成任务成绩。

该轮离散检查中159/159个参考点有IK解，72/72个高出物体80mm的接近／抬起点满足5mm实际几何净空。固定待命手型的15个近物体点未通过5mm净空检查，原失败保留；后续有限姿态搜索和布局修复另存证据，不覆盖旧记录。不据此宣称已验证抓取、携物或连续路径。该轮七项目标承托采用独立重置台架：早餐运行3秒，其余1.6秒，均检查最后连续0.4秒稳定；海绵材质版通过逐项物理身份核对绑定其既有接触／承托记录。

参考[RoboCasa独立海绵擦拭](https://github.com/robocasa/robocasa/blob/0c81ff9c39b771afc330349cbbe50be9776a74fa/robocasa/environments/kitchen/composite/sanitizing_surface/wipe_table.py)的抓取／接触／释放区分、[BEHAVIOR海绵尺寸](https://behavior.stanford.edu/knowledgebase/synsets/sponge.n.01.html)，以及[SPONGE柔性接触研究](https://arxiv.org/html/2303.14012v1)。当前没有导入其他引擎或柔性求解器；详细出处、模型差异和旧36件尺寸在研究记录（开发存储 `datasets/fr3_sharpa_bench/development/kitchen_prop_scale_20261023/research01/report.json`）。

## 2026-10-24 柔性海绵与速度修复

海绵采用 `elastic_cell_lattice_v2`：27个凸形材料单元与原4mm绿色擦层通过63个有限刚度的被动弹性连接组成整体，保持120×78×45mm、79.2g及原静止外形。原生GPU力弹簧的目标恒为零，不写入运行中的物体位姿或速度。35,401点多孔皮层从实际部件姿态生成独立可视网格。紧凑版消除重复皮层数组，任务文件从100MB降至2.24MB；v1/v2物理USD与依赖内容逐字节相同。

完整样本通过TGS／PGS无接触冲量、PGS校准弹簧秤及TGS原浅盘承托。0.02N·s冲量的动量相对误差低于百万分之一；承重0.776960N，名义重量0.776952N。动态压板产生最大1.3686mm压缩并完整回弹，真实接触载荷与承托增量误差0.36%；末段全部28部件的原始线／角速度均满足原门槛。回弹残差不代表实物材料精度，TGS测量架的绝对弹簧偏差不作为重量校准。19/19个离散IK与5mm净空点通过，不等于抓取或连续路径通过。

研究核对了[RoboSimStudio海绵实现](https://github.com/robosimstudio/robosimstudio/blob/eb3d46a839bf8d9b8df6e9c3f0205e141f5b3f7f/examples/02_soft/00_sponge.py)、Newton柔体示例及[PhysX力弹簧](https://docs.omniverse.nvidia.com/kit/docs/omni_physics/latest/dev_guide/rigid_bodies_articulations/joints.html)。不同引擎／耦合的例子不直接转移资格；固定运行库的连续FEM去掉附着后仍多传递约62.8%的重量，失败记录保留。当前为有真实受力与形变的离散弹性近似，参数仍是工程先验，不声明连续体标定、吸水或塑性。见研究记录（开发存储 `datasets/fr3_sharpa_bench/development/kitchen_flexible_velocity_fix_20261024/research/actionable_review01/report.json`）和完整原生压缩曲线（开发存储 `datasets/fr3_sharpa_bench/development/kitchen_flexible_velocity_fix_20261024/research/compression_curves01/compression_curves_v2.png`）。

Task3／Task6选择 `household_contacthybrid_v1`：勺子保留原开口勺碗的SDF与4个柄部凸块，叉子／刀保留原实体与开槽的精确凸块分解。柄部碰撞顶点使用512倍局部坐标与倒数子网格缩放，消除小凸块生成时的顶点损失，物理尺寸、独立视觉网格、质量、质心和惯量不变；替换的旧碰撞体明确停用。源垫／托盘只细分共面碰撞三角，不移动表面或改变材料。

两项任务各自源位、目标位四组GPU验证共12,800个原生步，保留12,804帧完整轨迹，最后连续0.4秒全部物体的原始线／角速度均满足原0.025m/s／0.25rad/s门槛。勺子最大末窗角速度0.134678rad/s；逐物体完整800区间的外部法向Z载荷与自重最大差约0.058%。没有裁剪、滤掉异常帧、按固定倍数修正载荷或调整共享控制器。两组真实HTTP／Agent完成四路RGB-D、只读IK、腕部微动、返回、保持和幂等，正常退出。旧SDF、220凸块及其速度／接触异常记录保留，不能作为当前版本的通过证据。

原生实际形状确认每勺5个碰撞体，全部12个柄部凸块的物理顶点与生成结果一致。稀疏SDF的全域“2倍请求网格间距”诊断在约5mm远场失败，原报告和exit1保留；独立碰撞边界复核使用每勺原有的72个近表面样点、24条正负法线夹界及53个材料／空腔／外部样点。近表面最大距离误差2.325µm，空腔符号与梯度检查通过，不把稀疏远场误差隐去或称为全场精度认证。

当前七项证据汇总见交付索引（开发存储 `datasets/fr3_sharpa_bench/development/kitchen_flexible_velocity_fix_20261024/qualification.json`）。新餐具56点工作空间检查与弹性海绵19点检查单独绑定对应物理资产，另四项沿用未变化发布的既有证据：合计159/159点IK、157/159固定待命手型净空、72/72高位接近净空。调味品罐与清台叉子的两个固定目标近接点仍未过5mm净空，未计为抓取或路径通过。每个原生运行保留自己的冻结源码，不将后续主分支整树声称为已原生验证；验证范围为物体动力学和普通接口，机器人完整任务与实物材料标定仍不在本轮资格内。

## 历史阶段：2026-10-24 布局修复与刚体复核

该阶段Task2选用 `household_access_v2`，将 `tomato_0` 的源位置x从0.20移至0.23m；Task4选用 `household_access_v3`，将源锅偏航角从90°改为−45°，改善握持入口。其余任务仍用上述尺寸版本，固定房间、原FR3/right SHARPA、质量材料、共享控制器和评分阈值不变。

当前七版本的离散工作空间汇总（开发存储 `datasets/fr3_sharpa_bench/development/kitchen_dynamics_repair_20261023/workspace_summary03/report.json`）核对159/159点IK、157/159固定待命手型净空及72/72高位接近净空。海绵19点从 `source19` 重新检查；果蔬／锅具32点保留其新布局 `source05` 证据，另108点按原发布及几何身份绑定，没有宣称一次新的全任务扫描。调味品罐与清台叉子的两个固定目标近接点仍未过5mm净空，不等于证明其他握法或合法放置不可达，也不构成连续抓取路径资格。

六个刚体任务的完整复核（开发存储 `datasets/fr3_sharpa_bench/development/kitchen_dynamics_repair_20261023/current_kitchen_matrix01/qualification.json`）绑定只读 `source15`：35件物体的目标承托共22,000个原生GPU步，逐帧检查真实质量、质心、惯量、支持与最后0.4秒稳定；六个真实HTTP／Agent检查均完成四路RGB-D、只读IK、微动、返回、保持和幂等，正常退出。末窗总外部法向Z分量与名义重量差不超过0.064%，但没有完整切向合力或全部物体材料系数的原生读回，不能称全力学标定。餐具及清台部分细长物品的原始瞬时角速度仍有短时超限；它们通过的是未修改的20ms位姿速度和姿态包络规则。

该阶段海绵入口保留120×78×45mm、79.2g的干态刚体近似。第七项独立复核（开发存储 `datasets/fr3_sharpa_bench/development/kitchen_dynamics_repair_20261023/current_rigid_wiping_plan01/qualification02.json`）绑定 `source19`，真实HTTP／Agent接口、1.6秒回收盘承托与6.6秒受限水平施力擦拭均完成。擦拭首块污渍覆盖100%、第二块5.882%、第三块0%，只是物体接触滑动验证，未完成全任务。独立台面与正式台面同高度和材质，但自动contact offsets未证明相同；最后0.2秒没有接触行，位姿与速度恒定，未读取原生休眠标志，也未将缺失行当成零载荷。

原生FEM候选已经具备真实压缩／回弹与只读皮层观测，但校准承重和无接触外力冲量检查不通过；减少网格、scene速度迭代上限及平移坐标原点均未使其合格。因此 `bonded_fem_v2` 和 `bonded_fem_coarse_v1` 仅作为显式诊断候选，不能继承正式海绵的接口或动力学资格。研究、失败轨迹与决定见海绵动力学决定（开发存储 `datasets/fr3_sharpa_bench/development/kitchen_dynamics_repair_20261023/sponge_dynamics_decision01/report.json`）。SPONGE使用旧Isaac Gym／FleX的FEM与专用接触接口，其演示不能直接证明本项目固定Isaac Sim 6.1／PhysX配置合格；未替换运行库、修改质量补偿误差或改变共享控制器。

全部记录、测试和最终代码版本见本轮总索引（开发存储 `datasets/fr3_sharpa_bench/development/kitchen_dynamics_repair_20261023/qualification.json`）。七个任务都已接普通Bench／Agent接口；目标承托和有界外力探针不替代机器人完整操作或全力学标定。

## 物品、布局和物理约定

原创 CC0 米制网格保留可用杯腔、碗腔、盘面、锅腔与开口把手，不用实心凸包封住容器。新瓶罐有不同肩颈、旋盖纹、标签色带和封边；纸盒有折顶与封口；罐头有卷边、拉环；大小盘、深浅碗、直筒杯和矮口茶杯具有不同形状。原托盘铆钉、面包芝麻与刻纹、果蔬蒂叶与根纹、勺面和叉齿继续保留。

除弹性海绵外，可动物品均为独立被动刚体；海绵采用被动弹性部件。质量、质心、惯量、碰撞和源／目标支撑写入资产元数据。瓶罐内容物密封并计入整体质量；不模拟液体、开盖、加热、切割、洗涤或食物变形。质量与摩擦为公开工程先验，未声称实物标定。图案细节允许只参与渲染，承托和容纳使用真实碰撞几何。

任务1、2、3、5、6、7使用吧台，机器人偏航−90°；任务4使用备菜台与冷灶，偏航+90°。基座固定在 `(.65, 0, .78)m`，原29关节待命状态不变。附加陈设在吧台外侧边缘，与中央保护工作区留有间隙；它们不作为评分容器或隐藏支撑。种子保留在合同中，当前初态是冻结合法供料布局，不暗中随机改变几何。

Task5目标盒通过深蓝、奶白、木色区分；Task6通过大奶白盘、蓝色杯盘、木质餐具盘区分。目标元数据给出对象说明、类别和目的地；实际Agent接口通过公开裁判指令说明可见目标，不暴露真值位姿、接触图或目标示例位姿。

## 规则与裁判

详细合同见[RULES.md](RULES.md)及各任务README。七项共用完整原生子步证据、真实承托、全部松手和共同稳定0.4秒。稳定速度使用每个0.5ms子步滚动的20ms位姿窗口，同时检查4mm／0.08rad稳定包络，接受小幅正常接触抖动。打开手指的命令本身不能替代松手证据。

- 早餐先在源区释放配盘0.3秒，再实际离台15mm、带载净移动0.50m；空盘先到终点再拼装不算端送。
- 锅先在指定冷炉位释放稳定0.3秒，再盖独立盖；锅中心允许35mm、盖偏心12mm，不规定耳柄转角。
- 分类任务按整个碰撞外形是否容纳及实际支撑判断，不按精确点位打分。
- 新任务允许有根、无环的同区域支撑链，盘碗可真实堆叠；类别错误、悬空、骑边或机器人扶住不能通过。
- 错放、重抓、扶正和安全台面恢复允许重试。Task5/6提供240秒仿真预算，原四项120秒，公开动作上限6000；超时／取消为未完成，证据缺失为unknown，落地和物理安全停止分别记录。

Task7 的污渍是可见、无碰撞的清洁单元；只有海绵擦面实际承载并沿面滑动才累计清洁，清洁状态只改变材质。允许推动、重抓、分段反复擦；海绵可压缩回弹，不模拟流体或化学去污。海绵初态就在回收盘内，但没有擦拭历史，不能获得初态完成信用。

初态仅建立基准，不给完成信用。裁判只读模拟器证据，不驱动物体，不附带任务专用抓取策略。

## Bench与Agent入口

七项均为普通 `/episodes`，共用现有公共函数。先启动与所选任务冻结release一致的bench，再在Agent使用 `--family kitchen --task task1` 至 `task7`，或上表的canonical ID。任务5、6、7应显式给Agent `--sim-budget none`；这会取消客户端的额外时限，由冻结裁判执行 Task5/6 的240秒或 Task7 的180秒任务时限；全局CLI默认预算仍为120秒，具体参数见[Agent入口](../../../agent/model_client/README.md)。

```bash
# 示例：Task5；替换为实际部署所用服务参数
BENCH_TASK_RELEASE=<开发存储>/datasets/fr3_sharpa_bench/isaacsim/releases/kitchen/pantry_stocking/household_v2 \
  .venv/bin/python -m uvicorn bench.runtime.server:app --host 127.0.0.1 --port 8000

# 查看某项冻结任务定义／供上层启动器选择
.venv/bin/python -m agent.model_client.scene_tasks --family kitchen
```

Agent家庭选择校验目录冻结发布；服务实例绑定一个release，不会因为改task参数就隐式切换场景。公开规则、初始化、资源哈希和只读裁判一起冻结；任务5/6使用v3规则，当前弹性Task7使用v6，历史刚体v4与FEM v5按原模块加载，原四项保留v2规则。

## 发布、验证和历史

共享资产：`<开发存储>/assets/task_parts/kitchen/<task>/<version>`；冻结发布：`<开发存储>/datasets/fr3_sharpa_bench/isaacsim/releases/kitchen/<task>/<version>`。

以下为2026-10-21历史：Task7当时使用资产v4／release v5，去掉此前六个文字／编号标识，将海绵、盘和三片污渍整体向台内移动4cm，盘外沿离近侧桌沿50mm；Task近景随之调整，overview不变。物品靠自然形状与颜色识别，公开描述用左／中／右，不设顺序要求。当前交付索引（开发存储 `datasets/fr3_sharpa_bench/development/kitchen_wiping_inset_20261021/qualification.json`）单独绑定新初态、接触探针和接口；14项CPU检查及原布局／新布局各8个参考腕端点检查通过。更深6/8cm的固定参考姿态失败记录保留，未改阈值或机器人配置。历史v3物物验证和v4标签图保留，不作为当前布局原生通过证据。完整手物操作仍交数采验证。

当前选择以各任务 `definition.py` 的实际路径与SHA256为准。历史六任务索引为2026-10-21交付证据（开发存储 `datasets/fr3_sharpa_bench/development/kitchen_six_tasks_20261021/qualification.json`），分别记录CPU合同、原生初态、独立目标承托和HTTP检查。`goal_support_*`是单独声明的目标重置台架，只验证尺寸、接触与静置，不是机器人完成任务。

2026-10-21的六项原生初态检查及六项目标承托检查均通过，每项运行1.6秒／3200个0.5ms物理步，并验证最后0.4秒的真实支撑与稳定。初态覆盖四路RGB-D，全部保持零完成信用、物理检查通过、worker退出0。Task5/6另外通过真实TCP HTTP的发现、RGB-D、只读IK、0.01rad关节微动／回程／保持及幂等重放。两项新任务的完整原生记录经独立重放复核；Task6确认小盘→大盘→收纳盘的真实支撑链。CPU、Agent接口回归和最终发布绑定检查的命令、JUnit与去重计数见证据索引。

六任务初态／目标原始图册（开发存储 `demo/interface/scenes/kitchen_six_tasks_20261021/index.html`）包含12张原生PNG及来源哈希。目标图属于独立重置台架；不会记成机器人整理成功。历史启动失败、早期预算配置测试失败及资源竞争导致的重复检查取消均保留，最终索引明确选择通过的运行。

准备manifest的 `prepared_unvalidated` / `physics_accepted=false` 保留原值，后续证据单独挂接。场景与接口通过不等于抓取、端送或完整机器人任务成功，模型调用与操作成绩单独记录。

此前四项资产v2、前三项release v2／锅盖v3，以及2026-10-19原始验证（开发存储 `datasets/fr3_sharpa_bench/development/kitchen_tasks_20261019/qualification.json`）均保留。旧四任务设计提案和桌面标签生成器已从主线退役，原实现保存在完整冻结源码（开发存储 `datasets/fr3_sharpa_bench/development/laundry_five_tasks_20261023/source_runtime03/scenes/workrooms/kitchen`）。现行任务保留各自的 `definition.py` 和 `task.py`，共用家族 `assets.py`、`rules.py`、`evaluation.py`；执行合同仍以发布中的冻结规则和裁判为准。
