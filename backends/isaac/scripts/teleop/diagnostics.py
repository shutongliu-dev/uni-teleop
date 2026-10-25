"""Console reasons for held teleoperation inputs, emitted on change or every 5 s."""
import json
import math


def joint_operator_status(mapper, health, feedback, contact_message):
    arm=getattr(mapper,'leader',mapper)
    source=health.get('leader',health)
    age=source.get('latest_age_s')
    lines=[f'主臂数据：{age*1000:.0f} 毫秒前' if age is not None else '主臂数据：尚未收到']
    issue=arm.issue or {}
    if issue.get('code')=='joint_saturated':
        for limit in issue['limits']:
            axis=limit['joint'].removeprefix('fr3_joint')
            q,lo,hi=[math.degrees(limit[k]) for k in ('target_rad','lower_rad','upper_rad')]
            lines.append(f'第 {axis} 轴限位：目标 {q:+.0f}°，范围 {lo:+.0f}°～{hi:+.0f}°')
        lines.append('限位轴停在边界，其余轴继续；转回范围内即可跟随。')
    elif issue.get('code')=='leader_stale':
        lines.append('主臂数据过期：仿真保持。数据恢复后点“开始 / 恢复跟随”。')
    elif issue:
        lines.append('仿真保持 / 同步中：'+issue.get('reason',''))
    else:
        following = ('仿真：直接关节跟随（不限角度和目标速度）' if getattr(arm,'unrestricted_arm',False)
                     else '仿真：正在限速跟随')
        lines.append(following if arm.armed or arm.alignment_pending else '仿真：已暂停')
    if 'manus' in health:
        glove_age=health['manus'].get('latest_age_s')
        lines.append('手套：无新数据，手指保持' if glove_age is None or glove_age>.15 else '手套：数据正常')
    if feedback:
        if feedback['enabled']:
            peak=max(abs(v) for v in feedback['current_ma'])
            lines.append(f'实体托举：开启，电流指令峰值 {peak:.0f} mA')
        else:
            lines.append('实体托举：关闭；扶稳后点“开启 / 恢复托举”。')
            reason=feedback.get('reason','')
            if 'speed' in reason:lines.append('原因：主臂转动速度超过设定值。')
            elif 'interval' in reason or 'stale' in reason or 'timed out' in reason:
                lines.append('原因：主臂控制或通信超时。')
    if contact_message:lines.append('接触提示：'+contact_message)
    return '\n'.join(lines)


def control_issues(mapper, guard):
    issues=[]
    arm=getattr(mapper,'issue',None)
    if arm:
        issues.append({'channel':'arm',**arm})
    for channel,reason in mapper.diagnostics.items():
        if channel=='arm' or channel.startswith('fr3_'):
            continue  # The structured arm issue already carries the cause.
        issues.append({'channel':channel,'code':'input_hold','reason':reason})
    if guard.trigger:
        issues.append({'channel':'physics','code':'contact_guard','reason':guard.trigger})
    return issues


class HoldReporter:
    def __init__(self):
        self.key=()
        self.last_ns=0

    def update(self, issues, *, now_ns):
        key=tuple((v['channel'],v['code'],v.get('joint'),v.get('motor_id')) for v in issues)
        if key!=self.key or (key and now_ns-self.last_ns>=5_000_000_000):
            print(json.dumps({'event':'teleop_hold' if issues else 'teleop_hold_cleared',
                'host_ns':now_ns,'issues':issues},ensure_ascii=False),flush=True)
            self.key,self.last_ns=key,now_ns
