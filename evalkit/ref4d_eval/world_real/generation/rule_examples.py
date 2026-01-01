# -*- coding: utf-8 -*-
"""
rule_examples.py
把可放入 Example 的“头部内容”集中管理，并提供渲染函数给主脚本调用。
主脚本用法：
    from rule_examples import (
        VIDEO_EXTS, TIME_SCALE, MAX_NUM_FRAMES, BLACKLIST,
        SIGNAL_LIST, SCHEMA, EXAMPLE_BANK_CORE, render_header_examples
    )
"""

import json
from typing import Dict, Any, List

# ---------------- 可放进 Example 的“常量/清单/Schema/示例库” ----------------

VIDEO_EXTS = {'.mp4', '.mkv', '.mov', '.avi', '.m4v', '.webm'}
TIME_SCALE = 0.1
MAX_NUM_FRAMES = 180

# 允许颜色相关“高阶合理性/一致性”判断，但禁止低阶描述
BLACKLIST = "exist,where,left-of,right-of,top-of,bottom-of,nearest,farthest,count"

SIGNAL_LIST: List[str] = [
"size_inconsistency(oX)","support(oA,oB)","containment(oA,oB)","balance(oX)","collision(oA,oB)","material_optics(oA)","post_collision_change(oX)","interpenetration(oA,oB)","continuity(oX)","frame_discontinuity(t)","trajectory_dir(oX,axis)","accel_sign(oX,axis)","rigid_transform(oX)","deformation(oX)","fracture(oX)","wheel_rolling(oX)","shadow_consistency(oX)","reflection_consistency(oX)","hue_histogram_jump(oX)","tilt_angle_threshold(oX,deg)","center_of_mass_over_base(oX)","friction_sliding(oA,on:oB)","elastic_bounce(oX)","slosh_response(oV,amp)","buoyancy_float(oA,in:oV)","compressibility(oX)","viscosity_effect(oV)","pour_direction(oV,axis)","leakage(oV,from:oContainer)","specular_highlight_consistency(oX)","cast_shadow_contact(oX,shadow)",
                          "grasp_stability(oHand,oObj)","slip(oA,on:oB)","tool_contact(oTool,oObj)","cuttable_progress(oObj)","pourable(oContainer)","openable(oContainer)","supportable(oObj,by:oSurface)","placeable(oObj,at:oTarget)",
                          "biological_motion(oA)","action_phase(oAct,phase)","step_order(prev:task,next:task)","state_change(oObj,from:state,to:state)","goal_placement(oObj,at:oTarget)","result_appearance(oObj)","handoff_between_hands(oObj)",
                          "tool_usage_correct(oTool,for:oTask)","spillage(oV)","finger_clearance(oHand,oTool)","blade_orientation_away(oTool,oHand)","hot_surface_contact(oObj,oSurface)","flame_proximity(oObj,dist)","prohibited_intersection(oA,oB)","smoke_direction_consistency(scene)","falling_hazard(oObj)"
]




SCHEMA: str = """
[
  {
    "id": "r1",
    "type": "Affordance|Task|Safety|Physics",
    "rule_text": "Write a complete, human-readable rule in English. Prefer an'During..., ...''...should...'sentence.",
    "anchors": [{"t0": 0.0, "t1": 2.0}],
    "required_signals": ["<from signal list>"],
    "difficulty": {"steps": 1, "span_sec": 0.0, "occlusion": "low"}
  }
]
"""



# ---- few-shot 示例库（你原来 EXAMPLE_BANK 的内容迁移至此） ----
EXAMPLE_BANK_CORE: str = r"""
[示例库（仅作风格与结构参考；不得伪造新对象/时间）]

- 重力一致性（Physics）
F片段：
{"entities":[{"id":"o1","name":"ball","spans":[[1.0,2.0]]}],
 "states":[{"eid":"o1","var":"phase","t":1.1,"val":"free_drop"}]}
期望输出（JSON数组片段）：
[
  {"id":"r1","type":"Physics",
   "rule_text":"if o1 is in free-drop then its vertical motion should trend downward with non-positive acceleration",
   "anchors":[{"eid":"o1","t0":1.1,"t1":1.8}],
   "required_signals":["trajectory_dir(o1,axis:\"-y\")","accel_sign(o1,axis:\"-y\")"],
   "difficulty":{"steps":2,"span_sec":0.7,"occlusion":"low"}}
]

- 惯性连续性（Physics）
F片段：
{"entities":[{"id":"o2","name":"cart","spans":[[3.0,5.0]]}],
 "states":[{"eid":"o2","var":"motion","t":3.1,"val":"moving"}]}
期望输出：
[
  {"id":"r1","type":"Physics",
   "rule_text":"o2 is moving on a flat surface, so its velocity should change smoothly and not abruptly to zero or reverse",
   "anchors":[{"eid":"o2","t0":3.1,"t1":4.8}],
   "required_signals":["continuity(o2)"],
   "difficulty":{"steps":1,"span_sec":1.7,"occlusion":"med"}}
]

- 相互穿透（Physics）
F片段：
{"entities":[{"id":"o3","name":"cup","spans":[[6.0,7.0]]},
             {"id":"o4","name":"table","spans":[[0.0,10.0]]}],
 "relations":[{"subject":"o3","predicate":"on","object":"o4","spans":[[6.0,6.8]]}]}
期望输出：
[
  {"id":"r1","type":"Physics",
   "rule_text":"o3 is supported by o4, and their geometries should not interpenetrate during contact",
   "anchors":[{"eid":"o3","t0":6.1,"t1":6.7}],
   "required_signals":["support(o3,o4)","interpenetration(o3,o4)"],
   "difficulty":{"steps":2,"span_sec":0.6,"occlusion":"low"}}
]

- 颜色合理性（人/光照）（Safety）
F片段：
{"entities":[{"id":"h1","name":"person","spans":[[8.0,10.0]]}]}
期望输出：
[
  {"id":"r1","type":"Safety",
   "rule_text":"h1 is a human and skin tone should be within plausible human ranges under the current scene illumination",
   "anchors":[{"eid":"h1","t0":8.2,"t1":9.6}],
   "required_signals":["skin_tone_plausibility(h1)","illumination_color_cast(scene)"],
   "difficulty":{"steps":2,"span_sec":1.4,"occlusion":"med"}}
]

- 烟雾的物理一致性
F片段：
{"entities":[{"id":"smoke","name":"smoke","spans":[[10.0,12.0]]}],
 "states":[{"eid":"smoke","var":"motion","t":10.1,"val":"ascending"}]}
期望输出：
[
  {"id":"r1","type":"Physics",
   "rule_text":"smoke rising from the surface should move upward with decreasing speed as it encounters air resistance",
   "anchors":[{"eid":"smoke","t0":10.1,"t1":12.0}],
   "required_signals":["trajectory_dir(smoke,axis:\"upward\")","accel_sign(smoke,axis:\"-y\")"],
   "difficulty":{"steps":2,"span_sec":1.9,"occlusion":"low"}}
]
""".strip()


# ---------------- 渲染函数：把“头部内容”打包成一段 Example 文本 ----------------

def render_header_examples(
    parser,                     # argparse.ArgumentParser（主脚本构造后传入）
    args,                       # argparse.Namespace（主脚本 parse_args 后传入）
    extra_consts: Dict[str, Any] | None = None
) -> str:
    """
    生成“头部示例文本”：CLI 帮助 + 常量摘要 + SIGNAL_LIST + SCHEMA。
    - 常量摘要 = 本模块的常量 ∪ 主脚本传入的 extra_consts（如与运行参数相关的派生量）
    """
    cli_help = parser.format_help()

    # 基础常量（本模块）
    base_consts: Dict[str, Any] = {
        "VIDEO_EXTS": sorted(list(VIDEO_EXTS)),
        "TIME_SCALE": TIME_SCALE,
        "MAX_NUM_FRAMES": MAX_NUM_FRAMES,
        "BLACKLIST": BLACKLIST,
    }

    if extra_consts:
        base_consts.update(extra_consts)

    parts: List[str] = []
    parts.append("[运行参数 / CLI 帮助]")
    parts.append("```text\n" + cli_help.rstrip() + "\n```")
    parts.append("[常量与默认值（摘要）]")
    parts.append("```json\n" + json.dumps(base_consts, ensure_ascii=False, indent=2) + "\n```")
    parts.append("[信号词表 SIGNAL_LIST]")
    parts.append("```json\n" + json.dumps(SIGNAL_LIST, ensure_ascii=False, indent=2) + "\n```")
    parts.append("[输出 Schema]")
    parts.append("```json\n" + SCHEMA + "\n```")

    return "\n\n".join(parts)
