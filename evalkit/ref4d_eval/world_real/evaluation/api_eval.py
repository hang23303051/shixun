# -*- coding: utf-8 -*-
from evaluate_bank import VideoEvaluator

# 1. 初始化 (只跑一次)
evaluator = VideoEvaluator(
    model_path_or_id="/root/autodl-tmp/aiv/models/openbmb__MiniCPM-V-4_5",
    device="cuda",
    verbose=False
)

# 2. 你的业务逻辑循环
my_tasks = [
    ("/path/video1.mp4", {"items": [...]}), # 直接传字典
    ("/path/video2.mp4", "/path/bank2.json") # 或者传路径
]

for vid, bank in my_tasks:
    # 如果 bank 是路径，先加载
    if isinstance(bank, str):
        import json
        with open(bank, 'r') as f: bank = json.load(f)
    
    # 3. 核心调用：得到 1-5 分
    result = evaluator.evaluate_single(vid, bank)
    
    print(f"视频: {vid} -> 得分: {result['band']}")
    # result['avg_score'] 是 0-100 的精细分
    # result['details'] 是每道题的详情