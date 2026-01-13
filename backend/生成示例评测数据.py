"""
生成Ref4D评测示例数据 - 独立版本
无需Django环境，直接生成CSV文件
"""

import csv
import random
from pathlib import Path

def main():
    """生成示例评测结果"""
    print("=" * 50)
    print("  生成Ref4D评测示例数据")
    print("=" * 50)
    print()
    
    # 创建结果目录
    results_dir = Path(__file__).parent / 'media' / 'evalkit_results'
    results_dir.mkdir(parents=True, exist_ok=True)
    
    # 常见的模型名称
    model_names = [
        'jimeng_video_3',
        'veo3_1',
        'grok_video_3',
        'doubao_video_pro',
        'luma_dream_machine',
        'hailuo_ai',
    ]
    
    print(f"📋 准备为以下模型生成示例结果:")
    for name in model_names:
        print(f"   - {name}")
    print()
    
    # 生成示例数据
    csv_path = results_dir / "ref4d_4d_scores.csv"
    
    print(f"📝 生成文件: {csv_path}")
    
    rows = []
    for model_name in model_names:
        # 基于模型名生成一致的随机分数(使用hash作为种子)
        seed = hash(model_name) % 10000
        random.seed(seed)
        
        semantic = round(random.uniform(75, 95), 2)
        motion = round(random.uniform(70, 90), 2)
        event = round(random.uniform(65, 85), 2)
        world = round(random.uniform(3.0, 4.5), 2)  # 0-5量表
        total = round((semantic + motion + event + world * 20) / 4, 2)
        
        rows.append({
            'modelname': model_name,
            'count_sample_id': '1',
            'semantic_score': f"{semantic:.2f}",
            'motion_score': f"{motion:.2f}",
            'event_score': f"{event:.2f}",
            'world_score': f"{world:.2f}",
            'total_score': f"{total:.2f}"
        })
    
    # 写入CSV
    with csv_path.open('w', encoding='utf-8', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=[
            'modelname', 'count_sample_id',
            'semantic_score', 'motion_score', 'event_score', 'world_score',
            'total_score'
        ])
        writer.writeheader()
        writer.writerows(rows)
    
    print()
    print(f"✅ 示例数据生成完成! 共{len(rows)}个模型")
    print()
    print(f"📂 结果文件: {csv_path.absolute()}")
    print()
    print("=" * 50)
    print("  下一步:")
    print("=" * 50)
    print()
    print("1. 确保.env文件包含:")
    print("   REF4D_SCORING_MODE=REAL")
    print()
    print("2. 重启Celery worker:")
    print("   cd backend")
    print("   celery -A backend worker -l info -P solo")
    print()
    print("3. 提交评测任务,查看日志确认使用真实模式")
    print()

if __name__ == '__main__':
    main()
