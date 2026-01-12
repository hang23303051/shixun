"""
视频评分算法（模拟实现）
"""
import random
from typing import Dict


class VideoScoring:
    """视频评分器（模拟）"""
    
    @staticmethod
    def calculate_semantic_score(gen_video_path: str, ref_video_path: str) -> float:
        """计算语义一致性评分（模拟）"""
        return round(random.uniform(80, 95), 1)
    
    @staticmethod
    def calculate_temporal_score(gen_video_path: str, ref_video_path: str) -> float:
        """计算时序一致性评分（模拟）"""
        return round(random.uniform(75, 90), 1)
    
    @staticmethod
    def calculate_motion_score(gen_video_path: str, ref_video_path: str) -> float:
        """计算运动属性评分（模拟）"""
        return round(random.uniform(80, 92), 1)
    
    @staticmethod
    def calculate_reality_score(gen_video_path: str, ref_video_path: str) -> float:
        """计算真实性评分（模拟）"""
        return round(random.uniform(78, 88), 1)
    
    @classmethod
    def evaluate_model(cls, gendata_list, refdata_dict) -> Dict[str, float]:
        """评估整个模型（所有视频对）"""
        all_scores = {'semantic': [], 'temporal': [], 'motion': [], 'reality': []}
        
        for gendata in gendata_list:
            video_id = gendata.video_id
            
            # 模拟评分：即使没有参考视频也生成随机分数
            # 真实打分算法应该在这里进行视频特征提取和对比
            
            # 计算各维度评分（模拟）
            all_scores['semantic'].append(cls.calculate_semantic_score('', ''))
            all_scores['temporal'].append(cls.calculate_temporal_score('', ''))
            all_scores['motion'].append(cls.calculate_motion_score('', ''))
            all_scores['reality'].append(cls.calculate_reality_score('', ''))
        
        # 计算平均分
        avg_scores = {}
        for dimension in ['semantic', 'temporal', 'motion', 'reality']:
            if all_scores[dimension]:
                avg_scores[dimension] = round(sum(all_scores[dimension]) / len(all_scores[dimension]), 1)
            else:
                avg_scores[dimension] = 0.0
        
        avg_scores['total'] = round(sum(avg_scores.values()) / 4, 1)
        return avg_scores
