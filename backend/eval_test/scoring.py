"""
视频评分算法

当前支持两种模式:
1. 模拟模式 (默认): 快速返回随机分数,用于开发测试
2. 真实模式: 调用Ref4D-VideoBench evalkit进行四维度评测

配置方式:
- 环境变量: REF4D_SCORING_MODE=SIMULATION 或 REAL
- 代码配置: VideoScoring(mode='REAL')
"""
import logging
from typing import Dict
from .ref4d_scoring import get_scorer, Ref4DScorer

logger = logging.getLogger(__name__)


class VideoScoring:
    """视频评分器 (Ref4D-VideoBench四维度评测)"""
    
    def __init__(self, mode='SIMULATION'):
        """
        初始化评分器
        
        Args:
            mode: 'SIMULATION'(模拟) 或 'REAL'(真实evalkit评测)
        """
        self.scorer = get_scorer(mode=mode)
    
    def evaluate_model(self, gendata_list, refdata_dict=None) -> Dict[str, float]:
        """
        评估整个模型(所有生成视频)
        
        Args:
            gendata_list: GenData查询集(模型生成的视频列表)
            refdata_dict: {video_id: RefData对象}(参考数据,当前实现不需要)
            
        Returns:
            {
                'semantic': float,   # 语义一致性 (0-100) [Ref4D: Semantic]
                'motion': float,     # 运动一致性 (0-100) [Ref4D: Motion]
                'temporal': float,   # 时序一致性 (0-100) [Ref4D: Event]
                'reality': float,    # 真实性 (0-100) [Ref4D: World]
                'total': float       # 总分 (0-100)
            }
            
            注: 字段名mapping: temporal=event, reality=world
        """
        if not gendata_list or gendata_list.count() == 0:
            logger.warning("[VideoScoring] 没有可评测的视频数据!")
            return {
                'semantic': 0.0,
                'motion': 0.0,
                'temporal': 0.0,
                'reality': 0.0,
                'total': 0.0
            }
        
        # 获取模型名称
        model_name = gendata_list.first().model_name if gendata_list.exists() else 'unknown'
        
        # 调用评分器
        scores = self.scorer.evaluate_model(model_name, gendata_list)
        
        logger.info(f"[VideoScoring] 评测完成: model={model_name}, total={scores['total']:.2f}")
        return scores

