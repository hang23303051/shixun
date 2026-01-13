"""
Ref4D简化真实模式 - 直接读取已有evalkit结果

此模块提供简化的真实评测模式：
- 不需要运行evalkit脚本
- 直接读取已有的评测结果CSV
- 适合项目演示和作业报告
"""

import csv
import logging
from pathlib import Path
from typing import Dict, Optional
from django.conf import settings

logger = logging.getLogger(__name__)


class Ref4DSimpleReader:
    """Ref4D简化结果读取器"""
    
    def __init__(self, results_dir=None):
        """
        初始化结果读取器
        
        Args:
            results_dir: 评测结果目录,默认backend/media/evalkit_results
        """
        if results_dir is None:
            self.results_dir = Path(settings.BASE_DIR) / 'media' / 'evalkit_results'
        else:
            self.results_dir = Path(results_dir)
        
        # 确保目录存在
        self.results_dir.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"[Ref4DSimpleReader] 结果目录: {self.results_dir}")
    
    def get_model_scores(self, model_name: str) -> Optional[Dict[str, float]]:
        """
        读取指定模型的评测分数
        
        Args:
            model_name: 模型名称
            
        Returns:
            {
                'semantic': float,
                'motion': float, 
                'event': float,
                'world': float,
                'total': float
            } 或 None(未找到)
        """
        # 查找该模型的结果CSV
        csv_path = self.results_dir / f"{model_name}_scores.csv"
        
        if not csv_path.exists():
            # 尝试通用结果文件
            csv_path = self.results_dir / "ref4d_4d_scores.csv"
            if not csv_path.exists():
                logger.warning(f"[Ref4DSimpleReader] 未找到结果文件: {csv_path}")
                return None
        
        logger.info(f"[Ref4DSimpleReader] 读取结果: {csv_path}")
        
        # 解析CSV
        try:
            with csv_path.open('r', encoding='utf-8') as f:
                reader = csv.DictReader(f)
                for row in reader:
                    # 查找匹配的模型
                    row_model = row.get('modelname', row.get('model', '')).strip()
                    if row_model == model_name:
                        return self._parse_row(row)
            
            logger.warning(f"[Ref4DSimpleReader] CSV中未找到模型: {model_name}")
            return None
            
        except Exception as e:
            logger.error(f"[Ref4DSimpleReader] 读取CSV失败: {e}")
            return None
    
    def _parse_row(self, row: Dict[str, str]) -> Dict[str, float]:
        """解析CSV行数据"""
        try:
            semantic = float(row.get('semantic_score', row.get('semantic', 0)))
            motion = float(row.get('motion_score', row.get('motion', 0)))
            event = float(row.get('event_score', row.get('event', 0)))
            world_raw = float(row.get('world_score', row.get('world', 0)))
            
            # world可能是0-5量表,需要scaled
            if world_raw <= 5:
                world = round(world_raw * 20, 2)
            else:
                world = world_raw
            
            # 计算总分(如果CSV中没有)
            total = float(row.get('total_score', row.get('total', 0)))
            if total == 0:
                total = round((semantic + motion + event + world) / 4, 2)
            
            return {
                'semantic': semantic,
                'motion': motion,
                'event': event,
                'world': world,
                'total': total
            }
        except (ValueError, KeyError) as e:
            logger.error(f"[Ref4DSimpleReader] 解析分数失败: {e}")
            return {
                'semantic': 0.0,
                'motion': 0.0,
                'event': 0.0,
                'world': 0.0,
                'total': 0.0
            }
    
    def create_sample_results(self, model_names: list):
        """
        为演示创建示例评测结果
        
        Args:
            model_names: 模型名称列表
        """
        import random
        
        csv_path = self.results_dir / "ref4d_4d_scores.csv"
        
        logger.info(f"[Ref4DSimpleReader] 生成示例结果: {csv_path}")
        
        # 生成示例数据
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
                'count_sample_id': '1',  # 示例模式只有1个样本
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
        
        logger.info(f"[Ref4DSimpleReader] 示例结果已生成: {len(rows)}个模型")


# 全局实例
_simple_reader = None

def get_simple_reader() -> Ref4DSimpleReader:
    """获取简化结果读取器实例(单例)"""
    global _simple_reader
    if _simple_reader is None:
        _simple_reader = Ref4DSimpleReader()
    return _simple_reader
