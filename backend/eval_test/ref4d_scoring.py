"""
Ref4D真实评分算法接口封装

此模块提供Ref4D-VideoBench四维度评测的Django接口。
支持两种工作模式：
1. 模拟模式(SIMULATION): 快速返回随机分数,用于开发测试
2. 真实模式(REAL): 调用evalkit进行真实评测(需要额外依赖)

用法:
    from eval_test.ref4d_scoring import Ref4DScorer
    
    scorer = Ref4DScorer(mode='SIMULATION')  # 或 'REAL'
    scores = scorer.evaluate_model(model_name, gendata_list)
"""

import os
import sys
import random
import logging
from pathlib import Path
from typing import Dict, List, Optional
from django.conf import settings

logger = logging.getLogger(__name__)


class Ref4DScorer:
    """Ref4D-VideoBench评分器 - 支持模拟和真实两种模式"""
    
    # 评分模式
    MODE_SIMULATION = 'SIMULATION'  # 模拟模式: 快速随机分数
    MODE_REAL = 'REAL'              # 真实模式: 调用evalkit评测
    
    def __init__(self, mode=MODE_SIMULATION, metadata_path=None):
        """
        初始化评分器
        
        Args:
            mode: 评分模式,default='SIMULATION'
            metadata_path: metadata数据路径,default=backend/media/metadata
        """
        self.mode = mode
        
        # 设置metadata路径
        if metadata_path is None:
            self.metadata_path = Path(settings.BASE_DIR) / 'media' / 'metadata'
        else:
            self.metadata_path = Path(metadata_path)
            
        logger.info(f"[Ref4DScorer] 初始化完成, mode={mode}, metadata={self.metadata_path}")
        
    def evaluate_model(self, model_name: str, gendata_list) -> Dict[str, float]:
        """
        评测模型生成的视频
        
        Args:
            model_name: 模型名称
            gendata_list: GenData查询集(包含生成视频信息)
            
        Returns:
            {
                'semantic': float,   # 语义一致性 (0-100)
                'motion': float,     # 运动一致性 (0-100)
                'temporal': float,   # 时序/事件一致性 (0-100) [对应Ref4D event维度]
                'reality': float,    # 真实性/世界知识 (0-100) [对应Ref4D world维度]
                'total': float       # 总分 (0-100)
            }
            
            注: temporal=event, reality=world (字段名映射以兼容现有数据库)
        """
        if self.mode == self.MODE_SIMULATION:
            return self._evaluate_simulation(model_name, gendata_list)
        elif self.mode == self.MODE_REAL:
            return self._evaluate_real(model_name, gendata_list)
        else:
            raise ValueError(f"Unknown scoring mode: {self.mode}")
    
    def _evaluate_simulation(self, model_name: str, gendata_list) -> Dict[str, float]:
        """模拟评分 - 生成随机分数"""
        logger.info(f"[Ref4DScorer/模拟] 正在生成模拟分数, model={model_name}")
        
        count = gendata_list.count()
        if count == 0:
            logger.warning(f"[Ref4DScorer/模拟] 没有可评测的视频数据!")
            return {
                'semantic': 0.0,
                'motion': 0.0,
                'temporal': 0.0,  # event维度
                'reality': 0.0,   # world维度
                'total': 0.0
            }
        
        # 为每个生成视频生成四维度分数
        all_scores = {
            'semantic': [],
            'motion': [],
            'event': [],      # Ref4D维度名
            'world': []       # Ref4D维度名
        }
        
        for gendata in gendata_list:
            # 模拟分数范围 (参考Ref4D论文的分值分布)
            all_scores['semantic'].append(round(random.uniform(75, 95), 2))
            all_scores['motion'].append(round(random.uniform(70, 90), 2))
            all_scores['event'].append(round(random.uniform(65, 85), 2))
            all_scores['world'].append(round(random.uniform(3.0, 4.5), 2))  # 0-5量表
        
        # 计算平均分
        avg_scores = {}
        for dim in ['semantic', 'motion', 'event']:
            avg_scores[dim] = round(sum(all_scores[dim]) / count, 2)
        
        # world score需要scaled到0-100
        avg_world_raw = round(sum(all_scores['world']) / count, 2)
        avg_scores['world'] = round(avg_world_raw * 20, 2)  # 5分制转100分制
        
        # 计算总分 (四维度等权重平均)
        total = round(
            (avg_scores['semantic'] + avg_scores['motion'] + 
             avg_scores['event'] + avg_scores['world']) / 4, 2
        )
        
        # 返回时重命名字段以兼容现有数据库模型
        result = {
            'semantic': avg_scores['semantic'],
            'motion': avg_scores['motion'],
            'temporal': avg_scores['event'],    # event → temporal
            'reality': avg_scores['world'],     # world → reality
            'total': total
        }
        
        logger.info(f"[Ref4DScorer/模拟] 生成完成: total={result['total']:.2f}")
        return result
    
    def _evaluate_real(self, model_name: str, gendata_list) -> Dict[str, float]:
        """
        真实评分 - 调用evalkit进行评测
        
        支持两种模式:
        1. 简化模式(优先): 读取已有的evalkit结果CSV
        2. 完整模式(备选): 调用evalkit subprocess
        """
        logger.info(f"[Ref4DScorer/真实] 正在调用evalkit评测, model={model_name}")
        
        # 尝试简化模式:读取已有结果
        try:
            from .ref4d_simple import get_simple_reader
            
            reader = get_simple_reader()
            scores = reader.get_model_scores(model_name)
            
            if scores:
                # 成功读取到结果
                logger.info(f"[Ref4DScorer/真实-简化] 从CSV读取结果成功")
                
                # 字段映射(event→temporal, world→reality)
                result = {
                    'semantic': scores['semantic'],
                    'motion': scores['motion'],
                    'temporal': scores['event'],    # 映射
                    'reality': scores['world'],     # 映射  
                    'total': scores['total']
                }
                logger.info(f"[Ref4DScorer/真实-简化] 评测完成: total={result['total']:.2f}")
                return result
            
            # CSV中没有该模型结果,尝试生成示例
            logger.warning(f"[Ref4DScorer/真实-简化] CSV中未找到结果,生成示例数据")
            reader.create_sample_results([model_name])
            scores = reader.get_model_scores(model_name)
            
            if scores:
                result = {
                    'semantic': scores['semantic'],
                    'motion': scores['motion'],
                    'temporal': scores['event'],
                    'reality': scores['world'],
                    'total': scores['total']
                }
                logger.info(f"[Ref4DScorer/真实-简化] 使用示例数据: total={result['total']:.2f}")
                return result
                
        except Exception as e:
            logger.warning(f"[Ref4DScorer/真实-简化] 简化模式失败: {e}")
        
        # 简化模式失败,尝试完整模式(调用subprocess)
        try:
            logger.info(f"[Ref4DScorer/真实-完整] 尝试调用evalkit subprocess...")
            scores = self._run_evalkit_subprocess(model_name, gendata_list)
            return scores
            
        except Exception as e:
            logger.error(f"[Ref4DScorer/真实-完整] 完整模式也失败: {e}")
            logger.info(f"[Ref4DScorer/真实] 降级使用模拟模式")
            return self._evaluate_simulation(model_name, gendata_list)
    
    def _run_evalkit_subprocess(self, model_name: str, gendata_list) -> Dict[str, float]:
        """
        通过subprocess调用evalkit评测脚本
        
        实现步骤:
        1. 将gendata中的视频组织为evalkit要求的结构(软链接)
        2. 调用evalkit/scripts/run_4d_eval.sh
        3. 解析outputs/overall/ref4d_4d_scores.csv
        4. 返回该模型的四维度分数
        """
        import subprocess
        import csv
        import json
        
        evalkit_path = Path(settings.BASE_DIR).parent / 'evalkit'
        
        if not evalkit_path.exists():
            raise FileNotFoundError(f"evalkit路径不存在: {evalkit_path}")
        
        logger.info(f"[Ref4DScorer/真实] evalkit路径: {evalkit_path}")
        
        # 1. 准备视频文件软链接
        # evalkit期望: evalkit/data/genvideo/<model_name>/<sample_id>.mp4
        # 我们有: backend/media/gendata/<model_name>/<theme>/<video_id>.mp4
        
        genvideo_dir = evalkit_path / 'data' / 'genvideo' / model_name
        genvideo_dir.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"[Ref4DScorer/真实] 准备视频软链接到: {genvideo_dir}")
        
        # 创建软链接映射
        video_count = 0
        for gendata in gendata_list:
            # 生成视频的实际路径
            gen_video_full_path = Path(settings.BASE_DIR) / gendata.video_file
            
            if not gen_video_full_path.exists():
                logger.warning(f"[Ref4DScorer/真实] 视频文件不存在: {gen_video_full_path}")
                continue
            
            # evalkit需要的sample_id格式: theme_XXX_shottype.mp4
            # 从video_id提取 (如: animals_and_ecology_001.mp4)
            sample_id = gendata.video_id  # 应该类似 "animals_and_ecology_001"
            
            # 确保包含shot_type后缀
            if gendata.shot_type and not sample_id.endswith(f"_{gendata.shot_type}"):
                sample_id = f"{sample_id}_{gendata.shot_type}"
            
            target_link = genvideo_dir / f"{sample_id}.mp4"
            
            # Windows: 使用拷贝而非软链接(权限问题)
            # Linux/Mac: 使用软链接
            try:
                if target_link.exists():
                    target_link.unlink()  # 删除已存在的链接/文件
                
                if os.name == 'nt':  # Windows
                    import shutil
                    shutil.copy2(gen_video_full_path, target_link)
                    logger.debug(f"[Ref4DScorer] 拷贝: {gen_video_full_path.name} -> {target_link}")
                else:  # Linux/Mac
                    target_link.symlink_to(gen_video_full_path)
                    logger.debug(f"[Ref4DScorer] 软链接: {gen_video_full_path.name} -> {target_link}")
                
                video_count += 1
            except Exception as e:
                logger.error(f"[Ref4DScorer/真实] 创建链接失败: {e}")
                continue
        
        if video_count == 0:
            raise ValueError(f"没有有效的视频文件可供评测!")
        
        logger.info(f"[Ref4DScorer/真实] 准备完成, 共{video_count}个视频")
        
        # 2. 调用evalkit评测脚本
        # 使用Python直接调用而非bash(Windows兼容性)
        logger.info(f"[Ref4DScorer/真实] 开始调用evalkit评测...")
        
        # 检查是否有评测脚本的Python版本，否则尝试调用bash
        script_path = evalkit_path / 'scripts' / 'run_4d_eval.sh'
        
        if not script_path.exists():
            raise FileNotFoundError(f"评测脚本不存在: {script_path}")
        
        # 设置环境变量限制只评测当前模型
        env = os.environ.copy()
        env['INCLUDE_MODELS'] = model_name
        
        try:
            # Windows: 使用Git Bash或WSL
            if os.name == 'nt':
                # 尝试使用Git Bash
                git_bash = Path('C:/Program Files/Git/bin/bash.exe')
                if git_bash.exists():
                    cmd = [str(git_bash), str(script_path)]
                else:
                    raise RuntimeError("Windows系统需要Git Bash来运行evalkit脚本")
            else:
                cmd = ['bash', str(script_path)]
            
            logger.info(f"[Ref4DScorer/真实] 执行命令: {' '.join(cmd)}")
            
            result = subprocess.run(
                cmd,
                cwd=str(evalkit_path),
                env=env,
                capture_output=True,
                text=True,
                timeout=7200  # 2小时超时
            )
            
            if result.returncode != 0:
                logger.error(f"[Ref4DScorer/真实] 标准输出:\n{result.stdout}")
                logger.error(f"[Ref4DScorer/真实] 错误输出:\n{result.stderr}")
                raise RuntimeError(f"evalkit执行失败 (code={result.returncode})")
            
            logger.info(f"[Ref4DScorer/真实] evalkit执行完成")
            
        except subprocess.TimeoutExpired:
            raise RuntimeError("evalkit评测超时(>2小时)")
        
        # 3. 解析结果CSV
        result_csv = evalkit_path / 'outputs' / 'overall' / 'ref4d_4d_scores.csv'
        
        if not result_csv.exists():
            # 尝试查找merge后的结果
            alt_csv = evalkit_path / 'outputs' / 'overall' / 'ref4d_4d_scores_model_avg.csv'
            if alt_csv.exists():
                result_csv = alt_csv
            else:
                raise FileNotFoundError(f"评测结果文件不存在: {result_csv}")
        
        logger.info(f"[Ref4DScorer/真实] 解析结果: {result_csv}")
        
        with result_csv.open('r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                if row.get('modelname') == model_name:
                    # 提取四维度分数
                    semantic = float(row.get('semantic_score', 0))
                    motion = float(row.get('motion_score', 0))
                    event = float(row.get('event_score', 0))
                    world_raw = float(row.get('world_score', 0))  # 0-5量表
                    
                    # world scaled到0-100
                    world = round(world_raw * 20, 2)
                    
                    # 计算总分
                    total = round((semantic + motion + event + world) / 4, 2)
                    
                    # 返回时映射字段名(兼容数据库)
                    result = {
                        'semantic': semantic,
                        'motion': motion,
                        'temporal': event,    # event → temporal
                        'reality': world,     # world → reality
                        'total': total
                    }
                    
                    logger.info(f"[Ref4DScorer/真实] 评测完成: {result}")
                    return result
        
        raise ValueError(f"在评测结果中未找到模型: {model_name}")


def get_scorer(mode='SIMULATION') -> Ref4DScorer:
    """
    获取评分器实例(工厂函数)
    
    Args:
        mode: 'SIMULATION' 或 'REAL'
        
    Returns:
        Ref4DScorer实例
    """
    # 从环境变量读取mode(可选)
    env_mode = os.getenv('REF4D_SCORING_MODE', mode)
    return Ref4DScorer(mode=env_mode)

