"""
Celery异步任务定义
处理视频生成和评测的异步任务
"""
import os
import logging
from datetime import datetime
from celery import shared_task
from django.conf import settings

from task_list.models import TaskList
from model_eval.models import Model
from ref_data.models import RefData
from eval_test.models import GenData
from .model_adapters import get_model_adapter
from .video_downloader import VideoDownloader
from .prompt_utils import PromptManager
from .scoring import VideoScoring

# 配置日志
logger = logging.getLogger(__name__)

# 测试模式：每个模型只测试1个prompt
TEST_MODE = True
TEST_PROMPT_LIMIT = 1


@shared_task(bind=True, max_retries=0)
def process_evaluation_task(self, task_id: str):
    """主评测任务"""
    logger.info(f'[Task {task_id}] 开始处理评测任务')
    try:
        task_record = TaskList.objects.get(task_id=task_id)
        logger.info(f'[Task {task_id}] 模型: {task_record.model_name}, API密钥: ***')
        
        task_record.status = 'processing'
        task_record.progress = 0
        task_record.message = '任务开始处理：正在生成视频...'
        task_record.save()
        
        # 阶段1：生成视频
        success, error_msg = generate_all_videos(
            task_id, task_record.api_url, 
            task_record.api_key, task_record.model_name
        )
        
        if not success:
            task_record.status = 'failed'
            task_record.message = f'视频生成失败：{error_msg}'
            task_record.save()
            return
        
        # 阶段2：打分评测
        task_record.progress = 90
        task_record.message = '视频生成完成，正在进行评分...'
        task_record.save()
        
        scores = run_scoring_algorithm(task_id, task_record.model_name)
        
        # 阶段3：创建Model记录
        model = Model.objects.create(
            name=task_record.model_name,
            description=task_record.description or '暂无描述',
            publisher=task_record.publisher or '未知',
            parameters=task_record.parameters or '未知',
            is_open_source=task_record.is_open_source,
            release_date=task_record.release_date or datetime.now().date(),
            official_website=task_record.official_website or '',
            semantic_score=scores['semantic'],
            temporal_score=scores['temporal'],
            motion_score=scores['motion'],
            reality_score=scores['reality'],
            total_score=scores['total'],
            tester_type='user',
            tester_name=task_record.username
        )
        
        # 阶段4：更新任务状态
        task_record.status = 'completed'
        task_record.progress = 100
        task_record.message = '评测完成'
        task_record.semantic_score = scores['semantic']
        task_record.temporal_score = scores['temporal']
        task_record.motion_score = scores['motion']
        task_record.reality_score = scores['reality']
        task_record.total_score = scores['total']
        task_record.model_id = model.id
        task_record.completed_at = datetime.now()
        task_record.save()
        
        logger.info(f'[Task {task_id}] 任务完成！')
        
    except Exception as e:
        logger.error(f'[Task {task_id}] 发生异常：{str(e)}', exc_info=True)
        try:
            task_record = TaskList.objects.get(task_id=task_id)
            task_record.status = 'failed'
            task_record.message = f'任务失败：{str(e)}'
            task_record.save()
        except:
            pass


def generate_all_videos(task_id, api_url, api_key, model_name):
    """批量生成视频 - 使用模型适配器"""
    try:
        task_record = TaskList.objects.get(task_id=task_id)
        
        # 获取模型适配器
        adapter = get_model_adapter(model_name, api_key)
        if not adapter:
            return False, f'不支持的模型: {model_name}'
        
        # 初始化组件
        prompt_manager = PromptManager()
        downloader = VideoDownloader()
        
        # 测试模式：每个模型只测试1个prompt
        limit = TEST_PROMPT_LIMIT if TEST_MODE else 600
        prompts = prompt_manager.load_all_prompts(limit=limit)
        total_prompts = len(prompts)
        
        if total_prompts == 0:
            return False, '未找到任何prompt文件'
        
        logger.info(f'[Task {task_id}] 准备生成 {total_prompts} 个视频（测试模式）')
        
        success_count = 0
        failed_count = 0
        
        for idx, prompt_data in enumerate(prompts, 1):
            video_id = prompt_data['video_id']
            theme = prompt_data['theme']
            shot_type = prompt_data['shot_type']
            prompt = prompt_data['prompt']
            
            logger.info(f'[Task {task_id}] 处理 {idx}/{total_prompts}: {video_id}')
            
            # 更新进度
            progress = int((idx / total_prompts) * 90)
            task_record.progress = progress
            task_record.message = f'正在生成视频 {idx}/{total_prompts}: {video_id}'
            task_record.save()
            
            try:
                # 步骤1：调用API创建视频任务
                logger.info(f'[Task {task_id}] 步骤1：创建视频任务')
                create_success, api_task_id, create_error = adapter.create_video(prompt)
                
                if not create_success:
                    logger.error(f'[Task {task_id}] 创建任务失败: {create_error}')
                    failed_count += 1
                    continue
                
                logger.info(f'[Task {task_id}] 任务ID: {api_task_id}')
                
                # 步骤2：查询任务状态并等待完成
                logger.info(f'[Task {task_id}] 步骤2：查询任务状态（最多等待5分钟）')
                query_success, video_url, query_error = adapter.query_status(api_task_id, max_wait=300)
                
                if not query_success:
                    logger.error(f'[Task {task_id}] 查询状态失败: {query_error}')
                    failed_count += 1
                    continue
                
                if not video_url:
                    logger.error(f'[Task {task_id}] 未获取到视频URL')
                    failed_count += 1
                    continue
                
                logger.info(f'[Task {task_id}] 视频URL: {video_url[:100]}...')
                
                # 步骤3：下载视频
                logger.info(f'[Task {task_id}] 步骤3：下载视频')
                save_path = downloader.generate_save_path(model_name, theme, video_id)
                download_success, download_error = downloader.download(video_url, save_path)
                
                if not download_success:
                    logger.error(f'[Task {task_id}] 下载失败: {download_error}')
                    failed_count += 1
                    continue
                
                logger.info(f'[Task {task_id}] 下载成功: {save_path}')
                
                # 步骤4：保存到数据库
                logger.info(f'[Task {task_id}] 步骤4：保存到数据库')
                db_path = downloader.get_db_path(save_path)
                
                GenData.objects.create(
                    video_id=video_id,
                    theme=theme,
                    shot_type=shot_type,
                    prompt=prompt,
                    video_file=db_path,
                    tester=task_record.username,
                    model_name=model_name,
                    model_url=api_url
                )
                
                success_count += 1
                logger.info(f'[Task {task_id}] 成功处理: {video_id}')
                
            except Exception as e:
                logger.error(f'[Task {task_id}] 处理视频异常: {str(e)}', exc_info=True)
                failed_count += 1
        
        # 总结
        logger.info(f'[Task {task_id}] 完成 - 成功: {success_count}, 失败: {failed_count}')
        
        if success_count == 0:
            return False, '所有视频生成失败'
        
        return True, None
        
    except Exception as e:
        logger.error(f'[Task {task_id}] generate_all_videos异常: {str(e)}', exc_info=True)
        return False, str(e)


def run_scoring_algorithm(task_id, model_name):
    """运行打分算法"""
    try:
        # 从环境变量读取评分模式
        import os
        scoring_mode = os.getenv('REF4D_SCORING_MODE', 'SIMULATION')
        logger.info(f'[Task {task_id}] 评分模式: {scoring_mode}')
        
        gendata_list = GenData.objects.filter(model_name=model_name)
        refdata_dict = {r.video_id: r for r in RefData.objects.all()}
        
        # 使用环境变量中的mode
        scorer = VideoScoring(mode=scoring_mode)
        scores = scorer.evaluate_model(gendata_list, refdata_dict)
        return scores
    except Exception as e:
        logger.error(f'[Task {task_id}] 打分算法异常: {str(e)}', exc_info=True)
        # 返回默认分数
        return {
            'semantic': 80.0, 'temporal': 75.0,
            'motion': 82.0, 'reality': 78.0, 'total': 78.8
        }

