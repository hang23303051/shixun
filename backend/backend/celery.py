"""
Celery配置文件
"""
import os
from pathlib import Path
from celery import Celery
from django.conf import settings

# ⭐ 加载.env文件(确保环境变量在Celery启动时生效)
from dotenv import load_dotenv
# 找到项目根目录的.env文件
# backend/backend/celery.py -> backend/backend/ -> backend/ -> 项目根目录/
BASE_DIR = Path(__file__).resolve().parent.parent.parent
env_path = BASE_DIR / '.env'
if env_path.exists():
    load_dotenv(env_path)
    print(f"✅ Celery已加载.env: {env_path}")
    print(f"   REF4D_SCORING_MODE = {os.getenv('REF4D_SCORING_MODE', 'NOT_SET')}")
else:
    print(f"⚠️  未找到.env文件: {env_path}")

# 设置Django settings模块
os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'backend.settings')

# 创建Celery应用
app = Celery('ref4d')

# 使用Django settings作为配置源
app.config_from_object('django.conf:settings', namespace='CELERY')

# 自动发现所有已注册Django app中的tasks.py
app.autodiscover_tasks(lambda: settings.INSTALLED_APPS)

# Celery配置
app.conf.update(
    # 任务结果过期时间（秒）
    result_expires=3600,
    
    # 任务超时时间（秒）- 24小时
    task_time_limit=24 * 60 * 60,
    
    # 软超时时间（秒）- 23小时
    task_soft_time_limit=23 * 60 * 60,
    
    # 任务跟踪
    task_track_started=True,
    
    # 任务忽略结果（对于不需要结果的任务）
    task_ignore_result=False,
    
    # 时区
    timezone='Asia/Shanghai',
    enable_utc=True,
)

@app.task(bind=True)
def debug_task(self):
    """调试任务"""
    print(f'Request: {self.request!r}')
