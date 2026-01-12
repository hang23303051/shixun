import pymysql
pymysql.install_as_MySQLdb()

# 导入Cel ery应用，确保Django启动时初始化Celery
from .celery import app as celery_app

__all__ = ('celery_app',)
