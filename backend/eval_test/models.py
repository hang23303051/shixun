from django.db import models


class GenData(models.Model):
    """生成数据表 - 存储模型生成的视频及评测信息"""
    
    # 主题选择（与RefData保持一致）
    THEME_CHOICES = [
        ('animals_and_ecology', '动物与生态'),
        ('architecture', '建筑'),
        ('commercial_marketing', '商业营销'),
        ('food', '食物'),
        ('industrial_activity', '工业活动'),
        ('landscape', '风景'),
        ('people_daily', '人物日常'),
        ('sports_competition', '体育竞技'),
        ('transportation', '交通'),
    ]
    
    # 镜头类型选择
    SHOT_TYPE_CHOICES = [
        ('single', '单镜头'),
        ('multi', '多镜头'),
    ]
    
    # 基础字段
    video_id = models.CharField(max_length=100, verbose_name='视频ID')
    theme = models.CharField(max_length=50, choices=THEME_CHOICES, verbose_name='主题类型')
    shot_type = models.CharField(max_length=10, choices=SHOT_TYPE_CHOICES, verbose_name='镜头类型')
    prompt = models.TextField(verbose_name='Prompt描述')
    video_file = models.CharField(max_length=500, verbose_name='视频文件路径')
    
    # 新增字段
    tester = models.CharField(max_length=150, verbose_name='测试人')
    model_name = models.CharField(max_length=100, verbose_name='模型名称')
    model_url = models.URLField(max_length=500, verbose_name='模型API URL')
    test_time = models.DateTimeField(auto_now_add=True, verbose_name='测试时间')
    
    created_at = models.DateTimeField(auto_now_add=True, verbose_name='创建时间')
    
    class Meta:
        db_table = 'gendata'
        verbose_name = '生成数据'
        verbose_name_plural = '生成数据'
        indexes = [
            models.Index(fields=['model_name']),
            models.Index(fields=['tester']),
        ]
    
    def __str__(self):
        return f"{self.video_id} - {self.model_name}"
