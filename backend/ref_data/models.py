from django.db import models


class RefData(models.Model):
    """参考数据集 - 600条测试用例"""
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
    
    SHOT_TYPE_CHOICES = [
        ('single', '单镜头'),
        ('multi', '多镜头'),
    ]
    
    video_id = models.CharField(max_length=100, primary_key=True, verbose_name='视频ID')
    theme = models.CharField(max_length=50, choices=THEME_CHOICES, verbose_name='主题类型')
    shot_type = models.CharField(max_length=10, choices=SHOT_TYPE_CHOICES, verbose_name='镜头类型')
    prompt = models.TextField(verbose_name='Prompt描述')
    video_file = models.FileField(upload_to='refdata/videos/', verbose_name='视频文件')
    created_at = models.DateTimeField(auto_now_add=True, verbose_name='创建时间')
    
    class Meta:
        db_table = 'refdata'
        verbose_name = '参考数据'
        verbose_name_plural = '参考数据'
    
    def __str__(self):
        return f"{self.video_id} - {self.get_theme_display()}"


class GenData(models.Model):
    """生成数据 - 用于打分算法过程中的数据存储"""
    THEME_CHOICES = RefData.THEME_CHOICES
    SHOT_TYPE_CHOICES = RefData.SHOT_TYPE_CHOICES
    
    video_id = models.CharField(max_length=100, verbose_name='视频ID')
    theme = models.CharField(max_length=50, choices=THEME_CHOICES, verbose_name='主题类型')
    shot_type = models.CharField(max_length=10, choices=SHOT_TYPE_CHOICES, verbose_name='镜头类型')
    model_name = models.CharField(max_length=100, verbose_name='模型名称')
    prompt = models.TextField(verbose_name='Prompt描述')
    video_file = models.FileField(upload_to='gendata/videos/', verbose_name='视频文件')
    created_at = models.DateTimeField(auto_now_add=True, verbose_name='创建时间')
    
    class Meta:
        db_table = 'gendata'
        verbose_name = '生成数据'
        verbose_name_plural = '生成数据'
        indexes = [
            models.Index(fields=['model_name']),
        ]
    
    def __str__(self):
        return f"{self.video_id} - {self.model_name}"
