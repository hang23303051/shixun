from django.db import models


class TaskList(models.Model):
    """评测任务列表 - 记录所有评测提交"""
    STATUS_CHOICES = [
        ('pending', '等待处理'),
        ('processing', '处理中'),
        ('completed', '已完成'),
        ('failed', '失败'),
    ]
    
    task_id = models.CharField(max_length=100, unique=True, verbose_name='任务ID')
    user_email = models.EmailField(max_length=255, verbose_name='提交用户邮箱')
    username = models.CharField(max_length=150, verbose_name='提交用户名')
    
    # 提交的模型信息
    model_name = models.CharField(max_length=100, verbose_name='模型名称')
    api_url = models.URLField(max_length=500, verbose_name='API地址')
    api_key = models.CharField(max_length=500, verbose_name='API密钥')
    
    # 可选的模型信息
    description = models.TextField(blank=True, verbose_name='模型简介')
    publisher = models.CharField(max_length=200, blank=True, verbose_name='发布者')
    parameters = models.CharField(max_length=100, blank=True, verbose_name='参数规模')
    is_open_source = models.BooleanField(default=False, verbose_name='是否开源')
    release_date = models.DateField(null=True, blank=True, verbose_name='发布时间')
    official_website = models.URLField(max_length=500, blank=True, verbose_name='官网链接')
    
    # 任务状态
    status = models.CharField(max_length=20, choices=STATUS_CHOICES, default='pending', verbose_name='任务状态')
    progress = models.IntegerField(default=0, verbose_name='进度百分比')
    message = models.TextField(blank=True, verbose_name='状态消息')
    
    # 评测结果（完成后填充）
    semantic_score = models.FloatField(null=True, blank=True, verbose_name='语义一致性评分')
    temporal_score = models.FloatField(null=True, blank=True, verbose_name='时序一致性评分')
    motion_score = models.FloatField(null=True, blank=True, verbose_name='运动属性评分')
    reality_score = models.FloatField(null=True, blank=True, verbose_name='真实性评分')
    total_score = models.FloatField(null=True, blank=True, verbose_name='总分')
    
    # 关联的模型ID（评测完成后创建）
    model_id = models.IntegerField(null=True, blank=True, verbose_name='关联模型ID')
    
    # 时间戳
    created_at = models.DateTimeField(auto_now_add=True, verbose_name='创建时间')
    updated_at = models.DateTimeField(auto_now=True, verbose_name='更新时间')
    completed_at = models.DateTimeField(null=True, blank=True, verbose_name='完成时间')
    
    class Meta:
        db_table = 'task_list'
        verbose_name = '评测任务'
        verbose_name_plural = '评测任务列表'
        ordering = ['-created_at']
    
    def __str__(self):
        return f"{self.task_id} - {self.model_name} ({self.get_status_display()})"
