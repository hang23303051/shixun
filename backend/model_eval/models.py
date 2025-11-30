from django.db import models


class Model(models.Model):
    """视频生成模型"""
    TESTER_CHOICES = [
        ('admin', '管理员'),
        ('user', '普通用户'),
    ]
    
    name = models.CharField(max_length=100, unique=True, verbose_name='模型名称')
    description = models.TextField(verbose_name='文字简介')
    publisher = models.CharField(max_length=200, verbose_name='发布者')
    parameters = models.CharField(max_length=100, verbose_name='参数规模')
    is_open_source = models.BooleanField(default=False, verbose_name='是否开源')
    release_date = models.DateField(verbose_name='发布时间')
    official_website = models.URLField(max_length=500, verbose_name='官网链接')
    
    # 四个维度评分
    semantic_score = models.FloatField(verbose_name='基础语义一致性评分')
    temporal_score = models.FloatField(verbose_name='时序一致性评分')
    motion_score = models.FloatField(verbose_name='运动属性评分')
    reality_score = models.FloatField(verbose_name='世界知识真实性评分')
    total_score = models.FloatField(verbose_name='总分')
    
    tester_type = models.CharField(max_length=10, choices=TESTER_CHOICES, verbose_name='测试人类型')
    tester_name = models.CharField(max_length=150, verbose_name='测试人姓名')
    
    created_at = models.DateTimeField(auto_now_add=True, verbose_name='创建时间')
    updated_at = models.DateTimeField(auto_now=True, verbose_name='更新时间')
    
    class Meta:
        db_table = 'model'
        verbose_name = '模型'
        verbose_name_plural = '模型'
        ordering = ['-total_score']
    
    def save(self, *args, **kwargs):
        """保存时自动计算总分"""
        if self.semantic_score and self.temporal_score and self.motion_score and self.reality_score:
            self.total_score = (self.semantic_score + self.temporal_score + 
                              self.motion_score + self.reality_score) / 4
        super().save(*args, **kwargs)
    
    def __str__(self):
        return self.name
