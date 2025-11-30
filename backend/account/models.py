from django.db import models
from django.contrib.auth.hashers import make_password, check_password


class User(models.Model):
    """用户模型 - 以邮箱为主键"""
    email = models.EmailField(primary_key=True, max_length=255, verbose_name='邮箱')
    username = models.CharField(max_length=150, verbose_name='用户名')
    password = models.CharField(max_length=255, verbose_name='密码')
    avatar = models.ImageField(upload_to='avatars/', null=True, blank=True, verbose_name='头像')
    created_at = models.DateTimeField(auto_now_add=True, verbose_name='创建时间')
    updated_at = models.DateTimeField(auto_now=True, verbose_name='更新时间')
    
    class Meta:
        db_table = 'user'
        verbose_name = '用户'
        verbose_name_plural = '用户'
    
    def set_password(self, raw_password):
        """设置加密密码"""
        self.password = make_password(raw_password)
    
    def check_password(self, raw_password):
        """验证密码"""
        return check_password(raw_password, self.password)
    
    def __str__(self):
        return self.username
