from django.db import models
from django.contrib.auth.hashers import make_password, check_password
from django.utils import timezone
import secrets


class User(models.Model):
    """用户模型 - 以邮箱为主键"""
    email = models.EmailField(primary_key=True, max_length=255, verbose_name='邮箱')
    username = models.CharField(max_length=150, verbose_name='用户名')
    password = models.CharField(max_length=255, verbose_name='密码')
    avatar = models.ImageField(upload_to='avatars/', null=True, blank=True, verbose_name='头像')
    is_active = models.BooleanField(default=False, verbose_name='是否激活')
    
    # 激活token
    activation_token = models.CharField(max_length=64, null=True, blank=True, verbose_name='激活令牌')
    activation_token_created = models.DateTimeField(null=True, blank=True, verbose_name='激活令牌创建时间')
    
    # 密码重置
    reset_password_token = models.CharField(max_length=6, null=True, blank=True, verbose_name='密码重置验证码')
    reset_password_token_created = models.DateTimeField(null=True, blank=True, verbose_name='重置令牌创建时间')
    
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
    
    def generate_activation_token(self):
        """生成激活令牌"""
        self.activation_token = secrets.token_urlsafe(32)
        self.activation_token_created = timezone.now()
        return self.activation_token
    
    def generate_reset_password_token(self):
        """生成6位数字密码重置验证码"""
        self.reset_password_token = ''.join([str(secrets.randbelow(10)) for _ in range(6)])
        self.reset_password_token_created = timezone.now()
        return self.reset_password_token
    
    def is_activation_token_valid(self, hours=24):
        """检查激活令牌是否有效（默认24小时）"""
        if not self.activation_token_created:
            return False
        expiry = self.activation_token_created + timezone.timedelta(hours=hours)
        return timezone.now() < expiry
    
    def is_reset_token_valid(self, minutes=15):
        """检查重置令牌是否有效（默认15分钟）"""
        if not self.reset_password_token_created:
            return False
        expiry = self.reset_password_token_created + timezone.timedelta(minutes=minutes)
        return timezone.now() < expiry
    
    def __str__(self):
        return self.username
