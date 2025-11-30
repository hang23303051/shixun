from django.contrib import admin
from django import forms
from .models import User


class UserAdminForm(forms.ModelForm):
    """自定义用户表单"""
    password = forms.CharField(
        label='密码',
        widget=forms.PasswordInput,
        help_text='请输入密码（将自动加密）',
        required=False
    )
    
    class Meta:
        model = User
        fields = '__all__'


@admin.register(User)
class UserAdmin(admin.ModelAdmin):
    form = UserAdminForm
    list_display = ['email', 'username', 'avatar', 'created_at']
    search_fields = ['email', 'username']
    readonly_fields = ['created_at', 'updated_at']
    fieldsets = (
        ('基本信息', {
            'fields': ('email', 'username', 'password')
        }),
        ('头像', {
            'fields': ('avatar',)
        }),
        ('时间信息', {
            'fields': ('created_at', 'updated_at')
        }),
    )
    
    def save_model(self, request, obj, form, change):
        """保存时自动加密密码"""
        if form.cleaned_data.get('password'):
            # 如果密码字段有值，进行加密
            obj.set_password(form.cleaned_data['password'])
        super().save_model(request, obj, form, change)
