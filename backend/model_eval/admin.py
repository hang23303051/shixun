from django.contrib import admin
from django import forms
from .models import Model
from account.models import User


class ModelAdminForm(forms.ModelForm):
    """自定义模型表单"""
    tester_name = forms.ChoiceField(
        label='测试人姓名',
        required=False,
        help_text='如果是管理员类型则自动设置为root，如果是普通用户请从列表中选择'
    )
    
    class Meta:
        model = Model
        fields = '__all__'
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # 获取所有用户的用户名作为选项
        user_choices = [('', '---------')] + [(user.username, user.username) for user in User.objects.all()]
        self.fields['tester_name'].choices = user_choices


@admin.register(Model)
class ModelAdmin(admin.ModelAdmin):
    form = ModelAdminForm
    list_display = ['name', 'publisher', 'total_score', 'tester_type', 'tester_name', 'created_at']
    list_filter = ['tester_type', 'is_open_source']
    search_fields = ['name', 'publisher', 'description']
    readonly_fields = ['total_score', 'created_at', 'updated_at']
    fieldsets = (
        ('基本信息', {
            'fields': ('name', 'description', 'publisher', 'parameters', 'is_open_source', 
                      'release_date', 'official_website')
        }),
        ('评分信息', {
            'fields': ('semantic_score', 'temporal_score', 'motion_score', 'reality_score', 'total_score')
        }),
        ('测试信息', {
            'fields': ('tester_type', 'tester_name'),
            'description': '如果选择管理员类型，测试人姓名将自动设置为root；如果选择普通用户，请从下拉列表中选择已有用户'
        }),
        ('时间信息', {
            'fields': ('created_at', 'updated_at')
        }),
    )
    
    def save_model(self, request, obj, form, change):
        """保存时根据测试人类型自动设置姓名"""
        if obj.tester_type == 'admin':
            obj.tester_name = 'root'
        super().save_model(request, obj, form, change)
