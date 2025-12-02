from django.contrib import admin
from .models import TaskList


@admin.register(TaskList)
class TaskListAdmin(admin.ModelAdmin):
    """任务列表管理"""
    list_display = [
        'task_id', 'model_name', 'username', 'status', 
        'progress', 'total_score', 'created_at', 'completed_at'
    ]
    list_filter = ['status', 'is_open_source', 'created_at']
    search_fields = ['task_id', 'model_name', 'username', 'user_email']
    readonly_fields = ['task_id', 'created_at', 'updated_at', 'completed_at']
    
    fieldsets = (
        ('基本信息', {
            'fields': ('task_id', 'user_email', 'username', 'status', 'progress', 'message')
        }),
        ('模型信息', {
            'fields': (
                'model_name', 'api_url', 'api_key', 'description', 
                'publisher', 'parameters', 'is_open_source', 
                'release_date', 'official_website'
            )
        }),
        ('评测结果', {
            'fields': (
                'semantic_score', 'temporal_score', 'motion_score', 
                'reality_score', 'total_score', 'model_id'
            )
        }),
        ('时间信息', {
            'fields': ('created_at', 'updated_at', 'completed_at')
        }),
    )
    
    def has_add_permission(self, request):
        """禁止在后台手动添加任务"""
        return False
