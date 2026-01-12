from django.contrib import admin
from .models import GenData


@admin.register(GenData)
class GenDataAdmin(admin.ModelAdmin):
    list_display = ['video_id', 'model_name', 'tester', 'theme', 'shot_type', 'test_time']
    list_filter = ['model_name', 'tester', 'theme', 'shot_type']
    search_fields = ['video_id', 'model_name', 'tester', 'prompt']
    readonly_fields = ['test_time', 'created_at']
    
    fieldsets = (
        ('基础信息', {
            'fields': ('video_id', 'theme', 'shot_type', 'prompt', 'video_file')
        }),
        ('模型信息', {
            'fields': ('model_name', 'model_url', 'tester')
        }),
        ('时间信息', {
            'fields': ('test_time', 'created_at')
        }),
    )
