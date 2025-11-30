from django.contrib import admin
from .models import Model


@admin.register(Model)
class ModelAdmin(admin.ModelAdmin):
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
            'fields': ('tester_type', 'tester_name')
        }),
        ('时间信息', {
            'fields': ('created_at', 'updated_at')
        }),
    )
