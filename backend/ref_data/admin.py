from django.contrib import admin
from .models import RefData, GenData


@admin.register(RefData)
class RefDataAdmin(admin.ModelAdmin):
    list_display = ['video_id', 'theme', 'shot_type', 'created_at']
    list_filter = ['theme', 'shot_type']
    search_fields = ['video_id', 'prompt']
    readonly_fields = ['created_at']


@admin.register(GenData)
class GenDataAdmin(admin.ModelAdmin):
    list_display = ['video_id', 'model_name', 'theme', 'shot_type', 'created_at']
    list_filter = ['model_name', 'theme', 'shot_type']
    search_fields = ['video_id', 'model_name', 'prompt']
    readonly_fields = ['created_at']
