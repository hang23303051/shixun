from django.contrib import admin
from .models import RefData


@admin.register(RefData)
class RefDataAdmin(admin.ModelAdmin):
    list_display = ['video_id', 'theme', 'shot_type', 'created_at']
    list_filter = ['theme', 'shot_type']
    search_fields = ['video_id', 'prompt']
    readonly_fields = ['created_at']
