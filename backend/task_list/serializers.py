from rest_framework import serializers
from .models import TaskList


class TaskListSerializer(serializers.ModelSerializer):
    """任务列表序列化器 - 完整信息"""
    status_display = serializers.CharField(source='get_status_display', read_only=True)
    
    class Meta:
        model = TaskList
        fields = '__all__'
        read_only_fields = ['task_id', 'created_at', 'updated_at', 'completed_at']


class TaskListCreateSerializer(serializers.ModelSerializer):
    """创建任务序列化器"""
    
    class Meta:
        model = TaskList
        fields = [
            'task_id', 'user_email', 'username', 'model_name', 
            'api_url', 'api_key', 'description', 'publisher', 
            'parameters', 'is_open_source', 'release_date', 'official_website'
        ]


class TaskListStatusSerializer(serializers.ModelSerializer):
    """任务状态序列化器 - 用于查询状态和详情"""
    status_display = serializers.CharField(source='get_status_display', read_only=True)
    
    class Meta:
        model = TaskList
        fields = [
            'task_id', 'model_name', 'api_url', 'user_email', 'username',
            'status', 'status_display', 'progress', 'message', 
            'semantic_score', 'temporal_score', 'motion_score', 'reality_score', 
            'total_score', 'model_id', 'description', 'publisher', 'parameters',
            'is_open_source', 'release_date', 'official_website',
            'created_at', 'updated_at', 'completed_at'
        ]


class TaskListListSerializer(serializers.ModelSerializer):
    """任务列表序列化器 - 用于列表展示"""
    status_display = serializers.CharField(source='get_status_display', read_only=True)
    
    class Meta:
        model = TaskList
        fields = [
            'task_id', 'model_name', 'username', 'status', 
            'status_display', 'progress', 'total_score', 
            'model_id', 'created_at', 'completed_at'
        ]
