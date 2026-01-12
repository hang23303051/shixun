from rest_framework import serializers
from datetime import date
from .models import GenData


class EvalRequestSerializer(serializers.Serializer):
    """评测请求序列化器"""
    api_url = serializers.URLField(max_length=500)
    api_key = serializers.CharField(max_length=500)
    model_name = serializers.CharField(max_length=100)
    
    # 模型信息
    description = serializers.CharField(required=False, allow_blank=True)
    publisher = serializers.CharField(max_length=200, required=False, allow_blank=True)
    parameters = serializers.CharField(max_length=100, required=False, allow_blank=True)
    is_open_source = serializers.BooleanField(required=False, default=False)
    release_date = serializers.DateField(required=False)
    official_website = serializers.URLField(max_length=500, required=False, allow_blank=True)
    
    def validate_release_date(self, value):
        """验证发布时间不能超过当前时间"""
        if value and value > date.today():
            raise serializers.ValidationError('发布时间不能超过当前时间')
        return value


class EvalStatusSerializer(serializers.Serializer):
    """评测状态序列化器"""
    task_id = serializers.CharField()
    status = serializers.ChoiceField(choices=['pending', 'running', 'completed', 'failed'])
    progress = serializers.IntegerField(min_value=0, max_value=100)
    message = serializers.CharField(required=False)
    result = serializers.DictField(required=False)


class GenDataSerializer(serializers.ModelSerializer):
    """生成数据序列化器"""
    theme_display = serializers.CharField(source='get_theme_display', read_only=True)
    shot_type_display = serializers.CharField(source='get_shot_type_display', read_only=True)
    
    class Meta:
        model = GenData
        fields = '__all__'
