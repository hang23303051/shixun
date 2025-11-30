from rest_framework import serializers


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


class EvalStatusSerializer(serializers.Serializer):
    """评测状态序列化器"""
    task_id = serializers.CharField()
    status = serializers.ChoiceField(choices=['pending', 'running', 'completed', 'failed'])
    progress = serializers.IntegerField(min_value=0, max_value=100)
    message = serializers.CharField(required=False)
    result = serializers.DictField(required=False)
