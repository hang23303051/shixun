from rest_framework import serializers
from .models import RefData, GenData


class RefDataSerializer(serializers.ModelSerializer):
    """参考数据序列化器"""
    theme_display = serializers.CharField(source='get_theme_display', read_only=True)
    shot_type_display = serializers.CharField(source='get_shot_type_display', read_only=True)
    
    class Meta:
        model = RefData
        fields = '__all__'


class GenDataSerializer(serializers.ModelSerializer):
    """生成数据序列化器"""
    theme_display = serializers.CharField(source='get_theme_display', read_only=True)
    shot_type_display = serializers.CharField(source='get_shot_type_display', read_only=True)
    
    class Meta:
        model = GenData
        fields = '__all__'
