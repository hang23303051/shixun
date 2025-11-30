from rest_framework import serializers
from .models import Model


class ModelSerializer(serializers.ModelSerializer):
    """模型序列化器"""
    tester_type_display = serializers.CharField(source='get_tester_type_display', read_only=True)
    
    class Meta:
        model = Model
        fields = '__all__'
        read_only_fields = ['total_score', 'created_at', 'updated_at']


class ModelListSerializer(serializers.ModelSerializer):
    """模型列表序列化器 - 精简版"""
    class Meta:
        model = Model
        fields = ['id', 'name', 'publisher', 'total_score', 'semantic_score', 
                  'temporal_score', 'motion_score', 'reality_score']


class RankingSerializer(serializers.Serializer):
    """排行榜序列化器"""
    rank = serializers.IntegerField()
    model_id = serializers.IntegerField()
    model_name = serializers.CharField()
    total_score = serializers.FloatField()
    semantic_score = serializers.FloatField(required=False)
    temporal_score = serializers.FloatField(required=False)
    motion_score = serializers.FloatField(required=False)
    reality_score = serializers.FloatField(required=False)
