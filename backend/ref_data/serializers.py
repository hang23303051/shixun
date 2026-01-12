from rest_framework import serializers
from .models import RefData


class RefDataSerializer(serializers.ModelSerializer):
    """参考数据序列化器"""
    theme_display = serializers.CharField(source='get_theme_display', read_only=True)
    shot_type_display = serializers.CharField(source='get_shot_type_display', read_only=True)
    video_file = serializers.SerializerMethodField()
    
    class Meta:
        model = RefData
        fields = '__all__'
    
    def get_video_file(self, obj):
        """处理视频文件路径，转换为正确的媒体URL"""
        # 数据库中存储的是: backend/media/refdata/videos/...
        # 需要转换为: refdata/videos/...
        if obj.video_file:
            # 移除 backend/media/ 前缀
            video_path = obj.video_file.replace('backend/media/', '', 1)
            return video_path
        return ''
