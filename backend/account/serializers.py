from rest_framework import serializers
from .models import User


class UserSerializer(serializers.ModelSerializer):
    """用户序列化器"""
    avatar = serializers.SerializerMethodField()
    
    class Meta:
        model = User
        fields = ['email', 'username', 'avatar', 'created_at']
        read_only_fields = ['created_at']
    
    def get_avatar(self, obj):
        """返回头像URL，如果有的话"""
        if obj.avatar:
            # 只返回相对于MEDIA_ROOT的路径，不包含media/前缀
            return obj.avatar.name
        return None


class UserRegisterSerializer(serializers.ModelSerializer):
    """用户注册序列化器"""
    password_confirm = serializers.CharField(write_only=True)
    
    class Meta:
        model = User
        fields = ['email', 'username', 'password', 'password_confirm']
        extra_kwargs = {
            'password': {'write_only': True}
        }
    
    def validate(self, data):
        if data['password'] != data['password_confirm']:
            raise serializers.ValidationError("两次密码不一致")
        return data
    
    def create(self, validated_data):
        validated_data.pop('password_confirm')
        user = User(
            email=validated_data['email'],
            username=validated_data['username']
        )
        user.set_password(validated_data['password'])
        user.save()
        return user


class UserLoginSerializer(serializers.Serializer):
    """用户登录序列化器"""
    username = serializers.CharField()
    password = serializers.CharField(write_only=True)


class UserUpdateSerializer(serializers.ModelSerializer):
    """用户信息更新序列化器"""
    old_password = serializers.CharField(write_only=True, required=False)
    new_password = serializers.CharField(write_only=True, required=False)
    
    class Meta:
        model = User
        fields = ['username', 'avatar', 'old_password', 'new_password']
    
    def update(self, instance, validated_data):
        if 'new_password' in validated_data:
            old_password = validated_data.pop('old_password', None)
            if not old_password or not instance.check_password(old_password):
                raise serializers.ValidationError("原密码错误")
            instance.set_password(validated_data.pop('new_password'))
        
        for attr, value in validated_data.items():
            setattr(instance, attr, value)
        instance.save()
        return instance
