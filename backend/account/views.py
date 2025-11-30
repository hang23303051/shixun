from rest_framework import status
from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework.parsers import MultiPartParser, FormParser, JSONParser
from django.views.decorators.csrf import ensure_csrf_cookie
from django.utils.decorators import method_decorator
from .models import User
from .serializers import (
    UserSerializer, UserRegisterSerializer, 
    UserLoginSerializer, UserUpdateSerializer
)


@method_decorator(ensure_csrf_cookie, name='dispatch')
class RegisterView(APIView):
    """用户注册"""
    def post(self, request):
        serializer = UserRegisterSerializer(data=request.data)
        if serializer.is_valid():
            user = serializer.save()
            return Response({
                'message': '注册成功',
                'user': UserSerializer(user).data
            }, status=status.HTTP_201_CREATED)
        return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)


class LoginView(APIView):
    """用户登录"""
    def post(self, request):
        serializer = UserLoginSerializer(data=request.data)
        if serializer.is_valid():
            username = serializer.validated_data['username']
            password = serializer.validated_data['password']
            
            try:
                user = User.objects.get(username=username)
                if user.check_password(password):
                    request.session['user_email'] = user.email
                    return Response({
                        'message': '登录成功',
                        'user': UserSerializer(user).data
                    })
                else:
                    return Response({'error': '密码错误'}, status=status.HTTP_401_UNAUTHORIZED)
            except User.DoesNotExist:
                return Response({'error': '用户不存在'}, status=status.HTTP_404_NOT_FOUND)
        
        return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)


class LogoutView(APIView):
    """用户登出"""
    def post(self, request):
        request.session.flush()
        return Response({'message': '登出成功'})


class UserProfileView(APIView):
    """获取和更新用户信息"""
    parser_classes = (JSONParser, MultiPartParser, FormParser)
    
    def get(self, request):
        """获取当前用户信息"""
        user_email = request.session.get('user_email')
        if not user_email:
            return Response({'error': '未登录'}, status=status.HTTP_401_UNAUTHORIZED)
        
        try:
            user = User.objects.get(email=user_email)
            return Response(UserSerializer(user).data)
        except User.DoesNotExist:
            return Response({'error': '用户不存在'}, status=status.HTTP_404_NOT_FOUND)
    
    def put(self, request):
        """更新用户信息"""
        user_email = request.session.get('user_email')
        if not user_email:
            return Response({'error': '未登录'}, status=status.HTTP_401_UNAUTHORIZED)
        
        try:
            user = User.objects.get(email=user_email)
            serializer = UserUpdateSerializer(user, data=request.data, partial=True)
            if serializer.is_valid():
                serializer.save()
                return Response({
                    'message': '更新成功',
                    'user': UserSerializer(user).data
                })
            return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)
        except User.DoesNotExist:
            return Response({'error': '用户不存在'}, status=status.HTTP_404_NOT_FOUND)
    
    def patch(self, request):
        """部分更新用户信息（别名）"""
        return self.put(request)


class CheckLoginView(APIView):
    """检查登录状态"""
    def get(self, request):
        user_email = request.session.get('user_email')
        if user_email:
            try:
                user = User.objects.get(email=user_email)
                return Response({
                    'logged_in': True,
                    'user': UserSerializer(user).data
                })
            except User.DoesNotExist:
                return Response({'logged_in': False})
        return Response({'logged_in': False})


@method_decorator(ensure_csrf_cookie, name='dispatch')
class GetCSRFTokenView(APIView):
    """获取CSRF Token"""
    def get(self, request):
        return Response({'detail': 'CSRF cookie set'})
