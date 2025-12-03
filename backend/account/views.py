from rest_framework import status
from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework.parsers import MultiPartParser, FormParser, JSONParser
from django.views.decorators.csrf import ensure_csrf_cookie
from django.utils.decorators import method_decorator
from django.conf import settings
from .models import User
from .serializers import (
    UserSerializer, UserRegisterSerializer, 
    UserLoginSerializer, UserUpdateSerializer
)
from .email_utils import send_activation_email, send_password_reset_email


@method_decorator(ensure_csrf_cookie, name='dispatch')
class RegisterView(APIView):
    """用户注册 - 发送激活邮件"""
    def post(self, request):
        serializer = UserRegisterSerializer(data=request.data)
        if serializer.is_valid():
            user = serializer.save()
            
            # 生成激活token
            activation_token = user.generate_activation_token()
            user.save()
            
            # 构造激活URL
            frontend_url = request.headers.get('Origin', 'http://localhost:8080')
            activation_url = f"{frontend_url}/activate/{user.email}/{activation_token}"
            
            try:
                # 发送激活邮件
                send_activation_email(user, activation_url)
                return Response({
                    'message': '注册成功！我们已向您的邮箱发送了激活链接，请查收邮件并点击链接激活账户。',
                    'email': user.email,
                    'require_activation': True
                }, status=status.HTTP_201_CREATED)
            except Exception as e:
                # 如果邮件发送失败，删除用户
                user.delete()
                return Response({
                    'error': f'邮件发送失败：{str(e)}',
                    'detail': '请检查邮箱地址是否正确或稍后重试'
                }, status=status.HTTP_500_INTERNAL_SERVER_ERROR)
        
        return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)


class LoginView(APIView):
    """用户登录 - 支持用户名或邮箱登录"""
    def post(self, request):
        serializer = UserLoginSerializer(data=request.data)
        if serializer.is_valid():
            username_or_email = serializer.validated_data['username']
            password = serializer.validated_data['password']
            
            try:
                # 智能识别：判断输入的是用户名还是邮箱
                if '@' in username_or_email:
                    # 输入的是邮箱
                    user = User.objects.get(email=username_or_email)
                else:
                    # 输入的是用户名
                    user = User.objects.get(username=username_or_email)
                
                # 检查密码
                if not user.check_password(password):
                    return Response({'error': '密码错误'}, status=status.HTTP_401_UNAUTHORIZED)
                
                # 检查是否激活
                if not user.is_active:
                    return Response({
                        'error': '账户未激活',
                        'detail': '请先前往注册邮箱查收激活邮件并激活账户',
                        'require_activation': True,
                        'email': user.email
                    }, status=status.HTTP_403_FORBIDDEN)
                
                # 登录成功
                request.session['user_email'] = user.email
                return Response({
                    'message': '登录成功',
                    'user': UserSerializer(user).data
                })
                
            except User.DoesNotExist:
                return Response({
                    'error': '用户不存在',
                    'detail': '请检查用户名/邮箱是否正确'
                }, status=status.HTTP_404_NOT_FOUND)
        
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


class ActivateAccountView(APIView):
    """激活账户"""
    def get(self, request, email, token):
        try:
            user = User.objects.get(email=email)
            
            # 检查token是否匹配
            if user.activation_token != token:
                return Response({
                    'error': '激活链接无效',
                    'detail': '请检查链接是否完整或重新注册'
                }, status=status.HTTP_400_BAD_REQUEST)
            
            # 检查token是否过期
            if not user.is_activation_token_valid():
                return Response({
                    'error': '激活链接已过期',
                    'detail': '链接有效期为24小时，请重新注册'
                }, status=status.HTTP_400_BAD_REQUEST)
            
            # 激活用户
            user.is_active = True
            user.activation_token = None
            user.activation_token_created = None
            user.save()
            
            return Response({
                'message': '账户激活成功！',
                'detail': '您现在可以登录了'
            })
            
        except User.DoesNotExist:
            return Response({
                'error': '用户不存在',
                'detail': '请先注册账户'
            }, status=status.HTTP_404_NOT_FOUND)


class ResendActivationEmailView(APIView):
    """重新发送激活邮件"""
    def post(self, request):
        email = request.data.get('email')
        if not email:
            return Response({'error': '请提供邮箱地址'}, status=status.HTTP_400_BAD_REQUEST)
        
        try:
            user = User.objects.get(email=email)
            
            # 检查是否已激活
            if user.is_active:
                return Response({
                    'error': '账户已激活',
                    'detail': '您可以直接登录'
                }, status=status.HTTP_400_BAD_REQUEST)
            
            # 重新生成激活token
            activation_token = user.generate_activation_token()
            user.save()
            
            # 发送激活邮件
            frontend_url = request.headers.get('Origin', 'http://localhost:8080')
            activation_url = f"{frontend_url}/activate/{user.email}/{activation_token}"
            
            try:
                send_activation_email(user, activation_url)
                return Response({
                    'message': '激活邮件已重新发送',
                    'detail': '请查收邮件并点击链接激活账户'
                })
            except Exception as e:
                return Response({
                    'error': f'邮件发送失败：{str(e)}'
                }, status=status.HTTP_500_INTERNAL_SERVER_ERROR)
                
        except User.DoesNotExist:
            return Response({
                'error': '邮箱未注册',
                'detail': '请先注册账户'
            }, status=status.HTTP_404_NOT_FOUND)


class RequestPasswordResetView(APIView):
    """请求密码重置（发送验证码）"""
    def post(self, request):
        email = request.data.get('email')
        if not email:
            return Response({'error': '请提供邮箱地址'}, status=status.HTTP_400_BAD_REQUEST)
        
        try:
            user = User.objects.get(email=email)
            
            # 检查账户是否激活
            if not user.is_active:
                return Response({
                    'error': '账户未激活',
                    'detail': '请先激活账户后再重置密码'
                }, status=status.HTTP_403_FORBIDDEN)
            
            # 生成6位数字验证码
            reset_code = user.generate_reset_password_token()
            user.save()
            
            # 发送验证码邮件
            try:
                send_password_reset_email(user, reset_code)
                return Response({
                    'message': '验证码已发送',
                    'detail': '请查收邮件并使用验证码重置密码（15分钟内有效）',
                    'email': user.email
                })
            except Exception as e:
                return Response({
                    'error': f'邮件发送失败：{str(e)}'
                }, status=status.HTTP_500_INTERNAL_SERVER_ERROR)
                
        except User.DoesNotExist:
            return Response({
                'error': '邮箱未注册',
                'detail': '请先注册账户'
            }, status=status.HTTP_404_NOT_FOUND)


class VerifyResetCodeView(APIView):
    """验证密码重置验证码"""
    def post(self, request):
        email = request.data.get('email')
        code = request.data.get('code')
        
        if not email or not code:
            return Response({
                'error': '请提供邮箱和验证码'
            }, status=status.HTTP_400_BAD_REQUEST)
        
        try:
            user = User.objects.get(email=email)
            
            # 检查验证码是否匹配
            if user.reset_password_token != code:
                return Response({
                    'error': '验证码错误',
                    'detail': '请检查验证码是否正确'
                }, status=status.HTTP_400_BAD_REQUEST)
            
            # 检查验证码是否过期
            if not user.is_reset_token_valid():
                return Response({
                    'error': '验证码已过期',
                    'detail': '验证码有效期为15分钟，请重新获取'
                }, status=status.HTTP_400_BAD_REQUEST)
            
            return Response({
                'message': '验证码正确',
                'detail': '请设置新密码'
            })
            
        except User.DoesNotExist:
            return Response({
                'error': '用户不存在'
            }, status=status.HTTP_404_NOT_FOUND)


class ResetPasswordView(APIView):
    """重置密码"""
    def post(self, request):
        email = request.data.get('email')
        code = request.data.get('code')
        new_password = request.data.get('new_password')
        
        if not all([email, code, new_password]):
            return Response({
                'error': '请提供完整信息',
                'detail': '需要邮箱、验证码和新密码'
            }, status=status.HTTP_400_BAD_REQUEST)
        
        try:
            user = User.objects.get(email=email)
            
            # 再次验证验证码
            if user.reset_password_token != code or not user.is_reset_token_valid():
                return Response({
                    'error': '验证码无效或已过期',
                    'detail': '请重新获取验证码'
                }, status=status.HTTP_400_BAD_REQUEST)
            
            # 重置密码
            user.set_password(new_password)
            user.reset_password_token = None
            user.reset_password_token_created = None
            user.save()
            
            return Response({
                'message': '密码重置成功',
                'detail': '请使用新密码登录'
            })
            
        except User.DoesNotExist:
            return Response({
                'error': '用户不存在'
            }, status=status.HTTP_404_NOT_FOUND)
