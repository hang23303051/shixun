from django.urls import path
from . import views

urlpatterns = [
    path('csrf/', views.GetCSRFTokenView.as_view(), name='csrf'),
    path('register/', views.RegisterView.as_view(), name='register'),
    path('login/', views.LoginView.as_view(), name='login'),
    path('logout/', views.LogoutView.as_view(), name='logout'),
    path('profile/', views.UserProfileView.as_view(), name='profile'),
    path('check-login/', views.CheckLoginView.as_view(), name='check-login'),
    
    # 账户激活
    path('activate/<str:email>/<str:token>/', views.ActivateAccountView.as_view(), name='activate'),
    path('resend-activation/', views.ResendActivationEmailView.as_view(), name='resend-activation'),
    
    # 密码重置
    path('request-password-reset/', views.RequestPasswordResetView.as_view(), name='request-password-reset'),
    path('verify-reset-code/', views.VerifyResetCodeView.as_view(), name='verify-reset-code'),
    path('reset-password/', views.ResetPasswordView.as_view(), name='reset-password'),
]
