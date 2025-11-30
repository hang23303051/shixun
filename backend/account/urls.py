from django.urls import path
from . import views

urlpatterns = [
    path('csrf/', views.GetCSRFTokenView.as_view(), name='csrf'),
    path('register/', views.RegisterView.as_view(), name='register'),
    path('login/', views.LoginView.as_view(), name='login'),
    path('logout/', views.LogoutView.as_view(), name='logout'),
    path('profile/', views.UserProfileView.as_view(), name='profile'),
    path('check-login/', views.CheckLoginView.as_view(), name='check-login'),
]
