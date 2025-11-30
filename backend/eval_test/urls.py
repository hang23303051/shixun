from django.urls import path
from . import views

urlpatterns = [
    path('submit/', views.EvalSubmitView.as_view(), name='eval-submit'),
    path('status/<str:task_id>/', views.EvalStatusView.as_view(), name='eval-status'),
    path('mock-complete/<str:task_id>/', views.EvalMockCompleteView.as_view(), name='eval-mock-complete'),
]
