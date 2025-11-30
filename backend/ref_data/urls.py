from django.urls import path, include
from rest_framework.routers import DefaultRouter
from . import views

router = DefaultRouter()
router.register(r'refdata', views.RefDataViewSet)
router.register(r'gendata', views.GenDataViewSet)

urlpatterns = [
    path('', include(router.urls)),
]
