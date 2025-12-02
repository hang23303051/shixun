from rest_framework import status, viewsets
from rest_framework.decorators import action
from rest_framework.response import Response
from .models import TaskList
from .serializers import (
    TaskListSerializer, 
    TaskListStatusSerializer, 
    TaskListListSerializer
)


class TaskListViewSet(viewsets.ModelViewSet):
    """
    任务列表视图集
    提供查询任务列表、任务详情和删除功能
    """
    queryset = TaskList.objects.all()
    serializer_class = TaskListSerializer
    lookup_field = 'task_id'
    http_method_names = ['get', 'delete']  # 只允许GET和DELETE方法
    
    def get_serializer_class(self):
        if self.action == 'list':
            return TaskListListSerializer
        elif self.action == 'retrieve':
            return TaskListStatusSerializer
        return TaskListSerializer
    
    def get_queryset(self):
        """根据用户过滤任务列表 - 每个用户只能看到自己的任务"""
        user_email = self.request.session.get('user_email')
        
        if not user_email:
            # 未登录用户返回空查询集
            return TaskList.objects.none()
        
        # 只返回当前登录用户的任务
        return TaskList.objects.filter(user_email=user_email)
    
    @action(detail=False, methods=['get'])
    def my_tasks(self, request):
        """获取当前用户的任务列表"""
        user_email = request.session.get('user_email')
        if not user_email:
            return Response({'error': '请先登录'}, status=status.HTTP_401_UNAUTHORIZED)
        
        tasks = TaskList.objects.filter(user_email=user_email)
        serializer = TaskListListSerializer(tasks, many=True)
        return Response(serializer.data)
    
    @action(detail=False, methods=['get'])
    def statistics(self, request):
        """获取当前用户的任务统计信息"""
        user_email = request.session.get('user_email')
        if not user_email:
            return Response({'error': '请先登录'}, status=status.HTTP_401_UNAUTHORIZED)
        
        # 只统计当前用户的任务
        user_tasks = TaskList.objects.filter(user_email=user_email)
        total = user_tasks.count()
        pending = user_tasks.filter(status='pending').count()
        processing = user_tasks.filter(status='processing').count()
        completed = user_tasks.filter(status='completed').count()
        failed = user_tasks.filter(status='failed').count()
        
        return Response({
            'total': total,
            'pending': pending,
            'processing': processing,
            'completed': completed,
            'failed': failed
        })
    
    def destroy(self, request, *args, **kwargs):
        """删除任务，如果任务已完成且创建了模型，同时删除关联的模型"""
        user_email = request.session.get('user_email')
        if not user_email:
            return Response({'error': '请先登录'}, status=status.HTTP_401_UNAUTHORIZED)
        
        task = self.get_object()
        
        # 再次验证任务属于当前用户（双重保护）
        if task.user_email != user_email:
            return Response({'error': '无权删除此任务'}, status=status.HTTP_403_FORBIDDEN)
        
        # 如果任务已完成且有关联的模型，删除模型
        if task.status == 'completed' and task.model_id:
            from model_eval.models import Model
            try:
                model = Model.objects.get(id=task.model_id)
                model.delete()
            except Model.DoesNotExist:
                pass
        
        # 删除任务记录
        task.delete()
        
        return Response({
            'message': '任务已删除'
        }, status=status.HTTP_200_OK)
