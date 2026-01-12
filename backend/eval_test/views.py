from rest_framework import status
from rest_framework.views import APIView
from rest_framework.response import Response
from account.models import User
from model_eval.models import Model
from task_list.models import TaskList
from .serializers import EvalRequestSerializer
import uuid


class EvalSubmitView(APIView):
    """提交评测任务"""
    
    def post(self, request):
        # 检查登录状态
        user_email = request.session.get('user_email')
        if not user_email:
            return Response({'error': '请先登录'}, status=status.HTTP_401_UNAUTHORIZED)
        
        try:
            user = User.objects.get(email=user_email)
        except User.DoesNotExist:
            return Response({'error': '用户不存在'}, status=status.HTTP_404_NOT_FOUND)
        
        # 验证请求数据
        serializer = EvalRequestSerializer(data=request.data)
        if not serializer.is_valid():
            return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)
        
        # 检查模型名是否已存在
        model_name = serializer.validated_data['model_name']
        if Model.objects.filter(name=model_name).exists():
            return Response({'error': '该模型名称已存在，请使用不同的名称'}, status=status.HTTP_400_BAD_REQUEST)
        
        # 检查是否有相同模型名的任务正在处理或等待中
        if TaskList.objects.filter(model_name=model_name, status__in=['pending', 'processing']).exists():
            return Response({'error': '该模型名称已有任务正在处理中，请使用不同的名称或等待当前任务完成'}, 
                          status=status.HTTP_400_BAD_REQUEST)
        
        # 创建任务ID
        task_id = str(uuid.uuid4())
        
        # 保存到数据库 task_list 表
        task_data = serializer.validated_data
        TaskList.objects.create(
            task_id=task_id,
            user_email=user_email,
            username=user.username,
            model_name=task_data['model_name'],
            api_url=task_data['api_url'],
            api_key=task_data['api_key'],
            description=task_data.get('description', ''),
            publisher=task_data.get('publisher', ''),
            parameters=task_data.get('parameters', ''),
            is_open_source=task_data.get('is_open_source', False),
            release_date=task_data.get('release_date'),
            official_website=task_data.get('official_website', ''),
            status='pending',
            progress=0,
            message='任务已创建，等待开始'
        )
        
        # 启动Celery异步任务
        from .tasks import process_evaluation_task
        process_evaluation_task.delay(task_id)
        
        return Response({
            'message': '评测任务已提交，正在后台处理',
            'task_id': task_id
        }, status=status.HTTP_201_CREATED)


class EvalStatusView(APIView):
    """查询评测任务状态"""
    
    def get(self, request, task_id):
        # 从数据库读取任务状态
        try:
            task_record = TaskList.objects.get(task_id=task_id)
            response_data = {
                'task_id': task_id,
                'status': task_record.status,
                'progress': task_record.progress,
                'message': task_record.message,
            }
            
            # 如果任务完成，返回结果
            if task_record.status == 'completed':
                response_data['result'] = {
                    'model_id': task_record.model_id,
                    'model_name': task_record.model_name,
                    'scores': {
                        'semantic': task_record.semantic_score,
                        'temporal': task_record.temporal_score,
                        'motion': task_record.motion_score,
                        'reality': task_record.reality_score,
                        'total': task_record.total_score,
                    }
                }
            
            return Response(response_data)
        except TaskList.DoesNotExist:
            return Response({'error': '任务不存在'}, status=status.HTTP_404_NOT_FOUND)
