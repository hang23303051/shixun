from rest_framework import status
from rest_framework.views import APIView
from rest_framework.response import Response
from datetime import datetime
from account.models import User
from model_eval.models import Model
from .serializers import EvalRequestSerializer, EvalStatusSerializer
import uuid


# 临时存储评测任务状态（生产环境应使用Redis或Celery）
eval_tasks = {}


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
            return Response({'error': '该模型已存在，请使用不同的名称'}, status=status.HTTP_400_BAD_REQUEST)
        
        # 创建任务ID
        task_id = str(uuid.uuid4())
        
        # 初始化任务状态
        eval_tasks[task_id] = {
            'status': 'pending',
            'progress': 0,
            'message': '任务已创建，等待开始',
            'data': serializer.validated_data,
            'user_email': user_email,
            'username': user.username,
            'created_at': datetime.now().isoformat()
        }
        
        # TODO: 这里应该启动异步任务调用打分算法
        # 示例: start_evaluation_task(task_id, serializer.validated_data)
        
        return Response({
            'message': '评测任务已提交',
            'task_id': task_id
        }, status=status.HTTP_201_CREATED)


class EvalStatusView(APIView):
    """查询评测任务状态"""
    
    def get(self, request, task_id):
        if task_id not in eval_tasks:
            return Response({'error': '任务不存在'}, status=status.HTTP_404_NOT_FOUND)
        
        task = eval_tasks[task_id]
        response_data = {
            'task_id': task_id,
            'status': task['status'],
            'progress': task['progress'],
            'message': task.get('message', ''),
        }
        
        # 如果任务完成，返回结果
        if task['status'] == 'completed':
            response_data['result'] = task.get('result', {})
        
        return Response(response_data)


class EvalMockCompleteView(APIView):
    """
    模拟评测完成（开发调试用）
    生产环境中，这个endpoint应该被移除，由实际的打分算法调用
    """
    
    def post(self, request, task_id):
        if task_id not in eval_tasks:
            return Response({'error': '任务不存在'}, status=status.HTTP_404_NOT_FOUND)
        
        task = eval_tasks[task_id]
        
        # 模拟评分结果（实际应该由打分算法提供）
        scores = {
            'semantic_score': request.data.get('semantic_score', 85.5),
            'temporal_score': request.data.get('temporal_score', 82.3),
            'motion_score': request.data.get('motion_score', 88.7),
            'reality_score': request.data.get('reality_score', 80.1),
        }
        
        # 计算总分
        total_score = sum(scores.values()) / len(scores)
        
        # 创建模型记录
        model_data = task['data']
        model = Model.objects.create(
            name=model_data['model_name'],
            description=model_data.get('description', ''),
            publisher=model_data.get('publisher', '未知'),
            parameters=model_data.get('parameters', '未知'),
            is_open_source=model_data.get('is_open_source', False),
            release_date=model_data.get('release_date', datetime.now().date()),
            official_website=model_data.get('official_website', ''),
            semantic_score=scores['semantic_score'],
            temporal_score=scores['temporal_score'],
            motion_score=scores['motion_score'],
            reality_score=scores['reality_score'],
            total_score=total_score,
            tester_type='user',
            tester_name=task['username']
        )
        
        # 更新任务状态
        task['status'] = 'completed'
        task['progress'] = 100
        task['message'] = '评测完成'
        task['result'] = {
            'model_id': model.id,
            'model_name': model.name,
            'scores': {
                'semantic': model.semantic_score,
                'temporal': model.temporal_score,
                'motion': model.motion_score,
                'reality': model.reality_score,
                'total': model.total_score,
            }
        }
        
        return Response({
            'message': '评测完成',
            'result': task['result']
        })
