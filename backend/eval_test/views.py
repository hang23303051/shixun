from rest_framework import status
from rest_framework.views import APIView
from rest_framework.response import Response
from datetime import datetime
from account.models import User
from model_eval.models import Model
from task_list.models import TaskList
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
        
        # 检查模型名是否已存在（在Model表和TaskList表中都检查）
        model_name = serializer.validated_data['model_name']
        if Model.objects.filter(name=model_name).exists():
            return Response({'error': '该模型名称已存在，请使用不同的名称'}, status=status.HTTP_400_BAD_REQUEST)
        
        # 检查是否有相同模型名的任务正在处理或等待中
        if TaskList.objects.filter(model_name=model_name, status__in=['pending', 'processing']).exists():
            return Response({'error': '该模型名称已有任务正在处理中，请使用不同的名称或等待当前任务完成'}, status=status.HTTP_400_BAD_REQUEST)
        
        # 创建任务ID
        task_id = str(uuid.uuid4())
        
        # 保存到数据库 task_list 表
        task_data = serializer.validated_data
        task_record = TaskList.objects.create(
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
        
        # 初始化任务状态（内存中保留用于兼容现有逻辑）
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
        # 优先从数据库读取
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
            # 如果数据库中不存在，尝试从内存中读取（兼容旧逻辑）
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
        # 从数据库获取任务记录
        try:
            task_record = TaskList.objects.get(task_id=task_id)
        except TaskList.DoesNotExist:
            # 兼容旧逻辑：从内存中查找
            if task_id not in eval_tasks:
                return Response({'error': '任务不存在'}, status=status.HTTP_404_NOT_FOUND)
            task = eval_tasks[task_id]
            model_data = task['data']
            username = task['username']
        else:
            model_data = {
                'model_name': task_record.model_name,
                'description': task_record.description,
                'publisher': task_record.publisher,
                'parameters': task_record.parameters,
                'is_open_source': task_record.is_open_source,
                'release_date': task_record.release_date,
                'official_website': task_record.official_website,
            }
            username = task_record.username
        
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
            tester_name=username
        )
        
        # 更新数据库中的任务记录
        try:
            task_record = TaskList.objects.get(task_id=task_id)
            task_record.status = 'completed'
            task_record.progress = 100
            task_record.message = '评测完成'
            task_record.semantic_score = scores['semantic_score']
            task_record.temporal_score = scores['temporal_score']
            task_record.motion_score = scores['motion_score']
            task_record.reality_score = scores['reality_score']
            task_record.total_score = total_score
            task_record.model_id = model.id
            task_record.completed_at = datetime.now()
            task_record.save()
        except TaskList.DoesNotExist:
            pass
        
        # 更新内存中的任务状态（兼容旧逻辑）
        if task_id in eval_tasks:
            task = eval_tasks[task_id]
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
            'result': {
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
        })
