from rest_framework import status, viewsets
from rest_framework.decorators import action
from rest_framework.response import Response
from .models import Model
from .serializers import ModelSerializer, ModelListSerializer, RankingSerializer


class ModelViewSet(viewsets.ModelViewSet):
    """模型视图集"""
    queryset = Model.objects.all()
    serializer_class = ModelSerializer
    
    def get_serializer_class(self):
        if self.action == 'list':
            return ModelListSerializer
        return ModelSerializer
    
    @action(detail=False, methods=['get'])
    def ranking(self, request):
        """
        获取排行榜
        参数: dimension - 可选值: total(默认), semantic, temporal, motion, reality
        """
        dimension = request.query_params.get('dimension', 'total')
        
        # 根据维度选择排序字段
        order_field_map = {
            'total': '-total_score',
            'semantic': '-semantic_score',
            'temporal': '-temporal_score',
            'motion': '-motion_score',
            'reality': '-reality_score',
        }
        
        order_field = order_field_map.get(dimension, '-total_score')
        models = Model.objects.all().order_by(order_field)
        
        # 构建排行榜数据
        ranking_data = []
        for rank, model in enumerate(models, 1):
            data = {
                'rank': rank,
                'model_id': model.id,
                'model_name': model.name,
                'total_score': model.total_score,
            }
            
            # 如果是单维度排行榜，添加该维度分数
            if dimension != 'total':
                score_field = f'{dimension}_score'
                data[score_field] = getattr(model, score_field)
            else:
                # 总榜包含所有维度分数
                data.update({
                    'semantic_score': model.semantic_score,
                    'temporal_score': model.temporal_score,
                    'motion_score': model.motion_score,
                    'reality_score': model.reality_score,
                })
            
            ranking_data.append(data)
        
        return Response({
            'dimension': dimension,
            'rankings': ranking_data
        })
    
    @action(detail=True, methods=['get'])
    def scores(self, request, pk=None):
        """获取模型的详细评分"""
        model = self.get_object()
        return Response({
            'model_name': model.name,
            'scores': {
                'semantic': model.semantic_score,
                'temporal': model.temporal_score,
                'motion': model.motion_score,
                'reality': model.reality_score,
                'total': model.total_score,
            }
        })
