from rest_framework import status, viewsets
from rest_framework.decorators import action
from rest_framework.response import Response
from .models import RefData, GenData
from .serializers import RefDataSerializer, GenDataSerializer


class RefDataViewSet(viewsets.ReadOnlyModelViewSet):
    """参考数据集视图 - 只读"""
    queryset = RefData.objects.all()
    serializer_class = RefDataSerializer
    
    @action(detail=False, methods=['get'])
    def themes(self, request):
        """获取所有主题的统计"""
        themes = RefData.objects.values('theme').distinct()
        theme_stats = []
        for theme_obj in themes:
            theme = theme_obj['theme']
            count = RefData.objects.filter(theme=theme).count()
            theme_stats.append({
                'theme': theme,
                'theme_display': dict(RefData.THEME_CHOICES).get(theme, theme),
                'count': count
            })
        return Response({
            'total': RefData.objects.count(),
            'themes': theme_stats
        })
    
    @action(detail=False, methods=['get'])
    def by_theme(self, request):
        """按主题筛选"""
        theme = request.query_params.get('theme')
        if theme:
            queryset = self.queryset.filter(theme=theme)
            serializer = self.get_serializer(queryset, many=True)
            return Response(serializer.data)
        return Response({'error': '请提供theme参数'}, status=status.HTTP_400_BAD_REQUEST)


class GenDataViewSet(viewsets.ModelViewSet):
    """生成数据视图"""
    queryset = GenData.objects.all()
    serializer_class = GenDataSerializer
    
    @action(detail=False, methods=['get'])
    def by_model(self, request):
        """按模型名筛选"""
        model_name = request.query_params.get('model_name')
        if model_name:
            queryset = self.queryset.filter(model_name=model_name)
            serializer = self.get_serializer(queryset, many=True)
            return Response(serializer.data)
        return Response({'error': '请提供model_name参数'}, status=status.HTTP_400_BAD_REQUEST)
