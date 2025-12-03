"""
自定义中间件 - 动态添加CSRF信任域名
用于支持局域网访问时的CSRF验证
"""
from django.conf import settings


class DynamicCSRFTrustedOriginsMiddleware:
    """
    动态添加请求来源到CSRF_TRUSTED_ORIGINS
    解决局域网IP访问时的CSRF验证问题
    """
    def __init__(self, get_response):
        self.get_response = get_response

    def __call__(self, request):
        # 获取请求的Origin
        origin = request.headers.get('Origin')
        
        if origin:
            # 如果origin不在信任列表中，动态添加
            if origin not in settings.CSRF_TRUSTED_ORIGINS:
                # 只添加本地和局域网IP
                if self._is_local_or_lan_origin(origin):
                    settings.CSRF_TRUSTED_ORIGINS.append(origin)
                    print(f"✅ 动态添加CSRF信任域名: {origin}")
        
        response = self.get_response(request)
        return response
    
    def _is_local_or_lan_origin(self, origin):
        """
        判断是否为本地或局域网地址
        """
        local_patterns = [
            'localhost',
            '127.0.0.1',
            '192.168.',  # 常见内网
            '10.',       # 常见内网
            '172.16.',   # 常见内网
            '172.17.',
            '172.18.',
            '172.19.',
            '172.20.',
            '172.21.',
            '172.22.',
            '172.23.',
            '172.24.',
            '172.25.',
            '172.26.',
            '172.27.',
            '172.28.',
            '172.29.',
            '172.30.',
            '172.31.',
        ]
        
        return any(pattern in origin for pattern in local_patterns)
