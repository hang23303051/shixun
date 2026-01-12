"""
账户工具函数模块
"""
import socket


def get_host_address(request):
    """
    获取主机的实际地址（IP或域名）
    
    Args:
        request: Django request对象
    
    Returns:
        str: 主机地址，包含协议和端口（如有）
    """
    # 1. 优先从请求头获取Host（浏览器直接访问的地址）
    host = request.get_host()
    
    # 2. 获取请求协议（http或https）
    scheme = 'https' if request.is_secure() else 'http'
    
    # 3. 如果获取到的host是localhost或127.0.0.1，尝试获取局域网IP
    if host.startswith('localhost') or host.startswith('127.0.0.1'):
        try:
            # 获取本机hostname
            hostname = socket.gethostname()
            # 获取本机IP地址
            local_ip = socket.gethostbyname(hostname)
            # 如果获取到的不是127.0.0.1，使用该IP
            if local_ip != '127.0.0.1':
                # 保留端口号（如果有）
                port = ''
                if ':' in host:
                    port = ':' + host.split(':')[1]
                host = local_ip + port
        except Exception:
            # 如果获取失败，保持原样
            pass
    
    return f"{scheme}://{host}"
