"""
视频生成模型适配器
支持多个AI视频生成模型的统一调用接口
"""
import requests
import time
import json
from typing import Tuple, Optional


class BaseModelAdapter:
    """模型适配器基类"""
    
    def __init__(self, api_key: str, base_url: str = "https://yunwu.ai"):
        self.api_key = api_key
        self.base_url = base_url
    
    def create_video(self, prompt: str, **kwargs) -> Tuple[bool, Optional[str], Optional[str]]:
        """
        创建视频任务
        
        Returns:
            (成功标志, 任务ID, 错误信息)
        """
        raise NotImplementedError
    
    def query_status(self, task_id: str, max_wait: int = 300) -> Tuple[bool, Optional[str], Optional[str]]:
        """
        查询任务状态并等待完成
        
        Returns:
            (成功标志, 视频URL, 错误信息)
        """
        raise NotImplementedError
    
    def _make_request(self, method: str, url: str, headers: dict = None, 
                     json_data: dict = None, timeout: int = 30) -> Tuple[bool, dict]:
        """统一的HTTP请求方法"""
        try:
            if headers is None:
                headers = {
                    'Authorization': f'Bearer {self.api_key}',
                    'Content-Type': 'application/json'
                }
            
            if method.upper() == 'POST':
                response = requests.post(url, headers=headers, json=json_data, timeout=timeout)
            else:
                response = requests.get(url, headers=headers, timeout=timeout)
            
            if response.status_code == 200:
                return True, response.json()
            else:
                return False, {"error": f"HTTP {response.status_code}", "detail": response.text}
        except Exception as e:
            return False, {"error": str(e)}


class JiMengAdapter(BaseModelAdapter):
    """即梦视频3.0适配器"""
    
    def create_video(self, prompt: str, **kwargs) -> Tuple[bool, Optional[str], Optional[str]]:
        url = f"{self.base_url}/v1/video/create"
        payload = {
            "model": "jimeng-video-3.0",
            "prompt": prompt
        }
        
        success, data = self._make_request('POST', url, json_data=payload)
        if success:
            task_id = data.get('id')
            return True, task_id, None
        else:
            return False, None, data.get('error', '未知错误')
    
    def query_status(self, task_id: str, max_wait: int = 300) -> Tuple[bool, Optional[str], Optional[str]]:
        url = f"{self.base_url}/v1/video/query?id={task_id}"
        start_time = time.time()
        
        while time.time() - start_time < max_wait:
            success, data = self._make_request('GET', url)
            if not success:
                time.sleep(5)
                continue
            
            status = data.get('status', '')
            if status == 'completed' or status == 'success':
                video_url = data.get('video_url')
                return True, video_url, None
            elif status == 'failed' or status == 'error':
                return False, None, data.get('error', '生成失败')
            
            time.sleep(5)
        
        return False, None, "等待超时"


class VeoAdapter(BaseModelAdapter):
    """Veo3.1适配器"""
    
    def create_video(self, prompt: str, **kwargs) -> Tuple[bool, Optional[str], Optional[str]]:
        url = f"{self.base_url}/v1/video/create"
        payload = {
            "model": "veo3.1-fast",
            "prompt": prompt,
            "upsampling": True,
            "prompt_enhancer": True
        }
        
        success, data = self._make_request('POST', url, json_data=payload)
        if success:
            task_id = data.get('id')
            return True, task_id, None
        else:
            return False, None, data.get('error', '未知错误')
    
    def query_status(self, task_id: str, max_wait: int = 300) -> Tuple[bool, Optional[str], Optional[str]]:
        url = f"{self.base_url}/v1/video/query?id={task_id}"
        start_time = time.time()
        
        while time.time() - start_time < max_wait:
            success, data = self._make_request('GET', url)
            if not success:
                time.sleep(5)
                continue
            
            status = data.get('status', '')
            if status == 'completed' or status == 'success':
                video_url = data.get('video_url')
                return True, video_url, None
            elif status == 'failed' or status == 'error':
                return False, None, data.get('error', '生成失败')
            
            time.sleep(5)
        
        return False, None, "等待超时"


class GrokAdapter(BaseModelAdapter):
    """Grok视频3适配器"""
    
    def create_video(self, prompt: str, **kwargs) -> Tuple[bool, Optional[str], Optional[str]]:
        url = f"{self.base_url}/v1/video/create"
        payload = {
            "model": "grok-video-3",
            "prompt": prompt,
            "aspect_ratio": "3:2"
        }
        
        success, data = self._make_request('POST', url, json_data=payload)
        if success:
            task_id = data.get('id')
            return True, task_id, None
        else:
            return False, None, data.get('error', '未知错误')
    
    def query_status(self, task_id: str, max_wait: int = 300) -> Tuple[bool, Optional[str], Optional[str]]:
        url = f"{self.base_url}/v1/video/query?id={task_id}"
        start_time = time.time()
        
        while time.time() - start_time < max_wait:
            success, data = self._make_request('GET', url)
            if not success:
                time.sleep(5)
                continue
            
            status = data.get('status', '')
            if status == 'completed' or status == 'success':
                video_url = data.get('video_url')
                return True, video_url, None
            elif status == 'failed' or status == 'error':
                return False, None, data.get('error', '生成失败')
            
            time.sleep(5)
        
        return False, None, "等待超时"


class DoubaoAdapter(BaseModelAdapter):
    """豆包Seedance适配器"""
    
    def create_video(self, prompt: str, **kwargs) -> Tuple[bool, Optional[str], Optional[str]]:
        url = f"{self.base_url}/volc/v1/contents/generations/tasks"
        payload = {
            "model": "doubao-seedance-1-0-pro-fast-251015",
            "content": [
                {
                    "type": "text",
                    "text": f"{prompt} --ratio 16:9"
                }
            ]
        }
        
        success, data = self._make_request('POST', url, json_data=payload)
        if success:
            task_id = data.get('task_id') or data.get('id')
            return True, task_id, None
        else:
            return False, None, data.get('error', '未知错误')
    
    def query_status(self, task_id: str, max_wait: int = 300) -> Tuple[bool, Optional[str], Optional[str]]:
        url = f"{self.base_url}/volc/v1/contents/generations/tasks/{task_id}"
        start_time = time.time()
        
        while time.time() - start_time < max_wait:
            success, data = self._make_request('GET', url)
            if not success:
                time.sleep(5)
                continue
            
            # 豆包的状态在data.status
            status = None
            if 'data' in data and isinstance(data['data'], dict):
                status = data['data'].get('status')
                if not status and 'data' in data['data']:
                    inner_data = data['data'].get('data', {})
                    if isinstance(inner_data, dict):
                        status = inner_data.get('status')
            
            if not status:
                status = data.get('status')
            
            if status in ['SUCCESS', 'Success', 'completed', 'success', 'succeeded']:
                # 提取视频URL
                video_url = None
                if 'content' in data and isinstance(data['content'], dict):
                    video_url = data['content'].get('video_url')
                
                if video_url:
                    return True, video_url, None
                else:
                    return False, None, "未找到视频URL"
            elif status in ['FAILED', 'failed', 'error']:
                return False, None, data.get('error', '生成失败')
            
            time.sleep(5)
        
        return False, None, "等待超时"


class LumaAdapter(BaseModelAdapter):
    """Luma Ray-V2适配器"""
    
    def create_video(self, prompt: str, **kwargs) -> Tuple[bool, Optional[str], Optional[str]]:
        url = f"{self.base_url}/luma/generations"
        payload = {
            "user_prompt": prompt,
            "model_name": "ray-v2",
            "duration": "5s",
            "resolution": "720p"
        }
        
        success, data = self._make_request('POST', url, json_data=payload)
        if success:
            task_id = data.get('id')
            return True, task_id, None
        else:
            return False, None, data.get('error', '未知错误')
    
    def query_status(self, task_id: str, max_wait: int = 300) -> Tuple[bool, Optional[str], Optional[str]]:
        url = f"{self.base_url}/luma/generations/{task_id}"
        start_time = time.time()
        
        while time.time() - start_time < max_wait:
            success, data = self._make_request('GET', url)
            if not success:
                time.sleep(5)
                continue
            
            state = data.get('state')
            if state == 'completed':
                # 提取视频URL
                video = data.get('video', {})
                video_url = video.get('url') if isinstance(video, dict) else None
                if video_url:
                    return True, video_url, None
                else:
                    return False, None, "未找到视频URL"
            elif state == 'failed':
                return False, None, data.get('failure_reason', '生成失败')
            
            time.sleep(5)
        
        return False, None, "等待超时"


class HailuoAdapter(BaseModelAdapter):
    """海螺MiniMax-Hailuo适配器"""
    
    def create_video(self, prompt: str, **kwargs) -> Tuple[bool, Optional[str], Optional[str]]:
        url = f"{self.base_url}/minimax/v1/video_generation"
        payload = {
            "model": "MiniMax-Hailuo-02",
            "prompt": prompt,
            "duration": 10
        }
        
        success, data = self._make_request('POST', url, json_data=payload)
        if success:
            task_id = data.get('task_id') or data.get('id')
            return True, task_id, None
        else:
            return False, None, data.get('error', '未知错误')
    
    def query_status(self, task_id: str, max_wait: int = 300) -> Tuple[bool, Optional[str], Optional[str]]:
        url = f"{self.base_url}/minimax/v1/query/video_generation?task_id={task_id}"
        start_time = time.time()
        
        while time.time() - start_time < max_wait:
            success, data = self._make_request('GET', url)
            if not success:
                time.sleep(5)
                continue
            
            # 海螺的status在data.status或data.data.status
            status = None
            if 'data' in data and isinstance(data['data'], dict):
                status = data['data'].get('status')
                if not status and 'data' in data['data']:
                    inner_data = data['data'].get('data', {})
                    if isinstance(inner_data, dict):
                        status = inner_data.get('status')
            
            if not status:
                status = data.get('status')
            
            if status in ['SUCCESS', 'Success', 'completed', 'success']:
                # 提取视频URL (在data.data.file.download_url)
                video_url = None
                if 'data' in data and isinstance(data['data'], dict):
                    data_obj = data['data']
                    if 'data' in data_obj and isinstance(data_obj['data'], dict):
                        inner_data = data_obj['data']
                        if 'file' in inner_data and isinstance(inner_data['file'], dict):
                            video_url = inner_data['file'].get('download_url')
                
                if video_url:
                    return True, video_url, None
                else:
                    return False, None, "未找到视频URL"
            elif status in ['Failed', 'failed', 'error']:
                return False, None, data.get('error_message', '生成失败')
            
            time.sleep(5)
        
        return False, None, "等待超时"


# 模型适配器工厂
MODEL_ADAPTERS = {
    'jimeng_video_3': JiMengAdapter,
    'jimeng': JiMengAdapter,
    'veo3_1': VeoAdapter,
    'veo': VeoAdapter,
    'grok_video_3': GrokAdapter,
    'grok': GrokAdapter,
    'doubao_seedance': DoubaoAdapter,
    'doubao': DoubaoAdapter,
    'luma_ray_v2': LumaAdapter,
    'luma': LumaAdapter,
    'minimax_hailuo': HailuoAdapter,
    'hailuo': HailuoAdapter,
}


def get_model_adapter(model_name: str, api_key: str) -> Optional[BaseModelAdapter]:
    """
    根据模型名称获取对应的适配器实例
    
    Args:
        model_name: 模型名称
        api_key: API密钥
    
    Returns:
        模型适配器实例，如果模型不支持则返回None
    """
    # 标准化模型名（转小写，替换空格和特殊字符）
    normalized_name = model_name.lower().replace('-', '_').replace(' ', '_')
    
    adapter_class = MODEL_ADAPTERS.get(normalized_name)
    if adapter_class:
        return adapter_class(api_key)
    
    return None
