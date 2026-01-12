"""
视频下载器
负责从URL下载视频并保存到指定路径
"""
import requests
import os
from typing import Tuple, Optional
from pathlib import Path


class VideoDownloader:
    """视频下载器"""
    
    def __init__(self, base_dir: str = None):
        """
        初始化下载器
        
        Args:
            base_dir: 基础目录，默认为backend/media/gendata
        """
        if base_dir is None:
            # 获取当前文件的目录，向上两级到backend，然后到media/gendata
            current_file = Path(__file__).resolve()
            backend_dir = current_file.parent.parent
            self.base_dir = backend_dir / 'media' / 'gendata'
        else:
            self.base_dir = Path(base_dir)
        
        # 确保基础目录存在
        self.base_dir.mkdir(parents=True, exist_ok=True)
    
    def download(self, video_url: str, save_path: str, timeout: int = 120) -> Tuple[bool, Optional[str]]:
        """
        下载视频
        
        Args:
            video_url: 视频URL
            save_path: 保存路径（相对于base_dir）
            timeout: 超时时间（秒）
        
        Returns:
            (成功标志, 错误信息)
        """
        try:
            # 构建完整路径
            full_path = self.base_dir / save_path
            
            # 确保父目录存在
            full_path.parent.mkdir(parents=True, exist_ok=True)
            
            # 下载视频
            response = requests.get(video_url, stream=True, timeout=timeout)
            
            if response.status_code != 200:
                return False, f"HTTP {response.status_code}"
            
            # 写入文件
            with open(full_path, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    if chunk:
                        f.write(chunk)
            
            return True, None
            
        except requests.exceptions.Timeout:
            return False, "下载超时"
        except requests.exceptions.ConnectionError:
            return False, "连接失败"
        except Exception as e:
            return False, str(e)
    
    def generate_save_path(self, model_name: str, theme: str, video_id: str) -> str:
        """
        生成视频保存路径
        
        Args:
            model_name: 模型名称
            theme: 主题分类
            video_id: 视频ID
        
        Returns:
            相对于base_dir的路径，例如: jimeng_video_3/animals_and_ecology/animals_and_ecology_001.mp4
        """
        # 标准化模型名
        normalized_model = model_name.lower().replace('-', '_').replace(' ', '_')
        
        # 构建路径: model_name/theme/video_id.mp4
        return f"{normalized_model}/{theme}/{video_id}.mp4"
    
    def get_full_path(self, relative_path: str) -> Path:
        """
        获取完整路径
        
        Args:
            relative_path: 相对路径
        
        Returns:
            完整路径
        """
        return self.base_dir / relative_path
    
    def get_db_path(self, relative_path: str) -> str:
        """
        获取数据库保存路径（相对于media目录）
        
        Args:
            relative_path: 相对于gendata的路径
        
        Returns:
            相对于media的路径，例如: gendata/jimeng_video_3/animals_and_ecology/animals_and_ecology_001.mp4
        """
        return f"gendata/{relative_path}"
