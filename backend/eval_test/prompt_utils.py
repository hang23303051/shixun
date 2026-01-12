"""
Prompt处理工具
用于读取和处理prompt文件
"""
import os
from typing import List, Dict, Tuple
from pathlib import Path


class PromptManager:
    """Prompt管理器"""
    
    def __init__(self, prompts_dir: str = None):
        """初始化Prompt管理器"""
        if prompts_dir is None:
            base_dir = Path(__file__).resolve().parent.parent
            prompts_dir = os.path.join(base_dir, 'media', 'refdata', 'prompts')
        self.prompts_dir = prompts_dir
    
    def load_all_prompts(self, limit: int = None) -> List[Dict]:
        """加载所有prompt文件"""
        prompts = []
        
        for theme_dir in os.listdir(self.prompts_dir):
            theme_path = os.path.join(self.prompts_dir, theme_dir)
            
            if not os.path.isdir(theme_path):
                continue
            
            for filename in os.listdir(theme_path):
                if not filename.endswith('.txt'):
                    continue
                
                file_path = os.path.join(theme_path, filename)
                video_id, theme, shot_type = self._parse_filename(filename)
                
                try:
                    with open(file_path, 'r', encoding='utf-8') as f:
                        prompt = f.read().strip()
                    
                    prompts.append({
                        'video_id': video_id,
                        'theme': theme,
                        'shot_type': shot_type,
                        'prompt': prompt,
                        'file_path': file_path
                    })
                    
                    if limit and len(prompts) >= limit:
                        return prompts
                except Exception as e:
                    print(f'读取文件失败 {file_path}: {str(e)}')
                    continue
        
        return prompts
    
    def _parse_filename(self, filename: str) -> Tuple[str, str, str]:
        """从文件名解析video_id、theme和shot_type"""
        name = filename.replace('.txt', '')
        parts = name.rsplit('_', 1)
        
        if len(parts) != 2:
            raise ValueError(f'无效的文件名格式：{filename}')
        
        shot_type = parts[1]
        video_id = name
        theme_parts = parts[0].rsplit('_', 1)
        theme = theme_parts[0] if len(theme_parts) >= 2 else parts[0]
        
        return video_id, theme, shot_type


def build_video_save_path(base_dir: str, model_name: str, theme: str, video_id: str) -> str:
    """构建视频保存路径"""
    return os.path.join(base_dir, model_name, theme, f'{video_id}.mp4')
