# Ref4D 视频生成模型评测平台 - 技术说明文档

## 目录
- [1. 项目概述](#1-项目概述)
- [2. 技术栈](#2-技术栈)
- [3. 项目结构](#3-项目结构)
- [4. 数据库设计](#4-数据库设计)
- [5. 后端架构设计](#5-后端架构设计)
- [6. 前端架构设计](#6-前端架构设计)
- [7. 关键业务实现](#7-关键业务实现)
- [8. 打分算法集成指南](#8-打分算法集成指南)
- [9. 部署指南](#9-部署指南)

---

## 1. 项目概述

### 1.1 项目简介
Ref4D是一个视频生成模型评测平台，用于评测和展示各种视频生成模型的性能。平台支持用户注册、模型评测提交、排行榜展示等功能。

### 1.2 核心功能
- **用户系统**：注册、登录、个人信息管理、头像上传
- **数据集展示**：展示参考数据集信息和统计
- **模型管理**：模型列表、详情、四维度评分展示
- **排行榜**：支持总分和各维度排序
- **评测系统**：提交评测任务、查询状态、模拟完成

### 1.3 评分维度
1. **基础语义一致性** (Semantic Score)
2. **时序一致性** (Temporal Score)
3. **运动属性** (Motion Score)
4. **世界知识真实性** (Reality Score)

---

## 2. 技术栈

### 2.1 后端技术栈
- **语言**：Python 3.x
- **框架**：Django 5.2.8
- **REST API**：Django REST Framework
- **数据库**：MySQL 8.0
- **跨域处理**：django-cors-headers
- **认证方式**：Django Session（Cookie-based）

### 2.2 前端技术栈
- **框架**：Vue 3 (Composition API)
- **路由**：Vue Router 4
- **状态管理**：Vuex 4
- **HTTP客户端**：Axios
- **UI样式**：Tailwind CSS 3
- **图表库**：ECharts 5 + vue-echarts

### 2.3 开发工具
- **包管理**：npm (前端), pip (后端)
- **开发服务器**：Vue CLI Dev Server + Django runserver
- **版本控制**：Git

---

## 3. 项目结构

```
shixun/
├── backend/                    # Django后端
│   ├── account/               # 用户账户应用
│   │   ├── models.py         # User模型
│   │   ├── serializers.py    # 序列化器
│   │   ├── views.py          # API视图
│   │   ├── urls.py           # 路由配置
│   │   └── admin.py          # 后台管理
│   ├── ref_data/             # 参考数据应用
│   │   ├── models.py         # RefData、GenData模型
│   │   ├── serializers.py
│   │   ├── views.py
│   │   └── urls.py
│   ├── model_eval/           # 模型评测应用
│   │   ├── models.py         # Model模型
│   │   ├── serializers.py
│   │   ├── views.py
│   │   └── urls.py
│   ├── eval_test/            # 评测任务应用
│   │   ├── models.py         # (暂无数据库模型)
│   │   ├── serializers.py
│   │   ├── views.py          # ⭐ 打分算法集成位置
│   │   └── urls.py
│   ├── backend/              # 项目配置
│   │   ├── settings.py       # 核心配置
│   │   └── urls.py           # 根路由
│   ├── media/                # 媒体文件目录
│   │   └── avatars/          # 用户头像
│   └── manage.py
├── frontend/                  # Vue前端
│   ├── src/
│   │   ├── api/              # API封装
│   │   │   ├── axios.js      # Axios配置
│   │   │   ├── account.js    # 账户API
│   │   │   ├── data.js       # 数据API
│   │   │   ├── model.js      # 模型API
│   │   │   └── eval.js       # 评测API
│   │   ├── components/       # 公共组件
│   │   │   ├── Header.vue
│   │   │   └── ScoreChart.vue
│   │   ├── views/            # 页面组件
│   │   │   ├── Home.vue
│   │   │   ├── Login.vue
│   │   │   ├── Register.vue
│   │   │   ├── Models.vue
│   │   │   ├── ModelDetail.vue
│   │   │   ├── Ranking.vue
│   │   │   ├── Evaluation.vue
│   │   │   ├── Dataset.vue
│   │   │   └── Profile.vue
│   │   ├── router/           # 路由配置
│   │   ├── store/            # Vuex状态管理
│   │   ├── assets/           # 静态资源
│   │   └── App.vue
│   ├── public/
│   └── package.json
├── venv/                      # Python虚拟环境
├── .gitignore
├── README.md
└── 需求文档.md
```

---

## 4. 数据库设计

### 4.1 数据库配置
- **数据库名**：`ref4d`
- **字符集**：utf8mb4
- **连接信息**：见 `backend/backend/settings.py`

### 4.2 数据表结构

#### 4.2.1 user (用户表)
```sql
CREATE TABLE user (
    email VARCHAR(255) PRIMARY KEY,           -- 邮箱（主键）
    username VARCHAR(150) NOT NULL,           -- 用户名
    password VARCHAR(255) NOT NULL,           -- 加密密码
    avatar VARCHAR(100),                      -- 头像路径
    created_at DATETIME NOT NULL,             -- 创建时间
    updated_at DATETIME NOT NULL              -- 更新时间
);
```

**字段说明**：
- `email`：用户唯一标识，作为主键
- `password`：使用Django的`make_password()`加密存储
- `avatar`：存储相对路径，如 `avatars/xxx.jpg`

#### 4.2.2 ref_data (参考数据表)
```sql
CREATE TABLE ref_data (
    id BIGINT AUTO_INCREMENT PRIMARY KEY,
    text_prompt TEXT NOT NULL,                -- 文本提示词
    video_path VARCHAR(500) NOT NULL,         -- 视频路径
    category VARCHAR(100),                    -- 视频类别
    duration FLOAT,                           -- 时长(秒)
    resolution VARCHAR(50),                   -- 分辨率
    frame_rate INT,                           -- 帧率
    created_at DATETIME NOT NULL
);
```

#### 4.2.3 gen_data (生成数据表)
```sql
CREATE TABLE gen_data (
    id BIGINT AUTO_INCREMENT PRIMARY KEY,
    ref_id BIGINT,                            -- 关联ref_data
    model_name VARCHAR(100) NOT NULL,         -- 模型名称
    video_path VARCHAR(500) NOT NULL,         -- 生成视频路径
    semantic_score FLOAT,                     -- 语义一致性评分
    temporal_score FLOAT,                     -- 时序一致性评分
    motion_score FLOAT,                       -- 运动属性评分
    reality_score FLOAT,                      -- 真实性评分
    generation_time FLOAT,                    -- 生成耗时(秒)
    created_at DATETIME NOT NULL,
    FOREIGN KEY (ref_id) REFERENCES ref_data(id)
);
```

#### 4.2.4 model (模型信息表)
```sql
CREATE TABLE model (
    id BIGINT AUTO_INCREMENT PRIMARY KEY,
    name VARCHAR(100) UNIQUE NOT NULL,        -- 模型名称（唯一）
    description TEXT NOT NULL,                -- 文字简介
    publisher VARCHAR(200) NOT NULL,          -- 发布者
    parameters VARCHAR(100) NOT NULL,         -- 参数规模
    is_open_source BOOLEAN DEFAULT FALSE,     -- 是否开源
    release_date DATE NOT NULL,               -- 发布时间
    official_website VARCHAR(500) NOT NULL,   -- 官网链接
    semantic_score FLOAT NOT NULL,            -- 语义评分
    temporal_score FLOAT NOT NULL,            -- 时序评分
    motion_score FLOAT NOT NULL,              -- 运动评分
    reality_score FLOAT NOT NULL,             -- 真实性评分
    total_score FLOAT NOT NULL,               -- 总分（自动计算）
    tester_type VARCHAR(10) NOT NULL,         -- 测试人类型(admin/user)
    tester_name VARCHAR(150) NOT NULL,        -- 测试人姓名
    created_at DATETIME NOT NULL,
    updated_at DATETIME NOT NULL
);
```

**业务逻辑**：
- `total_score = (semantic_score + temporal_score + motion_score + reality_score) / 4`
- 在Model的`save()`方法中自动计算

---

## 5. 后端架构设计

### 5.1 Django应用划分

#### 5.1.1 account (用户账户)
**职责**：用户注册、登录、个人信息管理

**核心API**：
- `POST /api/account/register/` - 用户注册
- `POST /api/account/login/` - 用户登录
- `POST /api/account/logout/` - 用户登出
- `GET /api/account/profile/` - 获取个人信息
- `PUT /api/account/profile/` - 更新个人信息（含头像）
- `GET /api/account/check-login/` - 检查登录状态
- `GET /api/account/csrf/` - 获取CSRF Token

**关键实现**：
```python
# models.py
class User(models.Model):
    def set_password(self, raw_password):
        """加密存储密码"""
        self.password = make_password(raw_password)
    
    def check_password(self, raw_password):
        """验证密码"""
        return check_password(raw_password, self.password)
```

#### 5.1.2 ref_data (参考数据)
**职责**：管理参考视频数据集

**核心API**：
- `GET /api/data/ref-data/` - 获取参考数据列表（支持分页、过滤）
- `GET /api/data/gen-data/` - 获取生成数据列表
- `GET /api/data/stats/` - 获取数据集统计信息

#### 5.1.3 model_eval (模型评测)
**职责**：模型信息展示和排行榜

**核心API**：
- `GET /api/model/models/` - 获取模型列表
- `GET /api/model/models/<id>/` - 获取模型详情
- `GET /api/model/ranking/` - 获取排行榜（支持维度筛选）

#### 5.1.4 eval_test (评测任务) ⭐
**职责**：处理评测任务的提交和状态查询

**核心API**：
- `POST /api/eval/submit/` - 提交评测任务
- `GET /api/eval/status/<task_id>/` - 查询任务状态
- `POST /api/eval/mock-complete/` - 模拟完成（开发用）

**任务状态管理**：
```python
eval_tasks = {}  # 内存存储，生产环境应使用Redis/数据库

# 状态类型：
# - pending: 等待处理
# - processing: 处理中
# - completed: 已完成
# - failed: 失败
```

### 5.2 认证机制

使用Django Session认证：
- 登录成功后，`request.session['user_email'] = email`
- 需要登录的API检查session中是否有user_email
- CSRF保护：非GET请求需要CSRF Token

### 5.3 CORS配置

```python
# settings.py
CORS_ALLOWED_ORIGINS = [
    "http://localhost:8080",
    "http://127.0.0.1:8080",
]
CORS_ALLOW_CREDENTIALS = True
```

### 5.4 Media文件处理

```python
# settings.py
MEDIA_URL = '/media/'
MEDIA_ROOT = BASE_DIR / 'media'

# urls.py
urlpatterns += static(settings.MEDIA_URL, document_root=settings.MEDIA_ROOT)
```

---

## 6. 前端架构设计

### 6.1 Axios配置

```javascript
// src/api/axios.js
const instance = axios.create({
  baseURL: '/api',
  timeout: 30000,
  withCredentials: true  // 携带Cookie
})

// CSRF Token处理
instance.interceptors.request.use(config => {
  if (config.method !== 'get') {
    const csrftoken = getCookie('csrftoken')
    if (csrftoken) {
      config.headers['X-CSRFToken'] = csrftoken
    }
  }
  // FormData自动处理
  if (!(config.data instanceof FormData)) {
    config.headers['Content-Type'] = 'application/json'
  }
  return config
})
```

### 6.2 路由守卫

```javascript
// src/router/index.js
router.beforeEach((to, from, next) => {
  const requiresAuth = to.matched.some(record => record.meta.requiresAuth)
  const isLoggedIn = store.getters.isLoggedIn
  
  if (requiresAuth && !isLoggedIn) {
    next('/login')
  } else {
    next()
  }
})
```

### 6.3 Vuex状态管理

```javascript
// src/store/index.js
export default createStore({
  state: {
    user: null,
    isLoggedIn: false
  },
  mutations: {
    SET_USER(state, user) {
      state.user = user
      state.isLoggedIn = !!user
    }
  },
  actions: {
    async checkLogin({ commit }) {
      const data = await accountAPI.checkLogin()
      if (data.logged_in) {
        commit('SET_USER', data.user)
      }
    }
  }
})
```

### 6.4 代理配置

```javascript
// vue.config.js
module.exports = {
  devServer: {
    proxy: {
      '/api': {
        target: 'http://localhost:8000',
        changeOrigin: true
      },
      '/media': {
        target: 'http://localhost:8000',
        changeOrigin: true
      }
    }
  }
}
```

---

## 7. 关键业务实现

### 7.1 用户注册流程

**前端流程**：
1. 用户填写表单（邮箱、用户名、密码、确认密码）
2. 前端验证（邮箱格式、密码一致性）
3. 发送POST请求到 `/api/account/register/`
4. 注册成功后跳转到登录页

**后端流程**：
```python
# account/views.py - RegisterView
def post(self, request):
    serializer = UserRegisterSerializer(data=request.data)
    if serializer.is_valid():
        user = serializer.save()  # 自动加密密码
        return Response({'message': '注册成功'})
    return Response(serializer.errors, status=400)
```

### 7.2 用户登录流程

**前端流程**：
1. 用户输入用户名和密码
2. 发送POST请求到 `/api/account/login/`
3. 登录成功后存储用户信息到Vuex
4. 跳转到首页

**后端流程**：
```python
# account/views.py - LoginView
def post(self, request):
    serializer = UserLoginSerializer(data=request.data)
    if serializer.is_valid():
        username = serializer.validated_data['username']
        password = serializer.validated_data['password']
        
        # 查找用户并验证密码
        user = User.objects.filter(username=username).first()
        if user and user.check_password(password):
            request.session['user_email'] = user.email
            return Response({'message': '登录成功'})
        
        return Response({'error': '用户名或密码错误'}, status=400)
```

### 7.3 头像上传流程

**前端流程**：
1. 用户选择图片文件
2. 验证文件大小（<2MB）和类型
3. 创建FormData对象
4. 发送PUT请求到 `/api/account/profile/`

```javascript
const handleAvatarChange = async (event) => {
  const file = event.target.files[0]
  const formData = new FormData()
  formData.append('avatar', file)
  
  await accountAPI.updateProfile(formData)
  await store.dispatch('checkLogin')  // 刷新用户信息
}
```

**后端流程**：
```python
# account/views.py - UserProfileView
parser_classes = (JSONParser, MultiPartParser, FormParser)

def put(self, request):
    user = User.objects.get(email=request.session['user_email'])
    serializer = UserUpdateSerializer(user, data=request.data, partial=True)
    if serializer.is_valid():
        serializer.save()  # Django自动保存文件到media/avatars/
        return Response({'message': '更新成功'})
```

### 7.4 排行榜实现

**前端流程**：
1. 用户选择维度（总分/语义/时序/运动/真实性）
2. 发送GET请求 `/api/model/ranking/?dimension=xxx`
3. 接收排序后的模型列表并渲染

**后端流程**：
```python
# model_eval/views.py - RankingView
def get(self, request):
    dimension = request.query_params.get('dimension', 'total')
    
    # 根据维度排序
    order_field = {
        'total': '-total_score',
        'semantic': '-semantic_score',
        'temporal': '-temporal_score',
        'motion': '-motion_score',
        'reality': '-reality_score'
    }.get(dimension, '-total_score')
    
    models = Model.objects.order_by(order_field)
    return Response(RankingSerializer(models, many=True).data)
```

### 7.5 评测任务提交流程

**前端流程**：
1. 用户上传模型输出视频和配置
2. 填写模型信息
3. 提交表单到 `/api/eval/submit/`
4. 获取task_id并跳转到状态查询页面
5. 轮询查询任务状态

**后端流程**：
```python
# eval_test/views.py - EvalSubmitView
def post(self, request):
    # 验证数据
    serializer = EvalRequestSerializer(data=request.data)
    if serializer.is_valid():
        # 生成任务ID
        task_id = str(uuid.uuid4())
        
        # 初始化任务
        eval_tasks[task_id] = {
            'status': 'pending',
            'progress': 0,
            'data': serializer.validated_data,
            'user_email': request.session['user_email']
        }
        
        # ⭐ 这里调用打分算法
        # start_evaluation_task(task_id, serializer.validated_data)
        
        return Response({'task_id': task_id})
```

---

## 8. 打分算法集成指南 ⭐⭐⭐

### 8.1 集成位置

**主要文件**：`backend/eval_test/views.py`

**集成点1**：任务提交时启动异步评测
```python
# eval_test/views.py - EvalSubmitView.post()
# 第57行附近

def post(self, request):
    # ... 前面的验证代码 ...
    
    task_id = str(uuid.uuid4())
    eval_tasks[task_id] = {...}
    
    # ⭐⭐⭐ 在这里调用你的打分算法 ⭐⭐⭐
    # 方式1：同步调用（会阻塞请求，不推荐）
    # result = your_scoring_algorithm(data)
    
    # 方式2：异步任务（推荐）
    from .tasks import start_evaluation_task
    start_evaluation_task.delay(task_id, serializer.validated_data)
    
    return Response({'task_id': task_id})
```

### 8.2 推荐方案：使用Celery异步任务

#### 步骤1：安装Celery
```bash
pip install celery redis
```

#### 步骤2：创建tasks.py
```python
# backend/eval_test/tasks.py
from celery import shared_task
from .views import eval_tasks
from datetime import datetime

@shared_task
def start_evaluation_task(task_id, eval_data):
    """
    异步评测任务
    
    参数：
        task_id: 任务ID
        eval_data: 包含以下字段的字典
            - model_name: 模型名称
            - description: 模型描述
            - publisher: 发布者
            - parameters: 参数规模
            - is_open_source: 是否开源
            - release_date: 发布日期
            - official_website: 官网
            - video_files: 上传的视频文件列表
            - api_key: API密钥（如果需要）
    """
    try:
        # 更新状态为处理中
        eval_tasks[task_id]['status'] = 'processing'
        eval_tasks[task_id]['progress'] = 10
        eval_tasks[task_id]['message'] = '正在分析视频...'
        
        # ⭐⭐⭐ 调用你的打分算法 ⭐⭐⭐
        # 示例：
        from your_algorithm import VideoScorer
        
        scorer = VideoScorer()
        
        # 假设你的算法需要视频路径
        video_paths = [f.path for f in eval_data.get('video_files', [])]
        
        # 逐个维度计算（可以更新进度）
        eval_tasks[task_id]['progress'] = 25
        eval_tasks[task_id]['message'] = '正在计算语义一致性...'
        semantic_score = scorer.calculate_semantic(video_paths)
        
        eval_tasks[task_id]['progress'] = 50
        eval_tasks[task_id]['message'] = '正在计算时序一致性...'
        temporal_score = scorer.calculate_temporal(video_paths)
        
        eval_tasks[task_id]['progress'] = 75
        eval_tasks[task_id]['message'] = '正在计算运动属性...'
        motion_score = scorer.calculate_motion(video_paths)
        
        eval_tasks[task_id]['progress'] = 90
        eval_tasks[task_id]['message'] = '正在计算真实性...'
        reality_score = scorer.calculate_reality(video_paths)
        
        # 计算总分
        total_score = (semantic_score + temporal_score + 
                      motion_score + reality_score) / 4
        
        # 保存到数据库
        from model_eval.models import Model
        from account.models import User
        
        user = User.objects.get(email=eval_tasks[task_id]['user_email'])
        
        model = Model.objects.create(
            name=eval_data['model_name'],
            description=eval_data['description'],
            publisher=eval_data['publisher'],
            parameters=eval_data['parameters'],
            is_open_source=eval_data['is_open_source'],
            release_date=eval_data['release_date'],
            official_website=eval_data['official_website'],
            semantic_score=semantic_score,
            temporal_score=temporal_score,
            motion_score=motion_score,
            reality_score=reality_score,
            total_score=total_score,
            tester_type='user',
            tester_name=user.username
        )
        
        # 更新任务状态为完成
        eval_tasks[task_id]['status'] = 'completed'
        eval_tasks[task_id]['progress'] = 100
        eval_tasks[task_id]['message'] = '评测完成'
        eval_tasks[task_id]['result'] = {
            'model_id': model.id,
            'semantic_score': semantic_score,
            'temporal_score': temporal_score,
            'motion_score': motion_score,
            'reality_score': reality_score,
            'total_score': total_score
        }
        
    except Exception as e:
        # 处理错误
        eval_tasks[task_id]['status'] = 'failed'
        eval_tasks[task_id]['message'] = f'评测失败：{str(e)}'
        raise
```

#### 步骤3：配置Celery
```python
# backend/backend/celery.py
from celery import Celery
import os

os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'backend.settings')

app = Celery('backend')
app.config_from_object('django.conf:settings', namespace='CELERY')
app.autodiscover_tasks()

# backend/backend/settings.py
CELERY_BROKER_URL = 'redis://localhost:6379/0'
CELERY_RESULT_BACKEND = 'redis://localhost:6379/0'
```

#### 步骤4：启动Celery Worker
```bash
# 新开一个终端
cd backend
celery -A backend worker -l info
```

### 8.3 简化方案：同步调用（开发测试用）

如果暂时不想使用Celery，可以直接在视图中同步调用：

```python
# backend/eval_test/scoring.py
class VideoScorer:
    """视频评分器"""
    
    def score_videos(self, video_paths, ref_prompts):
        """
        对视频进行四维度评分
        
        参数：
            video_paths: 视频文件路径列表
            ref_prompts: 参考提示词列表
        
        返回：
            {
                'semantic_score': float,
                'temporal_score': float,
                'motion_score': float,
                'reality_score': float
            }
        """
        # ⭐⭐⭐ 在这里实现你的算法 ⭐⭐⭐
        
        # 示例实现（需要替换为实际算法）
        semantic_score = self._calculate_semantic(video_paths, ref_prompts)
        temporal_score = self._calculate_temporal(video_paths)
        motion_score = self._calculate_motion(video_paths)
        reality_score = self._calculate_reality(video_paths)
        
        return {
            'semantic_score': semantic_score,
            'temporal_score': temporal_score,
            'motion_score': motion_score,
            'reality_score': reality_score
        }
    
    def _calculate_semantic(self, videos, prompts):
        """计算语义一致性"""
        # TODO: 实现你的算法
        # 例如：使用CLIP模型计算视频帧与文本的相似度
        return 85.5
    
    def _calculate_temporal(self, videos):
        """计算时序一致性"""
        # TODO: 实现你的算法
        # 例如：分析帧间光流一致性
        return 82.3
    
    def _calculate_motion(self, videos):
        """计算运动属性"""
        # TODO: 实现你的算法
        # 例如：分析运动轨迹的合理性
        return 88.7
    
    def _calculate_reality(self, videos):
        """计算真实性"""
        # TODO: 实现你的算法
        # 例如：使用预训练模型判断物理规律符合度
        return 90.2
```

```python
# backend/eval_test/views.py
from .scoring import VideoScorer

class EvalSubmitView(APIView):
    def post(self, request):
        # ... 验证代码 ...
        
        task_id = str(uuid.uuid4())
        eval_tasks[task_id] = {...}
        
        # 同步调用打分算法
        scorer = VideoScorer()
        scores = scorer.score_videos(
            video_paths=video_files,
            ref_prompts=prompts
        )
        
        # 直接保存结果
        model = Model.objects.create(
            name=data['model_name'],
            semantic_score=scores['semantic_score'],
            temporal_score=scores['temporal_score'],
            motion_score=scores['motion_score'],
            reality_score=scores['reality_score'],
            # ... 其他字段
        )
        
        eval_tasks[task_id]['status'] = 'completed'
        eval_tasks[task_id]['result'] = scores
        
        return Response({'task_id': task_id})
```

### 8.4 算法接口规范

你的打分算法需要实现以下接口：

```python
def calculate_scores(video_data, reference_data):
    """
    输入参数：
        video_data: {
            'paths': [视频文件路径列表],
            'format': 视频格式,
            'resolution': 分辨率,
            'fps': 帧率
        }
        reference_data: {
            'prompts': [文本提示词列表],
            'ref_videos': [参考视频路径列表],  # 可选
            'metadata': {...}  # 其他元数据
        }
    
    返回值：
        {
            'semantic_score': 0-100,      # 语义一致性
            'temporal_score': 0-100,      # 时序一致性
            'motion_score': 0-100,        # 运动属性
            'reality_score': 0-100,       # 真实性
            'details': {                  # 可选的详细信息
                'frame_scores': [...],
                'analysis_log': '...'
            }
        }
    """
    pass
```

---

## 9. 从零开始构建步骤

### 9.1 环境准备

```bash
# 1. 安装Python 3.8+
# 2. 安装Node.js 14+
# 3. 安装MySQL 8.0

# 4. 创建项目目录
mkdir shixun
cd shixun

# 5. 创建Python虚拟环境
python -m venv venv
venv\Scripts\activate  # Windows
source venv/bin/activate  # Linux/Mac

# 6. 安装后端依赖
pip install django djangorestframework django-cors-headers mysqlclient Pillow

# 7. 初始化前端项目
vue create frontend
cd frontend
npm install vue-router vuex axios echarts vue-echarts tailwindcss postcss autoprefixer
```

### 9.2 后端搭建

```bash
# 1. 创建Django项目
cd ..
django-admin startproject backend
cd backend

# 2. 创建应用
python manage.py startapp account
python manage.py startapp ref_data
python manage.py startapp model_eval
python manage.py startapp eval_test

# 3. 配置settings.py
# - 添加应用到INSTALLED_APPS
# - 配置数据库连接
# - 配置CORS
# - 配置MEDIA

# 4. 创建数据库
mysql -u root -p
CREATE DATABASE ref4d CHARACTER SET utf8mb4;

# 5. 创建模型（参考第4章）
# 6. 生成迁移文件
python manage.py makemigrations

# 7. 执行迁移
python manage.py migrate

# 8. 创建超级用户
python manage.py createsuperuser
```

### 9.3 前端搭建

```bash
# 1. 配置Tailwind CSS
cd frontend
npx tailwindcss init -p

# 2. 创建目录结构
mkdir src/api
mkdir src/views
mkdir src/components

# 3. 配置vue.config.js（代理）
# 4. 配置router
# 5. 配置vuex
# 6. 封装API
# 7. 创建页面组件
```

### 9.4 开发流程

```bash
# 1. 启动后端
cd backend
python manage.py runserver 8000

# 2. 启动前端（新终端）
cd frontend
npm run serve

# 3. 访问
# 前端: http://localhost:8080
# 后端: http://localhost:8000/admin
```

### 9.5 部署准备

```bash
# 1. 前端构建
cd frontend
npm run build

# 2. 收集静态文件
cd backend
python manage.py collectstatic

# 3. 配置生产环境settings
# - DEBUG = False
# - ALLOWED_HOSTS
# - 数据库连接
# - 静态文件服务

# 4. 使用gunicorn + nginx部署
```

---

## 附录A：常见问题

### Q1: CSRF Token错误
**解决**：确保前端设置了`withCredentials: true`并正确发送CSRF Token

### Q2: 头像上传失败
**解决**：检查FormData是否正确设置，Content-Type不要手动设置

### Q3: 跨域问题
**解决**：配置CORS和代理，确保credentials设置正确

---

## 附录B：API接口文档

详细API文档请参考自动生成的Swagger文档（如需要可以集成drf-yasg）

---

**文档版本**：v1.0  
**最后更新**：2025-11-30  
**作者**：Ref4D Team
