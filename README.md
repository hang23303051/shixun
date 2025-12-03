# Ref4D 视频生成模型评测平台

## 项目简介

Ref4D是一个专业的视频生成模型评测平台，用于展示主流视频生成模型的评测结果，并提供新模型API评测接口。

## 技术栈

### 后端
- Django 5.2.8
- Django REST Framework
- MySQL
- Python 3.x

### 前端
- Vue 3
- Vue Router 4
- Vuex 4
- Tailwind CSS
- ECharts (图表可视化)
- Axios

## 功能模块

1. **首页** - 平台介绍和数据统计
2. **数据集** - 展示600条测试数据集，覆盖9个主题
3. **模型** - 展示已评测的视频生成模型
4. **排行榜** - 总榜和4个维度的排行榜
5. **评测试用** - 用户提交新模型API进行评测
6. **个人中心** - 用户信息管理
7. **登录/注册** - 用户认证系统

## 快速开始

### 环境要求

- Python 3.8+
- Node.js 14+
- MySQL 8.0+

### ⚠️ 首次配置：设置邮箱密码（重要！）

**为了安全，邮箱密码已从代码中移除，需要配置 `.env` 文件：**

1. 在项目根目录创建 `.env` 文件
2. 添加以下内容（不要引号）：
   ```
   EMAIL_HOST_USER=your_email@qq.com
   EMAIL_HOST_PASSWORD=your_16_digit_code
   ```
3. 安装依赖：双击运行 **`安装依赖.bat`**
4. 启动测试

**详细说明：** 查看 `.env文件配置说明.md`

### 后端启动

1. 激活虚拟环境（已创建）
```bash
# Windows
venv\Scripts\activate
```

2. 安装依赖（已安装）
```bash
pip install django djangorestframework django-cors-headers pillow mysqlclient
```

3. 数据库已配置并迁移

4. 创建超级管理员
```bash
venv\Scripts\python backend\manage.py createsuperuser
```

5. 启动后端服务器
```bash
venv\Scripts\python backend\manage.py runserver 8000
```

后端API地址：http://localhost:8000

### 前端启动

1. 进入前端目录
```bash
cd frontend
```

2. 安装依赖（已安装）
```bash
npm install
```

3. 启动开发服务器
```bash
npm run serve
```

前端地址：http://localhost:8080

## API接口说明

### 账户模块 (`/api/account/`)
- POST `/register/` - 用户注册
- POST `/login/` - 用户登录
- POST `/logout/` - 用户登出
- GET `/check-login/` - 检查登录状态
- GET `/profile/` - 获取用户信息
- PUT `/profile/` - 更新用户信息

### 数据模块 (`/api/data/`)
- GET `/refdata/` - 获取参考数据集
- GET `/refdata/themes/` - 获取主题统计
- GET `/refdata/by_theme/?theme=<theme>` - 按主题筛选
- GET `/gendata/` - 获取生成数据
- GET `/gendata/by_model/?model_name=<name>` - 按模型筛选

### 模型模块 (`/api/model/`)
- GET `/models/` - 获取模型列表
- GET `/models/<id>/` - 获取模型详情
- GET `/models/ranking/?dimension=<dimension>` - 获取排行榜
- GET `/models/<id>/scores/` - 获取模型评分

### 评测模块 (`/api/eval/`)
- POST `/submit/` - 提交评测任务
- GET `/status/<task_id>/` - 查询评测状态
- POST `/mock-complete/<task_id>/` - 模拟评测完成（测试用）

## 数据库表结构

### User表
- email (主键)
- username
- password (加密)
- avatar
- created_at
- updated_at

### RefData表
- video_id (主键)
- theme (9个主题)
- shot_type (single/multi)
- prompt
- video_file
- created_at

### GenData表
- video_id
- theme
- shot_type
- model_name
- prompt
- video_file
- created_at

### Model表
- name (唯一)
- description
- publisher
- parameters
- is_open_source
- release_date
- official_website
- semantic_score (语义一致性)
- temporal_score (时序一致性)
- motion_score (运动属性)
- reality_score (真实性)
- total_score (总分)
- tester_type (admin/user)
- tester_name
- created_at
- updated_at

## 评测维度

1. **基础语义一致性** - 生成视频与prompt描述的语义匹配度
2. **时序一致性** - 视频帧间的连贯性和流畅度
3. **运动属性** - 物体运动的真实性和合理性
4. **世界知识真实性** - 内容符合现实世界规律

## 管理后台

访问 http://localhost:8000/admin 使用超级管理员账号登录，可以管理：
- 用户
- 模型数据
- 参考数据集
- 生成数据

## 开发说明

### 评测算法接口

在 `backend/eval_test/views.py` 的 `EvalSubmitView` 中有TODO标记，需要对接实际的打分算法。

当前使用模拟接口 `mock-complete` 用于测试，生产环境需要实现真实的评测算法调用。

### 跨域配置

- 后端：已配置 CORS，允许 http://localhost:8080
- 前端：vue.config.js 配置了代理转发到后端

## 注意事项

1. 确保MySQL服务正在运行
2. 数据库配置在 `backend/backend/settings.py`
3. SECRET_KEY在生产环境需要更换
4. 媒体文件存储在 `backend/media/` 目录
5. 静态文件存储在 `backend/static/` 目录

## 🔒 安全配置

**邮箱密码保护：**
- ✅ 敏感信息已从代码中移除
- ✅ 使用 `.env` 文件存储邮箱授权码
- ✅ `.gitignore` 已配置，防止 `.env` 上传
- ✅ 提供 `.env.example` 作为配置模板
- ✅ 使用 `python-decouple` 安全读取

**配置文件：**
- `.env` - 存储真实密码（不会上传）
- `.env.example` - 配置模板（会上传）
- `.env文件配置说明.md` - 详细文档
- `安装依赖.bat` - 自动安装依赖

## 🌐 局域网部署

**一键启动脚本：**
- `一键配置防火墙-管理员运行.bat` - 配置防火墙（首次）
- `启动后端.bat` - 启动后端（自动显示IP）
- `启动前端.bat` - 启动前端（自动显示IP）

**功能：**
- ✨ 自动获取本机IP地址
- ✨ 显示所有访问方式（本机/手机/其他电脑）
- ✨ 支持局域网内多设备访问

## 🔗 域名配置

**支持使用自定义域名访问项目：**

**方案1：局域网内使用域名**
- 运行：`配置域名-局域网.bat`（管理员）
- 适合：本地开发测试
- 时间：5分钟
- 费用：免费

**方案2：外网访问 + 真实域名**
- 使用内网穿透（frp/ngrok）
- 适合：演示给他人
- 需要：公网服务器或付费服务

**方案3：部署到云服务器**
- 推荐：阿里云 ECS
- 适合：生产环境
- 配置：Nginx + HTTPS

**详细教程：** 查看 `域名配置指南.md`

## 许可证

本项目仅用于教学和研究目的。
