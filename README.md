# Ref4D 视频生成模型评测平台

## 项目简介

Ref4D是一个专业的视频生成模型评测平台，用于展示主流视频生成模型的评测结果，并提供新模型API评测接口。

## 技术栈

### 后端
- Django 5.2.8
- Django REST Framework 3.14.0
- MySQL 8.0+
- Python 3.8+

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

---

## 📋 开发者完整配置指南

### 1️⃣ 环境要求

在开始之前，请确保你的开发环境已安装：

- **Python**: 3.8 或更高版本
- **Node.js**: 14.x 或更高版本
- **npm**: 6.x 或更高版本
- **MySQL**: 8.0 或更高版本
- **Git**: 用于克隆代码仓库

---

### 2️⃣ 数据库配置

#### 步骤 1：安装并启动 MySQL

确保 MySQL 服务正在运行。

**Windows 用户：**
```bash
# 检查 MySQL 服务状态
net start MySQL80

# 如果未启动，执行启动命令
net start MySQL80
```

#### 步骤 2：创建数据库

打开 MySQL 命令行客户端或使用 MySQL Workbench，执行以下 SQL 命令：

```sql
CREATE DATABASE ref4d CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci;
```

#### 步骤 3：配置数据库连接

打开文件 `backend/backend/settings.py`，找到 `DATABASES` 配置项（第 92-101 行）：

```python
DATABASES = {
    'default': {
        'ENGINE': 'django.db.backends.mysql',
        'NAME': 'ref4d',              # 数据库名称
        'USER': 'root',                # 你的 MySQL 用户名
        'PASSWORD': '8892297.qh',      # 你的 MySQL 密码（请修改为你自己的）
        'HOST': 'localhost',           # 数据库主机地址
        'PORT': '3306',                # MySQL 端口号
    }
}
```

**⚠️ 重要：请将 `PASSWORD` 修改为你自己的 MySQL 密码！**

#### 步骤 4：执行数据库迁移

在项目根目录下执行：

```bash
# 激活虚拟环境（如果已创建）
venv\Scripts\activate

# 执行迁移命令
python backend\manage.py makemigrations
python backend\manage.py migrate
```

---

### 3️⃣ 邮箱服务器配置

本项目使用邮箱发送用户激活邮件和通知。

#### 步骤 1：获取邮箱 SMTP 授权码

以 QQ 邮箱为例（推荐使用）：

1. 登录 QQ 邮箱网页版
2. 进入 **设置** -> **账户**
3. 找到 **POP3/IMAP/SMTP/Exchange/CardDAV/CalDAV服务**
4. 开启 **POP3/SMTP服务** 或 **IMAP/SMTP服务**
5. 按照提示发送短信，获取 **16位授权码**（这不是你的 QQ 密码！）

#### 步骤 2：创建 `.env` 配置文件

在项目根目录（与 `backend` 和 `frontend` 文件夹同级）创建 `.env` 文件：

```bash
# .env 文件内容
EMAIL_HOST_USER=你的邮箱地址@qq.com
EMAIL_HOST_PASSWORD=你的16位授权码
```

**示例：**
```
EMAIL_HOST_USER=2377355798@qq.com
EMAIL_HOST_PASSWORD=abcdefghijklmnop
```

**⚠️ 注意事项：**
- 不要添加引号
- 授权码是16位字母组合，不是你的 QQ 密码
- `.env` 文件已在 `.gitignore` 中，不会被上传到 Git

#### 步骤 3：验证邮箱配置

邮箱配置位于 `backend/backend/settings.py` 的第 176-184 行：

```python
EMAIL_BACKEND = 'django.core.mail.backends.smtp.EmailBackend'
EMAIL_HOST = 'smtp.qq.com'              # QQ 邮箱 SMTP 服务器
EMAIL_PORT = 587                         # 端口号（TLS）
EMAIL_USE_TLS = True                     # 使用 TLS 加密
EMAIL_HOST_USER = config('EMAIL_HOST_USER', default='2377355798@qq.com')
EMAIL_HOST_PASSWORD = config('EMAIL_HOST_PASSWORD', default='')
DEFAULT_FROM_EMAIL = f'Ref4D <{EMAIL_HOST_USER}>'
```

**如果使用其他邮箱服务商：**

| 邮箱服务商 | SMTP 服务器 | 端口 |
|---------|-----------|-----|
| QQ邮箱   | smtp.qq.com | 587 |
| 163邮箱  | smtp.163.com | 465 |
| Gmail   | smtp.gmail.com | 587 |
| Outlook | smtp.office365.com | 587 |

修改 `EMAIL_HOST` 和 `EMAIL_PORT` 即可。

---

### 4️⃣ SECRET_KEY 配置（可选，生产环境必须）

#### 当前配置

`backend/backend/settings.py` 第 25 行：

```python
SECRET_KEY = 'django-insecure-v_9*0_tb5)0ngm1vp+_xq-kz%vqc&ya18-=^lr@6zsbp^)m@@0'
```

#### 生产环境配置

**⚠️ 在生产环境中，必须更换为自己生成的 SECRET_KEY！**

生成新的 SECRET_KEY：

```python
# 在 Python 交互式环境中执行
from django.core.management.utils import get_random_secret_key
print(get_random_secret_key())
```

将生成的值替换到 `settings.py` 中的 `SECRET_KEY`。

---

### 5️⃣ 局域网/多设备访问配置

如果需要在局域网内的其他设备（如手机、其他电脑）访问本项目：

#### 步骤 1：配置防火墙（Windows）

**方法 1：使用提供的脚本（推荐）**

以**管理员身份**运行项目根目录下的：
```
一键配置防火墙-管理员运行.bat
```

**方法 2：手动配置**

以管理员身份打开 PowerShell，执行以下命令：

```powershell
# 允许 8000 端口（后端）
netsh advfirewall firewall add rule name="Django Backend 8000" dir=in action=allow protocol=TCP localport=8000

# 允许 8080 端口（前端）
netsh advfirewall firewall add rule name="Vue Frontend 8080" dir=in action=allow protocol=TCP localport=8080
```

#### 步骤 2：查看本机 IP 地址

```bash
# Windows
ipconfig

# 找到 "IPv4 地址"，例如：192.168.1.100
```

#### 步骤 3：验证 ALLOWED_HOSTS 配置

`backend/backend/settings.py` 第 30-36 行已配置为允许局域网访问：

```python
ALLOWED_HOSTS = [
    'localhost',
    '127.0.0.1',
    '192.168.*.*',  # 允许所有192.168网段访问
    '10.0.*.*',     # 允许所有10.0网段访问
    '*',            # 开发环境允许所有（生产环境请改为具体IP）
]
```

#### 步骤 4：验证 CORS 和 CSRF 配置

已配置为支持局域网跨域访问（第 149-158 行）：

```python
# CORS settings
CORS_ALLOW_ALL_ORIGINS = True
CORS_ALLOW_CREDENTIALS = True

# CSRF settings
CSRF_TRUSTED_ORIGINS = [
    'http://localhost:8080',
    'http://127.0.0.1:8080',
]
```

**注意：** 项目包含动态中间件 `DynamicCSRFTrustedOriginsMiddleware`，会自动添加局域网 IP 到信任列表。

#### 步骤 5：使用局域网访问

启动后端和前端后，其他设备可通过以下地址访问：

- **前端**: `http://你的IP地址:8080`
- **后端API**: `http://你的IP地址:8000`

例如：`http://192.168.1.100:8080`

---

### 6️⃣ 后端依赖安装

#### 步骤 1：创建虚拟环境（推荐）

```bash
# 在项目根目录执行
python -m venv venv
```

#### 步骤 2：激活虚拟环境

```bash
# Windows
venv\Scripts\activate

# 激活后，命令行前面会显示 (venv)
```

#### 步骤 3：安装依赖

项目已包含 `backend/requirements.txt` 文件，包含以下依赖：

```
Django==5.2.8
djangorestframework==3.14.0
django-cors-headers==4.3.1
PyMySQL==1.1.0
cryptography==41.0.7
Pillow==10.1.0
python-decouple==3.8
```

执行安装命令：

```bash
pip install -r backend/requirements.txt
```

**常见问题：**

- **如果安装 PyMySQL 出现问题**：可尝试安装 `mysqlclient` 作为替代：
  ```bash
  pip install mysqlclient
  ```

- **如果遇到编译错误**（特别是 Pillow）：
  - Windows 用户请确保安装了 [Microsoft C++ Build Tools](https://visualstudio.microsoft.com/visual-cpp-build-tools/)
  - 或使用预编译的二进制包：
    ```bash
    pip install --only-binary :all: Pillow
    ```

---

### 7️⃣ 前端依赖安装

#### 步骤 1：进入前端目录

```bash
cd frontend
```

#### 步骤 2：安装依赖

```bash
npm install
```

**如果 npm 速度较慢，可使用国内镜像：**

```bash
# 使用淘宝镜像
npm install --registry=https://registry.npmmirror.com

# 或者永久设置
npm config set registry https://registry.npmmirror.com
```

#### 步骤 3：验证安装

```bash
# 查看已安装的包
npm list --depth=0
```

应该看到以下主要依赖：
- vue@3.x
- vue-router@4.x
- vuex@4.x
- axios
- echarts
- vue-echarts
- tailwindcss

---

### 8️⃣ 启动后端服务器

#### 方法 1：使用提供的 .bat 脚本（推荐）

直接双击运行项目根目录下的：
```
启动后端.bat
```

脚本会自动：
- 获取并显示本机 IP 地址
- 激活虚拟环境（如果存在）
- 启动 Django 开发服务器
- 显示所有可用的访问地址

#### 方法 2：手动启动

```bash
# 1. 激活虚拟环境
venv\Scripts\activate

# 2. 进入 backend 目录
cd backend

# 3. 启动服务器（允许局域网访问）
python manage.py runserver 0.0.0.0:8000
```

**成功启动的标志：**
```
Starting development server at http://0.0.0.0:8000/
Quit the server with CTRL-BREAK.
```

**访问地址：**
- 本机访问: `http://localhost:8000`
- 局域网访问: `http://你的IP:8000`
- 管理后台: `http://localhost:8000/admin`

---

### 9️⃣ 启动前端开发服务器

#### 方法 1：使用提供的 .bat 脚本（推荐）

直接双击运行项目根目录下的：
```
启动前端.bat
```

脚本会自动：
- 进入 frontend 目录
- 启动 Vue 开发服务器
- 显示所有可用的访问地址

#### 方法 2：手动启动

```bash
# 1. 进入前端目录
cd frontend

# 2. 启动开发服务器
npm run serve
```

**成功启动的标志：**
```
App running at:
- Local:   http://localhost:8080/
- Network: http://你的IP:8080/
```

**访问地址：**
- 本机访问: `http://localhost:8080`
- 局域网访问: `http://你的IP:8080`

---

## 🎯 完整启动流程总结

### 首次部署

1. ✅ 安装 Python、Node.js、MySQL
2. ✅ 创建数据库 `ref4d`
3. ✅ 修改 `settings.py` 中的数据库密码
4. ✅ 创建 `.env` 文件，配置邮箱
5. ✅ 配置防火墙（局域网访问需要）
6. ✅ 安装后端依赖：`pip install -r backend/requirements.txt`
7. ✅ 执行数据库迁移：`python backend/manage.py migrate`
8. ✅ 创建超级用户：`python backend/manage.py createsuperuser`
9. ✅ 安装前端依赖：`cd frontend && npm install`
10. ✅ 启动后端：双击 `启动后端.bat`
11. ✅ 启动前端：双击 `启动前端.bat`
12. ✅ 访问 `http://localhost:8080`

### 日常开发

1. 双击 `启动后端.bat`
2. 双击 `启动前端.bat`
3. 开始开发！

---

## API 接口文档

### 账户模块 (`/api/account/`)

| 方法 | 端点 | 说明 |
|-----|------|------|
| POST | `/register/` | 用户注册 |
| POST | `/login/` | 用户登录 |
| POST | `/logout/` | 用户登出 |
| GET | `/check-login/` | 检查登录状态 |
| GET | `/profile/` | 获取用户信息 |
| PUT | `/profile/` | 更新用户信息 |

### 数据模块 (`/api/data/`)

| 方法 | 端点 | 说明 |
|-----|------|------|
| GET | `/refdata/` | 获取参考数据集 |
| GET | `/refdata/themes/` | 获取主题统计 |
| GET | `/refdata/by_theme/?theme=<theme>` | 按主题筛选 |
| GET | `/gendata/` | 获取生成数据 |
| GET | `/gendata/by_model/?model_name=<name>` | 按模型筛选 |

### 模型模块 (`/api/model/`)

| 方法 | 端点 | 说明 |
|-----|------|------|
| GET | `/models/` | 获取模型列表 |
| GET | `/models/<id>/` | 获取模型详情 |
| GET | `/models/ranking/?dimension=<dimension>` | 获取排行榜 |
| GET | `/models/<id>/scores/` | 获取模型评分 |

**排行榜维度参数：**
- `total` - 总分排行
- `semantic` - 语义一致性
- `temporal` - 时序一致性
- `motion` - 运动属性
- `reality` - 真实性

### 评测模块 (`/api/eval/`)

| 方法 | 端点 | 说明 |
|-----|------|------|
| POST | `/submit/` | 提交评测任务 |
| GET | `/status/<task_id>/` | 查询评测状态 |
| POST | `/mock-complete/<task_id>/` | 模拟评测完成（仅测试） |

---

## 数据库表结构

### User 表（用户）

| 字段 | 类型 | 说明 |
|------|------|------|
| email | VARCHAR (主键) | 邮箱地址 |
| username | VARCHAR | 用户名 |
| password | VARCHAR | 密码（加密） |
| avatar | VARCHAR | 头像路径 |
| created_at | DATETIME | 创建时间 |
| updated_at | DATETIME | 更新时间 |

### RefData 表（参考数据集）

| 字段 | 类型 | 说明 |
|------|------|------|
| video_id | VARCHAR (主键) | 视频ID |
| theme | VARCHAR | 主题分类（9个） |
| shot_type | VARCHAR | 镜头类型（single/multi） |
| prompt | TEXT | 提示词 |
| video_file | VARCHAR | 视频文件路径 |
| created_at | DATETIME | 创建时间 |

**9个主题分类：**
1. animals_and_ecology - 动物与生态
2. architecture - 建筑
3. commercial_marketing - 商业营销
4. food - 食物
5. industrial_activity - 工业活动
6. landscape - 风景
7. people_daily - 人物日常
8. sports_competition - 体育竞技
9. transportation - 交通

### GenData 表（生成数据）

| 字段 | 类型 | 说明 |
|------|------|------|
| video_id | VARCHAR | 视频ID |
| theme | VARCHAR | 主题分类 |
| shot_type | VARCHAR | 镜头类型 |
| model_name | VARCHAR | 模型名称 |
| prompt | TEXT | 提示词 |
| video_file | VARCHAR | 视频文件路径 |
| created_at | DATETIME | 创建时间 |

### Model 表（模型信息）

| 字段 | 类型 | 说明 |
|------|------|------|
| name | VARCHAR (唯一) | 模型名称 |
| description | TEXT | 模型简介 |
| publisher | VARCHAR | 发布者 |
| parameters | VARCHAR | 参数规模 |
| is_open_source | BOOLEAN | 是否开源 |
| release_date | DATE | 发布时间 |
| official_website | VARCHAR | 官网链接 |
| semantic_score | FLOAT | 语义一致性得分 |
| temporal_score | FLOAT | 时序一致性得分 |
| motion_score | FLOAT | 运动属性得分 |
| reality_score | FLOAT | 真实性得分 |
| total_score | FLOAT | 总分 |
| tester_type | VARCHAR | 测试者类型（admin/user） |
| tester_name | VARCHAR | 测试者姓名 |
| created_at | DATETIME | 创建时间 |
| updated_at | DATETIME | 更新时间 |

---

## 评测维度说明

本平台从4个维度对视频生成模型进行评估：

### 1. 基础语义一致性 (Semantic Consistency)
- **定义**: 生成视频与 prompt 描述的语义匹配程度
- **评估内容**: 物体、动作、场景、颜色、数量等是否与提示词一致
- **权重**: 25%

### 2. 时序一致性 (Temporal Consistency)
- **定义**: 视频帧与帧之间的连贯性和流畅度
- **评估内容**: 是否存在突变、闪烁、抖动等问题
- **权重**: 25%

### 3. 运动属性 (Motion Quality)
- **定义**: 物体运动的真实性和合理性
- **评估内容**: 运动轨迹、速度、加速度是否符合物理规律
- **权重**: 25%

### 4. 世界知识真实性 (Reality)
- **定义**: 生成内容是否符合现实世界的基本规律
- **评估内容**: 物理规律、常识、因果关系等
- **权重**: 25%

**总分计算：** Total Score = (Semantic + Temporal + Motion + Reality) / 4

---

## 管理后台

### 访问地址

```
http://localhost:8000/admin
```

### 创建超级管理员

```bash
# 激活虚拟环境
venv\Scripts\activate

# 进入后端目录
cd backend

# 创建超级用户
python manage.py createsuperuser

# 按提示输入用户名、邮箱和密码
```

### 管理功能

通过管理后台可以：

- ✅ 管理用户账户
- ✅ 查看和编辑模型数据
- ✅ 管理参考数据集
- ✅ 管理生成数据
- ✅ 查看评测任务
- ✅ 手动修改模型评分

---

## 开发说明

### 评测算法对接

**位置**: `backend/eval_test/views.py` 中的 `EvalSubmitView`

**当前状态**: 使用模拟接口进行测试

**生产环境需要**:
1. 对接实际的视频生成 API
2. 实现真实的打分算法
3. 处理异步评测任务队列
4. 实现视频下载和存储

**测试接口**: `/api/eval/mock-complete/<task_id>/`
- 仅用于开发测试
- 生产环境需移除

### 目录结构

```
shixun/
├── backend/                 # Django 后端
│   ├── account/            # 用户账户模块
│   ├── model_eval/         # 模型评测模块
│   ├── ref_data/           # 参考数据模块
│   ├── eval_test/          # 评测任务模块
│   ├── task_list/          # 任务列表模块
│   ├── media/              # 媒体文件存储
│   ├── backend/            # Django 项目配置
│   │   ├── settings.py     # 配置文件
│   │   └── urls.py         # 路由配置
│   ├── manage.py           # Django 管理脚本
│   └── requirements.txt    # Python 依赖
│
├── frontend/                # Vue 前端
│   ├── src/
│   │   ├── views/          # 页面组件
│   │   ├── components/     # 公共组件
│   │   ├── router/         # 路由配置
│   │   ├── store/          # Vuex 状态管理
│   │   ├── api/            # API 接口
│   │   └── App.vue         # 根组件
│   ├── package.json        # npm 依赖
│   └── vue.config.js       # Vue 配置
│
├── .env                     # 环境变量（需创建）
├── .gitignore              # Git 忽略文件
├── 启动后端.bat            # 后端启动脚本
├── 启动前端.bat            # 前端启动脚本
└── README.md               # 项目文档（本文件）
```

### 跨域配置

**后端配置** (`backend/backend/settings.py`):
```python
# CORS 配置
CORS_ALLOW_ALL_ORIGINS = True
CORS_ALLOW_CREDENTIALS = True

# CSRF 配置
CSRF_TRUSTED_ORIGINS = [
    'http://localhost:8080',
    'http://127.0.0.1:8080',
]
```

**前端配置** (`frontend/vue.config.js`):
```javascript
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

## ⚠️ 注意事项

### 开发环境

1. ✅ 确保 MySQL 服务正在运行
2. ✅ 数据库密码已在 `settings.py` 中正确配置
3. ✅ `.env` 文件已创建并配置邮箱信息
4. ✅ 防火墙已配置（局域网访问需要）
5. ✅ 媒体文件存储在 `backend/media/` 目录
6. ✅ 静态文件存储在 `backend/static/` 目录

### 生产环境

1. ⚠️ **必须更换 SECRET_KEY**
2. ⚠️ **设置 DEBUG = False**
3. ⚠️ **配置具体的 ALLOWED_HOSTS**
4. ⚠️ **使用 HTTPS 连接**
5. ⚠️ **配置真实的评测算法**
6. ⚠️ **使用生产级数据库**
7. ⚠️ **配置 Nginx 反向代理**
8. ⚠️ **使用 Gunicorn/uWSGI**

---

## 🔒 安全配置

### 敏感信息保护

- ✅ 邮箱密码已从代码中移除
- ✅ 使用 `.env` 文件存储敏感配置
- ✅ `.gitignore` 已配置，防止敏感文件上传
- ✅ 使用 `python-decouple` 安全读取环境变量

### 配置文件说明

| 文件 | 说明 | Git 状态 |
|------|------|---------|
| `.env` | 存储真实密码和密钥 | ❌ 不上传 |
| `.env.example` | 配置模板和示例 | ✅ 上传 |
| `requirements.txt` | Python 依赖列表 | ✅ 上传 |
| `package.json` | Node.js 依赖列表 | ✅ 上传 |

---

## 🚀 常见问题

### Q1: 数据库连接失败？
**A**: 检查以下项：
1. MySQL 服务是否启动
2. 数据库名称、用户名、密码是否正确
3. 数据库是否已创建

### Q2: 邮件发送失败？
**A**: 检查以下项：
1. `.env` 文件是否创建
2. 邮箱授权码是否正确（不是密码！）
3. 邮箱服务商 SMTP 设置是否正确

### Q3: 局域网无法访问？
**A**: 检查以下项：
1. 防火墙是否已配置
2. 后端启动时是否使用 `0.0.0.0:8000`
3. 前端和后端是否都在运行
4. IP 地址是否正确

### Q4: 安装依赖时报错？
**A**: 尝试以下方案：
1. 升级 pip：`python -m pip install --upgrade pip`
2. 使用国内镜像：`pip install -i https://pypi.tuna.tsinghua.edu.cn/simple`
3. 单独安装问题包

### Q5: 前端启动后页面空白？
**A**: 检查以下项：
1. 浏览器控制台是否有错误
2. 后端 API 是否正常运行
3. 前端配置文件是否正确

---

## 📞 技术支持

如遇到其他问题，请检查：
1. 项目日志文件
2. 浏览器开发者工具控制台
3. Django 后端控制台输出

---

## 许可证

本项目仅用于教学和研究目的。
