# Account应用重构实施指南

## 📋 概述

本次重构实现了以下功能：
1. ✅ 使用邮箱注册并发送激活邮件
2. ✅ 添加is_active字段控制账户激活状态
3. ✅ 实现忘记密码功能（邮件验证码）
4. ✅ 完整的邮件模板和错误处理

## 🚀 后端实施步骤

### 步骤1：安装依赖包

```bash
cd backend
pip install -r requirements.txt
```

新增的包：
- `djoser==2.2.2` - 认证扩展（未来可选）
- `djangorestframework-simplejwt==5.3.1` - JWT支持（未来可选）
- `cryptography==41.0.7` - MySQL认证所需

### 步骤2：生成并执行数据库迁移

```bash
# 生成迁移文件
python manage.py makemigrations account

# 执行迁移
python manage.py migrate
```

**新增的数据库字段：**
- `is_active` - 布尔值，默认False
- `is_staff` - 布尔值，默认False  
- `is_superuser` - 布尔值，默认False
- `activation_token` - 字符串，激活令牌
- `activation_token_created` - 日期时间，令牌创建时间
- `reset_password_token` - 字符串，6位数字验证码
- `reset_password_token_created` - 日期时间，验证码创建时间

### 步骤3：更新现有用户数据

由于添加了`is_active`字段（默认False），需要将现有用户设置为激活状态：

```bash
# 方式1：通过Django shell
python manage.py shell

# 在shell中执行
from account.models import User
User.objects.all().update(is_active=True, is_staff=False, is_superuser=False)
exit()
```

```bash
# 方式2：通过MySQL直接更新
mysql -u root -p
use ref4d;
UPDATE user SET is_active = 1, is_staff = 0, is_superuser = 0;
exit
```

### 步骤4：验证邮件配置

邮件配置在 `backend/settings.py`：

```python
EMAIL_BACKEND = 'django.core.mail.backends.smtp.EmailBackend'
EMAIL_HOST = 'smtp.qq.com'
EMAIL_PORT = 587
EMAIL_USE_TLS = True
EMAIL_HOST_USER = '2377355798@qq.com'
EMAIL_HOST_PASSWORD = 'ygcqbeitbnnvechf'
DEFAULT_FROM_EMAIL = 'Ref4D <2377355798@qq.com>'
```

**测试邮件发送：**

```bash
python manage.py shell

from django.core.mail import send_mail
send_mail(
    'Test Email',
    'This is a test message.',
    'Ref4D <2377355798@qq.com>',
    ['your_email@example.com'],
    fail_silently=False,
)
```

### 步骤5：启动后端服务器

```bash
python manage.py runserver 8000
```

## 🎨 前端实施步骤

### 步骤1：更新API封装

文件：`frontend/src/api/account.js`

需要添加以下新API：

```javascript
export const accountAPI = {
  // ... 现有的API ...
  
  // 激活账户
  activateAccount(email, token) {
    return request.get(`/account/activate/${email}/${token}/`)
  },
  
  // 重新发送激活邮件
  resendActivationEmail(email) {
    return request.post('/account/resend-activation/', { email })
  },
  
  // 请求密码重置（发送验证码）
  requestPasswordReset(email) {
    return request.post('/account/request-password-reset/', { email })
  },
  
  // 验证重置验证码
  verifyResetCode(email, code) {
    return request.post('/account/verify-reset-code/', { email, code })
  },
  
  // 重置密码
  resetPassword(email, code, new_password) {
    return request.post('/account/reset-password/', { 
      email, 
      code, 
      new_password 
    })
  }
}
```

### 步骤2：创建激活页面

创建 `frontend/src/views/Activate.vue`

### 步骤3：修改注册页面

修改 `frontend/src/views/Register.vue`
- 将用户名改为邮箱输入
- 注册成功后显示邮箱验证提示
- 添加重新发送激活邮件按钮

### 步骤4：修改登录页面

修改 `frontend/src/views/Login.vue`
- 处理未激活账户的情况
- 添加"忘记密码"按钮
- 添加重新发送激活邮件提示

### 步骤5：创建忘记密码页面

创建 `frontend/src/views/ForgotPassword.vue`
包含：
1. 输入邮箱步骤
2. 输入验证码步骤
3. 设置新密码步骤

### 步骤6：更新路由配置

在 `frontend/src/router/index.js` 添加路由：

```javascript
{
  path: '/activate/:email/:token',
  name: 'Activate',
  component: () => import('@/views/Activate.vue')
},
{
  path: '/forgot-password',
  name: 'ForgotPassword',
  component: () => import('@/views/ForgotPassword.vue')
}
```

## 📝 API文档

### 用户注册

**POST** `/api/account/register/`

请求：
```json
{
  "email": "user@example.com",
  "username": "username",
  "password": "password123",
  "password_confirm": "password123"
}
```

响应（成功）：
```json
{
  "message": "注册成功！我们已向您的邮箱发送了激活链接...",
  "email": "user@example.com",
  "require_activation": true
}
```

### 激活账户

**GET** `/api/account/activate/<email>/<token>/`

响应（成功）：
```json
{
  "message": "账户激活成功！",
  "detail": "您现在可以登录了"
}
```

### 用户登录

**POST** `/api/account/login/`

请求：
```json
{
  "username": "username",
  "password": "password123"
}
```

响应（未激活）：
```json
{
  "error": "账户未激活",
  "detail": "请先前往注册邮箱查收激活邮件并激活账户",
  "require_activation": true,
  "email": "user@example.com"
}
```

### 请求密码重置

**POST** `/api/account/request-password-reset/`

请求：
```json
{
  "email": "user@example.com"
}
```

响应（成功）：
```json
{
  "message": "验证码已发送",
  "detail": "请查收邮件并使用验证码重置密码（15分钟内有效）",
  "email": "user@example.com"
}
```

### 验证重置码

**POST** `/api/account/verify-reset-code/`

请求：
```json
{
  "email": "user@example.com",
  "code": "123456"
}
```

响应（成功）：
```json
{
  "message": "验证码正确",
  "detail": "请设置新密码"
}
```

### 重置密码

**POST** `/api/account/reset-password/`

请求：
```json
{
  "email": "user@example.com",
  "code": "123456",
  "new_password": "newpassword123"
}
```

响应（成功）：
```json
{
  "message": "密码重置成功",
  "detail": "请使用新密码登录"
}
```

## 🧪 测试步骤

### 1. 测试注册流程
1. 访问注册页面
2. 输入邮箱、用户名、密码
3. 提交注册
4. 检查邮箱是否收到激活邮件
5. 点击激活链接
6. 验证是否显示激活成功

### 2. 测试登录流程
1. 使用未激活账户尝试登录
2. 验证是否提示未激活
3. 激活账户后重新登录
4. 验证是否成功登录

### 3. 测试密码重置流程
1. 点击"忘记密码"
2. 输入注册邮箱
3. 检查邮箱是否收到验证码
4. 输入验证码
5. 设置新密码
6. 使用新密码登录

## ⚠️ 注意事项

1. **邮件发送失败**
   - 检查QQ邮箱授权码是否正确
   - 确保网络可以访问smtp.qq.com
   - 查看Django日志获取详细错误信息

2. **激活链接过期**
   - 激活链接有效期24小时
   - 用户可以使用"重新发送激活邮件"功能

3. **验证码过期**
   - 密码重置验证码有效期15分钟
   - 过期后需要重新请求

4. **数据库迁移**
   - 务必先备份数据库
   - 执行迁移后更新现有用户的is_active状态

5. **生产环境配置**
   - 修改邮件服务器配置使用生产环境的SMTP
   - 激活链接应使用生产环境域名
   - 启用HTTPS确保安全

## 🔧 故障排除

### 问题1：邮件发送失败

**错误：** SMTPAuthenticationError

**解决方案：**
- 检查QQ邮箱是否开启SMTP服务
- 确认授权码是否正确（不是邮箱密码）
- 尝试更换邮件服务商（如163、Gmail）

### 问题2：激活链接无效

**原因：** 前端路由配置错误或token不匹配

**解决方案：**
- 检查路由配置是否正确
- 验证API请求的URL格式
- 查看后端日志确认token是否生成

### 问题3：现有用户无法登录

**原因：** is_active默认为False

**解决方案：**
执行SQL更新现有用户：
```sql
UPDATE user SET is_active = 1;
```

## 📊 数据库Schema变更

**变更前：**
```sql
CREATE TABLE user (
    email VARCHAR(255) PRIMARY KEY,
    username VARCHAR(150),
    password VARCHAR(255),
    avatar VARCHAR(100),
    created_at DATETIME,
    updated_at DATETIME
);
```

**变更后：**
```sql
CREATE TABLE user (
    email VARCHAR(255) PRIMARY KEY,
    username VARCHAR(150),
    password VARCHAR(255),
    avatar VARCHAR(100),
    is_active BOOLEAN DEFAULT 0,
    is_staff BOOLEAN DEFAULT 0,
    is_superuser BOOLEAN DEFAULT 0,
    activation_token VARCHAR(64),
    activation_token_created DATETIME,
    reset_password_token VARCHAR(6),
    reset_password_token_created DATETIME,
    created_at DATETIME,
    updated_at DATETIME
);
```

## 🎉 完成检查清单

- [ ] 安装所有依赖包
- [ ] 执行数据库迁移
- [ ] 更新现有用户is_active状态
- [ ] 测试邮件发送功能
- [ ] 更新前端API封装
- [ ] 创建/修改前端页面
- [ ] 更新路由配置
- [ ] 测试完整注册流程
- [ ] 测试完整登录流程
- [ ] 测试密码重置流程
- [ ] 部署到生产环境

---

**文档版本：** v1.0  
**更新日期：** 2025-12-03  
**作者：** Ref4D Team
