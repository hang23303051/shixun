# 🚀 Account重构快速开始指南

## 立即执行的命令（5分钟完成）

### 1. 安装后端依赖
```bash
cd d:\code\github\shixun\backend
pip install cryptography==41.0.7
```

### 2. 生成并执行数据库迁移
```bash
# 生成迁移文件
python manage.py makemigrations account

# 执行迁移（会添加新字段到user表）
python manage.py migrate
```

### 3. ⚠️ 关键步骤：更新现有用户
**不执行此步骤，现有用户将无法登录！**

```bash
# 方式1：使用Django shell（推荐）
python manage.py shell
```

在shell中执行：
```python
from account.models import User
# 将所有现有用户设置为已激活
User.objects.all().update(is_active=True, is_staff=False, is_superuser=False)
# 查看更新结果
print(f"Updated {User.objects.filter(is_active=True).count()} users")
exit()
```

或者**方式2：直接用MySQL**
```sql
mysql -u root -p8892297.qh
use ref4d;
UPDATE user SET is_active = 1, is_staff = 0, is_superuser = 0;
SELECT email, username, is_active FROM user;
exit;
```

### 4. 测试邮件发送（可选但推荐）
```bash
python manage.py shell
```

```python
from django.core.mail import send_mail

# 发送测试邮件到你自己的邮箱
send_mail(
    'Ref4D测试邮件',
    '如果你收到这封邮件，说明邮件服务配置成功！',
    '2377355798@qq.com',
    ['your_email@example.com'],  # 改成你的邮箱
    fail_silently=False,
)

print("测试邮件已发送，请检查收件箱")
exit()
```

### 5. 启动服务器
```bash
# 启动后端
python manage.py runserver 8000

# 新开一个终端，启动前端
cd d:\code\github\shixun\frontend
npm run serve
```

---

## 📝 接下来需要修改的前端文件

### 文件1：修改注册页面
**位置：** `frontend/src/views/Register.vue`

**需要修改的内容：**
1. 将用户名改为邮箱字段
2. 注册成功后显示激活提示
3. 添加重新发送激活邮件按钮

**参考代码已提供在：** `ACCOUNT_REFACTOR_SUMMARY.md`

### 文件2：修改登录页面
**位置：** `frontend/src/views/Login.vue`

**需要修改的内容：**
1. 添加"忘记密码"链接
2. 处理未激活账户的提示
3. 显示重新发送激活邮件选项

**参考代码已提供在：** `ACCOUNT_REFACTOR_SUMMARY.md`

---

## ✅ 验证清单

完成上述步骤后，验证以下内容：

- [ ] 数据库中user表有is_active、activation_token等新字段
- [ ] 现有用户的is_active=1（可以登录）
- [ ] 邮件发送测试成功
- [ ] 后端服务器启动无错误
- [ ] 前端服务器启动无错误

---

## 🧪 测试新功能

### 测试1：注册流程
1. 访问 http://localhost:8080/register
2. 输入邮箱、用户名、密码
3. 提交后应该显示"激活邮件已发送"
4. 检查邮箱（可能在垃圾邮件）
5. 点击激活链接
6. 应该显示"激活成功"

### 测试2：未激活账户登录
1. 注册但不激活
2. 尝试登录
3. 应该提示"账户未激活"
4. 提供重新发送激活邮件选项

### 测试3：忘记密码
1. 访问 http://localhost:8080/forgot-password
2. 输入邮箱
3. 收到6位数字验证码
4. 输入验证码和新密码
5. 重置成功后登录

---

## ⚠️ 常见问题

### 问题1：邮件发送失败
**错误：** SMTPAuthenticationError 535

**解决：**
1. 检查QQ邮箱是否开启SMTP服务
2. 确认授权码是否正确（`ygcqbeitbnnvechf`）
3. 检查网络是否能访问smtp.qq.com

### 问题2：现有用户无法登录
**错误：** "账户未激活"

**原因：** 忘记执行步骤3

**解决：** 执行SQL更新：
```sql
UPDATE user SET is_active = 1;
```

### 问题3：数据库迁移失败
**错误：** cryptography包不存在

**解决：**
```bash
pip install cryptography==41.0.7
```

---

## 📚 相关文档

- **详细实施指南：** `IMPLEMENTATION_GUIDE.md`
- **完整总结：** `ACCOUNT_REFACTOR_SUMMARY.md`
- **API文档：** 见IMPLEMENTATION_GUIDE.md第7节

---

## 🆘 需要帮助？

如果遇到问题：
1. 查看终端错误信息
2. 检查Django日志
3. 查看浏览器Console
4. 参考详细文档

---

**预计完成时间：** 15-30分钟  
**难度：** ⭐⭐☆☆☆
