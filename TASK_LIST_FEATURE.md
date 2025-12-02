# 任务列表功能实现说明

## 概述
已成功实现任务列表功能，用于展示所有评测试用提交的内容。该功能包括后端API和前端页面。

## 后端实现

### 1. 数据库表 (task_list)
创建了 `TaskList` 模型，存储所有评测提交记录：

**表名**: `task_list`

**字段**:
- `task_id`: 任务ID（唯一）
- `user_email`: 提交用户邮箱
- `username`: 提交用户名
- `model_name`: 模型名称
- `api_url`: API地址
- `api_key`: API密钥
- `description`: 模型简介
- `publisher`: 发布者
- `parameters`: 参数规模
- `is_open_source`: 是否开源
- `release_date`: 发布时间
- `official_website`: 官网链接
- `status`: 任务状态（pending/processing/completed/failed）
- `progress`: 进度百分比
- `message`: 状态消息
- `semantic_score`: 语义一致性评分
- `temporal_score`: 时序一致性评分
- `motion_score`: 运动属性评分
- `reality_score`: 真实性评分
- `total_score`: 总分
- `model_id`: 关联模型ID
- `created_at`: 创建时间
- `updated_at`: 更新时间
- `completed_at`: 完成时间

### 2. API接口

**基础路径**: `/api/task/`

#### 2.1 获取任务列表
- **URL**: `GET /api/task/tasks/`
- **说明**: 获取所有任务列表
- **返回**: 任务列表数组

#### 2.2 获取任务详情
- **URL**: `GET /api/task/tasks/{task_id}/`
- **说明**: 根据task_id获取任务详情
- **返回**: 任务详细信息

#### 2.3 获取当前用户任务
- **URL**: `GET /api/task/tasks/my_tasks/`
- **说明**: 获取当前登录用户的任务列表
- **需要登录**: 是
- **返回**: 当前用户的任务列表

#### 2.4 获取统计信息
- **URL**: `GET /api/task/tasks/statistics/`
- **说明**: 获取任务统计信息
- **返回**: 
```json
{
  "total": 10,
  "pending": 2,
  "processing": 1,
  "completed": 6,
  "failed": 1
}
```

### 3. 评测提交流程更新

在 `eval_test/views.py` 中的 `EvalSubmitView` 已更新：
- 提交评测时自动保存到 `task_list` 表
- 评测完成时更新 `task_list` 表中的评分和状态
- 查询状态时优先从数据库读取

## 前端实现

### 1. 页面组件
**文件**: `frontend/src/views/TaskList.vue`

**功能**:
- 展示所有评测任务列表
- 显示统计信息（总任务数、等待处理、处理中、已完成、失败）
- 表格展示任务信息：
  - 序号
  - 模型名称
  - API地址
  - 提交用户
  - 状态（带颜色标签）
  - 进度条
  - 总分
  - 提交时间
  - 操作按钮
- 点击"详情"按钮查看任务完整信息
- 刷新按钮重新加载数据

### 2. API封装
**文件**: `frontend/src/api/task.js`

封装了所有任务相关的API调用：
- `getTasks()`: 获取任务列表
- `getTaskDetail(taskId)`: 获取任务详情
- `getMyTasks()`: 获取当前用户任务
- `getStatistics()`: 获取统计信息

### 3. 路由配置
**路径**: `/tasks`
**组件**: `TaskList`
**权限**: 无需登录即可访问

### 4. 导航菜单
在Header组件中添加了"任务列表"导航链接，位于"评测试用"之后。

## 使用说明

### 1. 访问任务列表
- 在导航栏点击"任务列表"
- 或直接访问: `http://localhost:8080/tasks`

### 2. 查看任务详情
- 在任务列表中点击"详情"按钮
- 弹出模态框显示完整的任务信息
- 如果任务已完成，显示四个维度的评分

### 3. 任务状态说明
- **等待处理** (pending): 黄色标签，任务已提交但未开始
- **处理中** (processing): 蓝色标签，正在进行评测
- **已完成** (completed): 绿色标签，评测完成并有评分
- **失败** (failed): 红色标签，评测过程中出现错误

## 数据流程

1. **提交评测**:
   - 用户在"评测试用"页面提交表单
   - 后端创建任务记录保存到 `task_list` 表
   - 状态设置为 `pending`

2. **评测进行中**:
   - 打分算法运行时更新状态为 `processing`
   - 更新进度百分比

3. **评测完成**:
   - 保存评分到 `task_list` 表
   - 创建模型记录到 `model` 表
   - 状态更新为 `completed`
   - 记录完成时间

4. **查看任务**:
   - 用户访问任务列表页面
   - 从数据库读取所有任务记录
   - 实时显示最新状态

## 管理后台

在Django管理后台 (`/admin`) 可以：
- 查看所有任务记录
- 按状态、时间筛选
- 搜索任务ID、模型名称、用户
- 查看和编辑任务详情
- 不允许手动添加任务（只能通过API提交）

## 技术特点

1. **数据持久化**: 所有提交记录永久保存在数据库中
2. **实时更新**: 状态和进度实时更新
3. **完整记录**: 保存提交信息、评测结果、时间戳等完整信息
4. **用户友好**: 清晰的状态标识、进度条、详情弹窗
5. **统计信息**: 实时统计各状态任务数量
6. **响应式设计**: 适配不同屏幕尺寸

## 后续扩展建议

1. **分页功能**: 当任务数量较多时添加分页
2. **筛选功能**: 按状态、用户、时间范围筛选
3. **搜索功能**: 搜索模型名称、任务ID
4. **导出功能**: 导出任务列表为Excel或CSV
5. **批量操作**: 批量删除、重试失败任务
6. **实时推送**: 使用WebSocket实时推送任务状态更新
7. **任务重试**: 对失败的任务提供重试功能

## 文件清单

### 后端文件
- `backend/task_list/models.py` - TaskList模型定义
- `backend/task_list/serializers.py` - 序列化器
- `backend/task_list/views.py` - 视图集
- `backend/task_list/urls.py` - URL配置
- `backend/task_list/admin.py` - 管理后台配置
- `backend/task_list/migrations/0001_initial.py` - 数据库迁移文件
- `backend/eval_test/views.py` - 更新评测提交逻辑
- `backend/backend/settings.py` - 添加task_list应用
- `backend/backend/urls.py` - 添加task_list路由

### 前端文件
- `frontend/src/views/TaskList.vue` - 任务列表页面组件
- `frontend/src/api/task.js` - 任务API封装
- `frontend/src/api/index.js` - 导出taskAPI
- `frontend/src/router/index.js` - 添加任务列表路由
- `frontend/src/components/Header.vue` - 添加导航链接

## 测试步骤

1. 启动后端服务器
2. 启动前端开发服务器
3. 访问"评测试用"页面提交一个测试任务
4. 访问"任务列表"页面查看提交的任务
5. 使用mock-complete接口完成评测
6. 刷新任务列表查看更新后的状态和评分
7. 点击"详情"按钮查看完整信息

## 完成时间
2025-12-02
