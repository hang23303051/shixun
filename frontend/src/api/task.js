import request from './axios'

export const taskAPI = {
  // 获取任务列表
  getTasks() {
    return request.get('/task/tasks/')
  },

  // 获取任务详情
  getTaskDetail(taskId) {
    return request.get(`/task/tasks/${taskId}/`)
  },

  // 获取当前用户的任务
  getMyTasks() {
    return request.get('/task/tasks/my_tasks/')
  },

  // 获取任务统计信息
  getStatistics() {
    return request.get('/task/tasks/statistics/')
  },

  // 删除任务
  deleteTask(taskId) {
    return request.delete(`/task/tasks/${taskId}/`)
  }
}
