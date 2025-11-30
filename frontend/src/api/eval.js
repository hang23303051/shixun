import request from './axios'

export const evalAPI = {
  // 提交评测任务
  submitEval(data) {
    return request.post('/eval/submit/', data)
  },

  // 查询评测状态
  getEvalStatus(taskId) {
    return request.get(`/eval/status/${taskId}/`)
  },

  // 模拟评测完成（仅用于测试）
  mockComplete(taskId, scores) {
    return request.post(`/eval/mock-complete/${taskId}/`, scores)
  }
}
