import request from './axios'

export const modelAPI = {
  // 获取所有模型列表
  getModelList() {
    return request.get('/model/models/')
  },

  // 获取模型详情
  getModelDetail(id) {
    return request.get(`/model/models/${id}/`)
  },

  // 获取排行榜
  getRanking(dimension = 'total') {
    return request.get(`/model/models/ranking/?dimension=${dimension}`)
  },

  // 获取模型评分
  getModelScores(id) {
    return request.get(`/model/models/${id}/scores/`)
  }
}
