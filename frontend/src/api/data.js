import request from './axios'

export const dataAPI = {
  // 获取所有参考数据
  getRefData() {
    return request.get('/data/refdata/')
  },

  // 获取主题统计
  getThemeStats() {
    return request.get('/data/refdata/themes/')
  },

  // 按主题筛选
  getDataByTheme(theme) {
    return request.get(`/data/refdata/by_theme/?theme=${theme}`)
  },

  // 获取生成数据
  getGenData() {
    return request.get('/data/gendata/')
  },

  // 按模型筛选生成数据
  getGenDataByModel(modelName) {
    return request.get(`/data/gendata/by_model/?model_name=${modelName}`)
  }
}
