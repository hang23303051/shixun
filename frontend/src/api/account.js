import request from './axios'

export const accountAPI = {
  // 注册
  register(data) {
    return request.post('/account/register/', data)
  },

  // 登录
  login(data) {
    return request.post('/account/login/', data)
  },

  // 登出
  logout() {
    return request.post('/account/logout/')
  },

  // 检查登录状态
  checkLogin() {
    return request.get('/account/check-login/')
  },

  // 获取用户信息
  getProfile() {
    return request.get('/account/profile/')
  },

  // 更新用户信息
  updateProfile(data) {
    return request.put('/account/profile/', data)
  }
}
