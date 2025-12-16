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
    // 后端接口：/api/account/check-login/
    return request.get('/account/check-login/')
  },

  // 获取用户信息
  getProfile() {
    return request.get('/account/profile/')
  },

  // 更新用户信息
  updateProfile(data) {
    return request.put('/account/profile/', data)
  },

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
