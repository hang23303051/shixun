import axios from 'axios'

// 从cookie中获取CSRF token
function getCookie(name) {
  let cookieValue = null
  if (document.cookie && document.cookie !== '') {
    const cookies = document.cookie.split(';')
    for (let i = 0; i < cookies.length; i++) {
      const cookie = cookies[i].trim()
      if (cookie.substring(0, name.length + 1) === (name + '=')) {
        cookieValue = decodeURIComponent(cookie.substring(name.length + 1))
        break
      }
    }
  }
  return cookieValue
}

const instance = axios.create({
  baseURL: '/api',
  timeout: 30000,
  withCredentials: true,
  headers: {
    'Content-Type': 'application/json'
  }
})

// 请求拦截器
instance.interceptors.request.use(
  config => {
    // 对于非GET请求，添加CSRF token
    if (config.method !== 'get') {
      const csrftoken = getCookie('csrftoken')
      if (csrftoken) {
        config.headers['X-CSRFToken'] = csrftoken
      }
    }
    return config
  },
  error => {
    return Promise.reject(error)
  }
)

// 响应拦截器
instance.interceptors.response.use(
  response => {
    return response.data
  },
  error => {
    if (error.response) {
      const { status, data } = error.response
      if (status === 401) {
        console.error('未登录或登录已过期')
      } else if (status === 403) {
        console.error('没有权限')
      } else if (status === 404) {
        console.error('资源不存在')
      } else if (status === 500) {
        console.error('服务器错误')
      }
      return Promise.reject(data)
    }
    return Promise.reject(error)
  }
)

export default instance
