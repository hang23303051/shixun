const { defineConfig } = require('@vue/cli-service')

// 获取后端地址（支持环境变量或默认localhost）
const BACKEND_HOST = process.env.VUE_APP_BACKEND_HOST || 'localhost'
const BACKEND_PORT = process.env.VUE_APP_BACKEND_PORT || '8000'
const BACKEND_URL = `http://${BACKEND_HOST}:${BACKEND_PORT}`

module.exports = defineConfig({
  transpileDependencies: true,
  devServer: {
    host: '0.0.0.0',  // 允许外部访问
    port: 8080,
    proxy: {
      '/api': {
        target: BACKEND_URL,
        changeOrigin: true,
        ws: true
      },
      '/media': {
        target: BACKEND_URL,
        changeOrigin: true
      }
    }
  }
})
