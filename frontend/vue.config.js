const { defineConfig } = require('@vue/cli-service')
const os = require('os')

// 获取本机局域网IP - 优先真实局域网，跳过虚拟网卡
function getLocalIP() {
  const interfaces = os.networkInterfaces()
  const candidates = []

  for (const name of Object.keys(interfaces)) {
    for (const iface of interfaces[name]) {
      // 跳过内部地址和非IPv4地址
      if (iface.family === 'IPv4' && !iface.internal) {
        const ip = iface.address

        // 优先级1: 192.168.x.x (最常见的局域网)
        if (ip.startsWith('192.168.')) {
          return ip
        }

        // 优先级2: 172.16.x.x - 172.31.x.x (企业网络)
        const match = ip.match(/^172\.(\d+)\./)
        if (match) {
          const second = parseInt(match[1])
          if (second >= 16 && second <= 31) {
            return ip
          }
        }

        // 优先级3: 10.x.x.x (大型企业网络)
        if (ip.startsWith('10.')) {
          return ip
        }

        // 记录其他IP作为备选
        candidates.push(ip)
      }
    }
  }

  // 如果没有找到标准局域网IP，使用第一个候选
  return candidates.length > 0 ? candidates[0] : 'localhost'
}

const localIP = getLocalIP()
console.log(`[Vue Config] 检测到本机IP: ${localIP}`)

// 获取后端地址
const BACKEND_HOST = process.env.VUE_APP_BACKEND_HOST || localIP
const BACKEND_PORT = process.env.VUE_APP_BACKEND_PORT || '8000'
const BACKEND_URL = `http://${BACKEND_HOST}:${BACKEND_PORT}`

console.log(`[Vue Config] 后端代理地址: ${BACKEND_URL}`)

module.exports = defineConfig({
  transpileDependencies: true,
  devServer: {
    host: '0.0.0.0',  // 允许外部访问
    port: 8080,
    proxy: {
      '/api': {
        target: BACKEND_URL,
        changeOrigin: true,
        ws: true,
        onProxyReq: (proxyReq, req, res) => {
          console.log(`[Proxy] ${req.method} ${req.url} -> ${BACKEND_URL}${req.url}`)
        }
      },
      '/media': {
        target: BACKEND_URL,
        changeOrigin: true,
        onProxyReq: (proxyReq, req, res) => {
          console.log(`[Proxy] ${req.method} ${req.url} -> ${BACKEND_URL}${req.url}`)
        }
      }
    }
  }
})
