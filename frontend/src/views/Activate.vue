<template>
  <div class="min-h-screen bg-gradient-to-br from-blue-50 to-indigo-100 flex items-center justify-center py-12 px-4 sm:px-6 lg:px-8">
    <div class="max-w-md w-full">
      <!-- Logo -->
      <div class="text-center mb-8">
        <div class="inline-flex items-center justify-center w-16 h-16 bg-gradient-to-br from-blue-500 to-indigo-600 rounded-2xl mb-4">
          <span class="text-white font-bold text-2xl">R</span>
        </div>
        <h2 class="text-3xl font-extrabold text-gray-900">账户激活</h2>
      </div>

      <!-- 加载中 -->
      <div v-if="loading" class="bg-white rounded-2xl shadow-xl p-8">
        <div class="text-center">
          <div class="inline-block animate-spin rounded-full h-12 w-12 border-b-2 border-blue-600 mb-4"></div>
          <p class="text-gray-600">正在激活您的账户...</p>
        </div>
      </div>

      <!-- 成功 -->
      <div v-else-if="success" class="bg-white rounded-2xl shadow-xl p-8">
        <div class="text-center">
          <div class="inline-flex items-center justify-center w-16 h-16 bg-green-100 rounded-full mb-4">
            <svg class="w-8 h-8 text-green-600" fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M5 13l4 4L19 7"></path>
            </svg>
          </div>
          <h3 class="text-2xl font-bold text-gray-900 mb-2">激活成功！</h3>
          <p class="text-gray-600 mb-6">{{ message }}</p>
          <button
            @click="goToLogin"
            class="w-full bg-gradient-to-r from-blue-600 to-indigo-600 text-white font-semibold py-3 px-4 rounded-lg hover:from-blue-700 hover:to-indigo-700 transition-all duration-200 transform hover:scale-105"
          >
            前往登录
          </button>
        </div>
      </div>

      <!-- 失败 -->
      <div v-else class="bg-white rounded-2xl shadow-xl p-8">
        <div class="text-center">
          <div class="inline-flex items-center justify-center w-16 h-16 bg-red-100 rounded-full mb-4">
            <svg class="w-8 h-8 text-red-600" fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M6 18L18 6M6 6l12 12"></path>
            </svg>
          </div>
          <h3 class="text-2xl font-bold text-gray-900 mb-2">激活失败</h3>
          <p class="text-gray-600 mb-2">{{ error }}</p>
          <p class="text-sm text-gray-500 mb-6">{{ detail }}</p>
          <div class="space-y-3">
            <button
              @click="goToRegister"
              class="w-full bg-gradient-to-r from-blue-600 to-indigo-600 text-white font-semibold py-3 px-4 rounded-lg hover:from-blue-700 hover:to-indigo-700 transition-all duration-200"
            >
              重新注册
            </button>
            <button
              @click="goToLogin"
              class="w-full bg-gray-100 text-gray-700 font-semibold py-3 px-4 rounded-lg hover:bg-gray-200 transition-all duration-200"
            >
              返回登录
            </button>
          </div>
        </div>
      </div>
    </div>
  </div>
</template>

<script>
import { ref, onMounted } from 'vue'
import { useRouter, useRoute } from 'vue-router'
import { accountAPI } from '@/api'

export default {
  name: 'Activate',
  setup() {
    const router = useRouter()
    const route = useRoute()
    
    const loading = ref(true)
    const success = ref(false)
    const error = ref('')
    const detail = ref('')
    const message = ref('')

    const activateAccount = async () => {
      const { email, token } = route.params
      
      if (!email || !token) {
        success.value = false
        error.value = '激活链接不完整'
        detail.value = '请检查邮件中的完整链接'
        loading.value = false
        return
      }

      try {
        const response = await accountAPI.activateAccount(email, token)
        success.value = true
        message.value = response.message || '账户激活成功！'
      } catch (err) {
        success.value = false
        error.value = err.error || '激活失败'
        detail.value = err.detail || '请稍后重试或联系管理员'
      } finally {
        loading.value = false
      }
    }

    const goToLogin = () => {
      router.push('/login')
    }

    const goToRegister = () => {
      router.push('/register')
    }

    onMounted(() => {
      activateAccount()
    })

    return {
      loading,
      success,
      error,
      detail,
      message,
      goToLogin,
      goToRegister
    }
  }
}
</script>
