<template>
  <div class="min-h-screen bg-gradient-to-br from-blue-50 via-white to-blue-50 flex items-center justify-center py-12 px-4 sm:px-6 lg:px-8">
    <div class="max-w-md w-full">
      <div class="bg-white rounded-2xl shadow-xl p-8">
        <!-- Header -->
        <div class="text-center mb-8">
          <h2 class="text-3xl font-bold text-gray-900">欢迎回来</h2>
          <p class="mt-2 text-sm text-gray-600">登录您的Ref4D账户</p>
        </div>

        <!-- Form -->
        <form @submit.prevent="handleLogin" class="space-y-6">
          <div>
            <label class="block text-sm font-medium text-gray-700 mb-2">用户名/邮箱</label>
            <input
              v-model="form.username"
              type="text"
              required
              class="w-full px-4 py-3 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-transparent transition-all"
              placeholder="请输入用户名或邮箱"
            />
          </div>

          <div>
            <div class="flex items-center justify-between mb-2">
              <label class="block text-sm font-medium text-gray-700">密码</label>
              <router-link to="/forgot-password" class="text-sm text-blue-600 hover:text-blue-700 font-medium">
                忘记密码？
              </router-link>
            </div>
            <input
              v-model="form.password"
              type="password"
              required
              class="w-full px-4 py-3 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-transparent transition-all"
              placeholder="请输入密码"
            />
          </div>

          <!-- 未激活账户提示 -->
          <div v-if="needActivation" class="bg-yellow-50 border border-yellow-200 rounded-lg p-4">
            <div class="flex items-start">
              <svg class="w-5 h-5 text-yellow-600 mr-2 mt-0.5" fill="currentColor" viewBox="0 0 20 20">
                <path fill-rule="evenodd" d="M8.257 3.099c.765-1.36 2.722-1.36 3.486 0l5.58 9.92c.75 1.334-.213 2.98-1.742 2.98H4.42c-1.53 0-2.493-1.646-1.743-2.98l5.58-9.92zM11 13a1 1 0 11-2 0 1 1 0 012 0zm-1-8a1 1 0 00-1 1v3a1 1 0 002 0V6a1 1 0 00-1-1z" clip-rule="evenodd"></path>
              </svg>
              <div class="flex-1">
                <h4 class="text-sm font-medium text-yellow-800 mb-1">账户未激活</h4>
                <p class="text-sm text-yellow-700 mb-2">请前往 <strong>{{ userEmail }}</strong> 查收激活邮件</p>
                <button
                  @click="resendActivation"
                  :disabled="resending"
                  class="text-sm text-blue-600 hover:text-blue-700 font-medium underline disabled:opacity-50"
                >
                  {{ resending ? '发送中...' : '重新发送激活邮件' }}
                </button>
              </div>
            </div>
          </div>

          <div v-else-if="error" class="bg-red-50 border border-red-200 text-red-600 px-4 py-3 rounded-lg text-sm">
            {{ error }}
          </div>

          <button
            type="submit"
            :disabled="loading"
            class="w-full bg-blue-600 hover:bg-blue-700 text-white font-medium py-3 rounded-lg transition-colors disabled:opacity-50 disabled:cursor-not-allowed"
          >
            {{ loading ? '登录中...' : '登录' }}
          </button>
        </form>

        <!-- Footer -->
        <div class="mt-6 text-center">
          <p class="text-sm text-gray-600">
            还没有账号？
            <router-link to="/register" class="text-blue-600 hover:text-blue-700 font-medium">
              立即注册
            </router-link>
          </p>
        </div>
      </div>
    </div>
  </div>
</template>

<script>
import { ref } from 'vue'
import { useStore } from 'vuex'
import { useRouter, useRoute } from 'vue-router'
import { accountAPI } from '@/api'

export default {
  name: 'Login',
  setup() {
    const store = useStore()
    const router = useRouter()
    const route = useRoute()

    const form = ref({
      username: '',
      password: ''
    })
    const loading = ref(false)
    const error = ref('')
    const needActivation = ref(false)
    const userEmail = ref('')
    const resending = ref(false)

    const handleLogin = async () => {
      loading.value = true
      error.value = ''
      needActivation.value = false

      try {
        await store.dispatch('login', form.value)
        const redirect = route.query.redirect || '/'
        router.push(redirect)
      } catch (err) {
        // 检查是否是未激活账户
        if (err.require_activation) {
          needActivation.value = true
          userEmail.value = err.email || ''
          error.value = ''
        } else {
          error.value = err.error || err.detail || '登录失败，请检查用户名和密码'
        }
      } finally {
        loading.value = false
      }
    }

    const resendActivation = async () => {
      resending.value = true
      try {
        await accountAPI.resendActivationEmail(userEmail.value)
        alert('激活邮件已重新发送，请查收邮箱')
      } catch (err) {
        alert(err.error || '发送失败，请稍后重试')
      } finally {
        resending.value = false
      }
    }

    return {
      form,
      loading,
      error,
      needActivation,
      userEmail,
      resending,
      handleLogin,
      resendActivation
    }
  }
}
</script>
