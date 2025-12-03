<template>
  <div class="min-h-screen bg-gradient-to-br from-blue-50 via-white to-blue-50 flex items-center justify-center py-12 px-4 sm:px-6 lg:px-8">
    <div class="max-w-md w-full">
      <div class="bg-white rounded-2xl shadow-xl p-8">
        <!-- 注册成功提示 -->
        <div v-if="registrationSuccess" class="text-center">
          <div class="inline-flex items-center justify-center w-16 h-16 bg-green-100 rounded-full mb-4">
            <svg class="w-8 h-8 text-green-600" fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M3 19v-8.93a2 2 0 01.89-1.664l7-4.666a2 2 0 012.22 0l7 4.666A2 2 0 0121 10.07V19M3 19a2 2 0 002 2h14a2 2 0 002-2M3 19l6.75-4.5M21 19l-6.75-4.5M3 10l6.75 4.5M21 10l-6.75 4.5m0 0l-1.14.76a2 2 0 01-2.22 0l-1.14-.76"></path>
            </svg>
          </div>
          <h3 class="text-2xl font-bold text-gray-900 mb-2">注册成功！</h3>
          <p class="text-gray-600 mb-4">{{ successMessage }}</p>
          <div class="bg-blue-50 border border-blue-200 rounded-lg p-4 mb-6">
            <p class="text-sm text-blue-800 mb-3">
              <strong>激活邮件已发送至：</strong><br>
              {{ registeredEmail }}
            </p>
            <p class="text-xs text-blue-600">
              ✓ 请查收邮件并点击激活链接<br>
              ✓ 激活链接24小时内有效<br>
              ✓ 激活后即可登录使用
            </p>
          </div>
          <div class="space-y-3">
            <button
              @click="resendActivation"
              :disabled="resending"
              class="w-full bg-gray-100 hover:bg-gray-200 text-gray-700 font-medium py-3 rounded-lg transition-colors disabled:opacity-50 disabled:cursor-not-allowed"
            >
              {{ resending ? '发送中...' : '重新发送激活邮件' }}
            </button>
            <router-link
              to="/login"
              class="block w-full bg-blue-600 hover:bg-blue-700 text-white font-medium py-3 rounded-lg transition-colors text-center"
            >
              前往登录
            </router-link>
          </div>
          <p class="mt-4 text-xs text-gray-500">
            没有收到邮件？请检查垃圾邮件箱
          </p>
        </div>

        <!-- 注册表单 -->
        <div v-else>
          <!-- Header -->
          <div class="text-center mb-8">
            <h2 class="text-3xl font-bold text-gray-900">创建账户</h2>
            <p class="mt-2 text-sm text-gray-600">加入Ref4D评测平台</p>
          </div>

          <!-- Form -->
          <form @submit.prevent="handleRegister" class="space-y-6">
            <div>
              <label class="block text-sm font-medium text-gray-700 mb-2">邮箱</label>
              <input
                v-model="form.email"
                type="email"
                required
                class="w-full px-4 py-3 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-transparent transition-all"
                placeholder="请输入邮箱地址"
              />
            </div>

            <div>
              <label class="block text-sm font-medium text-gray-700 mb-2">用户名</label>
              <input
                v-model="form.username"
                type="text"
                required
                class="w-full px-4 py-3 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-transparent transition-all"
                placeholder="请输入用户名"
              />
            </div>

            <div>
              <label class="block text-sm font-medium text-gray-700 mb-2">密码</label>
              <input
                v-model="form.password"
                type="password"
                required
                minlength="6"
                class="w-full px-4 py-3 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-transparent transition-all"
                placeholder="请输入密码（至少6位）"
              />
            </div>

            <div>
              <label class="block text-sm font-medium text-gray-700 mb-2">确认密码</label>
              <input
                v-model="form.password_confirm"
                type="password"
                required
                class="w-full px-4 py-3 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-transparent transition-all"
                placeholder="请再次输入密码"
              />
            </div>

            <div v-if="error" class="bg-red-50 border border-red-200 text-red-600 px-4 py-3 rounded-lg text-sm">
              {{ error }}
            </div>

            <button
              type="submit"
              :disabled="loading"
              class="w-full bg-blue-600 hover:bg-blue-700 text-white font-medium py-3 rounded-lg transition-colors disabled:opacity-50 disabled:cursor-not-allowed"
            >
              {{ loading ? '注册中...' : '注册' }}
            </button>
          </form>

          <!-- Footer -->
          <div class="mt-6 text-center">
            <p class="text-sm text-gray-600">
              已有账号？
              <router-link to="/login" class="text-blue-600 hover:text-blue-700 font-medium">
                立即登录
              </router-link>
            </p>
          </div>
        </div>
      </div>
    </div>
  </div>
</template>

<script>
import { ref } from 'vue'
import { accountAPI } from '@/api'

export default {
  name: 'Register',
  setup() {
    const form = ref({
      email: '',
      username: '',
      password: '',
      password_confirm: ''
    })
    const loading = ref(false)
    const error = ref('')
    const registrationSuccess = ref(false)
    const registeredEmail = ref('')
    const successMessage = ref('')
    const resending = ref(false)

    const handleRegister = async () => {
      loading.value = true
      error.value = ''

      if (form.value.password !== form.value.password_confirm) {
        error.value = '两次密码输入不一致'
        loading.value = false
        return
      }

      if (form.value.password.length < 6) {
        error.value = '密码至少需要6位'
        loading.value = false
        return
      }

      try {
        const response = await accountAPI.register(form.value)
        
        // 注册成功，显示激活提示
        if (response.require_activation) {
          registrationSuccess.value = true
          registeredEmail.value = response.email
          successMessage.value = response.message || '激活邮件已发送'
        }
      } catch (err) {
        error.value = err.error || err.email?.[0] || err.detail || '注册失败，请重试'
      } finally {
        loading.value = false
      }
    }

    const resendActivation = async () => {
      resending.value = true
      try {
        await accountAPI.resendActivationEmail(registeredEmail.value)
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
      registrationSuccess,
      registeredEmail,
      successMessage,
      resending,
      handleRegister,
      resendActivation
    }
  }
}
</script>
