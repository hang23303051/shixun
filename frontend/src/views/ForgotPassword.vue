<template>
  <div class="min-h-screen bg-gradient-to-br from-blue-50 to-indigo-100 flex items-center justify-center py-12 px-4 sm:px-6 lg:px-8">
    <div class="max-w-md w-full">
      <!-- Logo -->
      <div class="text-center mb-8">
        <router-link to="/" class="inline-flex items-center justify-center w-16 h-16 bg-gradient-to-br from-blue-500 to-indigo-600 rounded-2xl mb-4">
          <span class="text-white font-bold text-2xl">R</span>
        </router-link>
        <h2 class="text-3xl font-extrabold text-gray-900">重置密码</h2>
        <p class="mt-2 text-sm text-gray-600">{{ stepDescription }}</p>
      </div>

      <!-- 表单卡片 -->
      <div class="bg-white rounded-2xl shadow-xl p-8">
        <!-- 步骤指示器 -->
        <div class="flex items-center justify-between mb-8">
          <div v-for="(stepItem, index) in steps" :key="index" class="flex items-center flex-1">
            <div class="flex items-center justify-center w-10 h-10 rounded-full transition-all duration-200"
                 :class="step > index ? 'bg-green-500' : step === index ? 'bg-blue-600' : 'bg-gray-300'">
              <span v-if="step > index" class="text-white">
                <svg class="w-5 h-5" fill="currentColor" viewBox="0 0 20 20">
                  <path fill-rule="evenodd" d="M16.707 5.293a1 1 0 010 1.414l-8 8a1 1 0 01-1.414 0l-4-4a1 1 0 011.414-1.414L8 12.586l7.293-7.293a1 1 0 011.414 0z" clip-rule="evenodd"></path>
                </svg>
              </span>
              <span v-else class="text-white text-sm font-medium">{{ index + 1 }}</span>
            </div>
            <div v-if="index < steps.length - 1" class="flex-1 h-1 mx-2 transition-all duration-200"
                 :class="step > index ? 'bg-green-500' : 'bg-gray-300'"></div>
          </div>
        </div>

        <!-- 消息提示 -->
        <div v-if="message" class="mb-6 p-4 rounded-lg transition-all duration-200"
             :class="messageType === 'error' ? 'bg-red-50 border border-red-200' : 'bg-green-50 border border-green-200'">
          <p :class="messageType === 'error' ? 'text-red-800' : 'text-green-800'">
            {{ message }}
          </p>
        </div>

        <!-- 步骤1：输入邮箱 -->
        <form v-if="step === 0" @submit.prevent="requestReset" class="space-y-6">
          <div>
            <label class="block text-sm font-medium text-gray-700 mb-2">注册邮箱</label>
            <input
              v-model="email"
              type="email"
              required
              placeholder="请输入注册时使用的邮箱"
              class="w-full px-4 py-3 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-transparent transition-all duration-200"
            />
          </div>
          
          <button
            type="submit"
            :disabled="loading"
            class="w-full bg-gradient-to-r from-blue-600 to-indigo-600 text-white font-semibold py-3 px-4 rounded-lg hover:from-blue-700 hover:to-indigo-700 transition-all duration-200 transform hover:scale-105 disabled:opacity-50 disabled:cursor-not-allowed disabled:transform-none"
          >
            {{ loading ? '发送中...' : '发送验证码' }}
          </button>
        </form>

        <!-- 步骤2：输入验证码 -->
        <form v-if="step === 1" @submit.prevent="verifyCode" class="space-y-6">
          <div>
            <label class="block text-sm font-medium text-gray-700 mb-2">验证码</label>
            <input
              v-model="code"
              type="text"
              required
              maxlength="6"
              placeholder="请输入6位数字验证码"
              class="w-full px-4 py-3 border border-gray-300 rounded-lg text-center text-2xl font-mono tracking-widest focus:ring-2 focus:ring-blue-500 focus:border-transparent transition-all duration-200"
            />
            <p class="mt-2 text-sm text-gray-500">验证码已发送至：{{ email }}</p>
          </div>

          <div class="flex space-x-3">
            <button
              type="button"
              @click="step = 0"
              class="flex-1 bg-gray-100 text-gray-700 font-semibold py-3 px-4 rounded-lg hover:bg-gray-200 transition-all duration-200"
            >
              返回
            </button>
            <button
              type="submit"
              :disabled="loading"
              class="flex-1 bg-gradient-to-r from-blue-600 to-indigo-600 text-white font-semibold py-3 px-4 rounded-lg hover:from-blue-700 hover:to-indigo-700 transition-all duration-200 transform hover:scale-105 disabled:opacity-50 disabled:cursor-not-allowed disabled:transform-none"
            >
              {{ loading ? '验证中...' : '验证' }}
            </button>
          </div>

          <button
            type="button"
            @click="requestReset"
            :disabled="loading"
            class="w-full text-sm text-blue-600 hover:text-blue-700 font-medium transition-colors duration-200 disabled:opacity-50"
          >
            没有收到？重新发送
          </button>
        </form>

        <!-- 步骤3：设置新密码 -->
        <form v-if="step === 2" @submit.prevent="resetPassword" class="space-y-6">
          <div>
            <label class="block text-sm font-medium text-gray-700 mb-2">新密码</label>
            <input
              v-model="newPassword"
              type="password"
              required
              minlength="6"
              placeholder="请输入新密码（至少6位）"
              class="w-full px-4 py-3 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-transparent transition-all duration-200"
            />
          </div>

          <div>
            <label class="block text-sm font-medium text-gray-700 mb-2">确认密码</label>
            <input
              v-model="confirmPassword"
              type="password"
              required
              placeholder="请再次输入新密码"
              class="w-full px-4 py-3 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-transparent transition-all duration-200"
            />
          </div>

          <button
            type="submit"
            :disabled="loading"
            class="w-full bg-gradient-to-r from-blue-600 to-indigo-600 text-white font-semibold py-3 px-4 rounded-lg hover:from-blue-700 hover:to-indigo-700 transition-all duration-200 transform hover:scale-105 disabled:opacity-50 disabled:cursor-not-allowed disabled:transform-none"
          >
            {{ loading ? '重置中...' : '重置密码' }}
          </button>
        </form>

        <!-- 返回登录 -->
        <div class="mt-6 text-center">
          <router-link
            to="/login"
            class="text-sm text-blue-600 hover:text-blue-700 font-medium transition-colors duration-200"
          >
            返回登录
          </router-link>
        </div>
      </div>
    </div>
  </div>
</template>

<script>
import { ref, computed } from 'vue'
import { useRouter } from 'vue-router'
import { accountAPI } from '@/api'

export default {
  name: 'ForgotPassword',
  setup() {
    const router = useRouter()
    
    const step = ref(0)
    const steps = ['输入邮箱', '验证码', '新密码']
    
    const email = ref('')
    const code = ref('')
    const newPassword = ref('')
    const confirmPassword = ref('')
    
    const loading = ref(false)
    const message = ref('')
    const messageType = ref('success')

    const stepDescription = computed(() => {
      const descriptions = [
        '请输入您注册时使用的邮箱地址',
        '我们已向您的邮箱发送了验证码',
        '请设置一个新的密码'
      ]
      return descriptions[step.value]
    })

    const requestReset = async () => {
      if (!email.value) {
        message.value = '请输入邮箱地址'
        messageType.value = 'error'
        return
      }

      loading.value = true
      message.value = ''

      try {
        const response = await accountAPI.requestPasswordReset(email.value)
        message.value = response.message || '验证码已发送'
        messageType.value = 'success'
        step.value = 1
        code.value = ''
      } catch (error) {
        message.value = error.error || '发送失败，请重试'
        messageType.value = 'error'
      } finally {
        loading.value = false
      }
    }

    const verifyCode = async () => {
      if (!code.value || code.value.length !== 6) {
        message.value = '请输入6位数字验证码'
        messageType.value = 'error'
        return
      }

      loading.value = true
      message.value = ''

      try {
        await accountAPI.verifyResetCode(email.value, code.value)
        message.value = '验证码正确'
        messageType.value = 'success'
        step.value = 2
      } catch (error) {
        message.value = error.error || '验证码错误或已过期'
        messageType.value = 'error'
      } finally {
        loading.value = false
      }
    }

    const resetPassword = async () => {
      if (!newPassword.value || newPassword.value.length < 6) {
        message.value = '密码至少需要6位'
        messageType.value = 'error'
        return
      }

      if (newPassword.value !== confirmPassword.value) {
        message.value = '两次输入的密码不一致'
        messageType.value = 'error'
        return
      }

      loading.value = true
      message.value = ''

      try {
        const response = await accountAPI.resetPassword(email.value, code.value, newPassword.value)
        message.value = response.message || '密码重置成功'
        messageType.value = 'success'
        
        setTimeout(() => {
          router.push('/login')
        }, 2000)
      } catch (error) {
        message.value = error.error || '重置失败，请重试'
        messageType.value = 'error'
      } finally {
        loading.value = false
      }
    }

    return {
      step,
      steps,
      email,
      code,
      newPassword,
      confirmPassword,
      loading,
      message,
      messageType,
      stepDescription,
      requestReset,
      verifyCode,
      resetPassword
    }
  }
}
</script>
