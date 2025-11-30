<template>
  <div class="profile-page min-h-screen bg-gray-50 py-12">
    <div class="max-w-3xl mx-auto px-4 sm:px-6 lg:px-8">
      <h1 class="text-3xl font-bold text-gray-900 mb-8">个人中心</h1>

      <div class="bg-white rounded-xl shadow-sm p-8">
        <div class="flex items-center space-x-6 mb-8 pb-8 border-b">
          <div class="w-24 h-24 bg-gradient-to-br from-blue-400 to-blue-600 rounded-full flex items-center justify-center text-white text-3xl font-bold">
            {{ getUserInitial() }}
          </div>
          <div>
            <h2 class="text-2xl font-bold text-gray-900">{{ user?.username }}</h2>
            <p class="text-gray-600">{{ user?.email }}</p>
          </div>
        </div>

        <form @submit.prevent="handleUpdate" class="space-y-6">
          <div>
            <label class="block text-sm font-medium text-gray-700 mb-2">用户名</label>
            <input
              v-model="form.username"
              type="text"
              class="w-full px-4 py-3 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-transparent"
            />
          </div>

          <div>
            <label class="block text-sm font-medium text-gray-700 mb-2">邮箱（不可修改）</label>
            <input
              :value="user?.email"
              type="email"
              disabled
              class="w-full px-4 py-3 border border-gray-300 rounded-lg bg-gray-50 text-gray-500 cursor-not-allowed"
            />
          </div>

          <div class="pt-6 border-t">
            <h3 class="text-lg font-semibold text-gray-900 mb-4">修改密码</h3>
            <div class="space-y-4">
              <div>
                <label class="block text-sm font-medium text-gray-700 mb-2">原密码</label>
                <input
                  v-model="form.old_password"
                  type="password"
                  placeholder="如需修改密码请输入原密码"
                  class="w-full px-4 py-3 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-transparent"
                />
              </div>
              <div>
                <label class="block text-sm font-medium text-gray-700 mb-2">新密码</label>
                <input
                  v-model="form.new_password"
                  type="password"
                  placeholder="输入新密码"
                  class="w-full px-4 py-3 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-transparent"
                />
              </div>
            </div>
          </div>

          <div v-if="message" class="px-4 py-3 rounded-lg text-sm" :class="messageType === 'success' ? 'bg-green-50 border border-green-200 text-green-600' : 'bg-red-50 border border-red-200 text-red-600'">
            {{ message }}
          </div>

          <button
            type="submit"
            :disabled="updating"
            class="w-full bg-blue-600 hover:bg-blue-700 text-white font-medium py-3 rounded-lg transition-colors disabled:opacity-50 disabled:cursor-not-allowed"
          >
            {{ updating ? '保存中...' : '保存修改' }}
          </button>
        </form>
      </div>
    </div>
  </div>
</template>

<script>
import { ref, computed, onMounted } from 'vue'
import { useStore } from 'vuex'
import { accountAPI } from '@/api'

export default {
  name: 'Profile',
  setup() {
    const store = useStore()
    const user = computed(() => store.getters.user)

    const form = ref({
      username: '',
      old_password: '',
      new_password: ''
    })
    const updating = ref(false)
    const message = ref('')
    const messageType = ref('success')

    const getUserInitial = () => {
      return user.value?.username?.charAt(0).toUpperCase() || 'U'
    }

    const loadProfile = async () => {
      try {
        const profile = await accountAPI.getProfile()
        form.value.username = profile.username
      } catch (error) {
        console.error('加载用户信息失败:', error)
      }
    }

    const handleUpdate = async () => {
      updating.value = true
      message.value = ''

      try {
        const updateData = { username: form.value.username }
        if (form.value.old_password && form.value.new_password) {
          updateData.old_password = form.value.old_password
          updateData.new_password = form.value.new_password
        }

        await accountAPI.updateProfile(updateData)
        await store.dispatch('checkLogin')
        message.value = '更新成功'
        messageType.value = 'success'
        form.value.old_password = ''
        form.value.new_password = ''
      } catch (error) {
        message.value = error.error || '更新失败，请重试'
        messageType.value = 'error'
      } finally {
        updating.value = false
      }
    }

    onMounted(() => {
      loadProfile()
    })

    return {
      user,
      form,
      updating,
      message,
      messageType,
      getUserInitial,
      handleUpdate
    }
  }
}
</script>
