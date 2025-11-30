<template>
  <header class="bg-white shadow-sm sticky top-0 z-50">
    <nav class="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
      <div class="flex justify-between h-16">
        <!-- Logo and Navigation Links -->
        <div class="flex">
          <!-- Logo -->
          <div class="flex-shrink-0 flex items-center">
            <router-link to="/" class="flex items-center space-x-2">
              <div class="w-8 h-8 bg-gradient-to-br from-blue-500 to-blue-600 rounded-lg flex items-center justify-center">
                <span class="text-white font-bold text-lg">R</span>
              </div>
              <span class="text-xl font-bold bg-gradient-to-r from-blue-600 to-blue-500 bg-clip-text text-transparent">
                Ref4D
              </span>
            </router-link>
          </div>

          <!-- Navigation Links -->
          <div class="hidden sm:ml-8 sm:flex sm:space-x-8">
            <router-link
              v-for="item in navItems"
              :key="item.name"
              :to="item.path"
              class="inline-flex items-center px-1 pt-1 text-sm font-medium transition-colors"
              :class="isActive(item.path) ? 'text-blue-600 border-b-2 border-blue-600' : 'text-gray-700 hover:text-blue-600'"
            >
              {{ item.name }}
            </router-link>
          </div>
        </div>

        <!-- User Menu -->
        <div class="flex items-center space-x-4">
          <template v-if="isLoggedIn">
            <router-link
              to="/profile"
              class="text-gray-700 hover:text-blue-600 px-3 py-2 rounded-md text-sm font-medium transition-colors"
            >
              个人中心
            </router-link>
            <button
              @click="handleLogout"
              class="bg-gray-100 hover:bg-gray-200 text-gray-700 px-4 py-2 rounded-lg text-sm font-medium transition-colors"
            >
              登出
            </button>
          </template>
          <template v-else>
            <router-link
              to="/login"
              class="text-gray-700 hover:text-blue-600 px-3 py-2 rounded-md text-sm font-medium transition-colors"
            >
              登录
            </router-link>
            <router-link
              to="/register"
              class="bg-blue-600 hover:bg-blue-700 text-white px-4 py-2 rounded-lg text-sm font-medium transition-colors shadow-sm"
            >
              注册
            </router-link>
          </template>
        </div>

        <!-- Mobile menu button -->
        <div class="flex items-center sm:hidden">
          <button
            @click="mobileMenuOpen = !mobileMenuOpen"
            class="inline-flex items-center justify-center p-2 rounded-md text-gray-700 hover:text-blue-600 hover:bg-gray-100"
          >
            <svg class="h-6 w-6" fill="none" viewBox="0 0 24 24" stroke="currentColor">
              <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M4 6h16M4 12h16M4 18h16" />
            </svg>
          </button>
        </div>
      </div>

      <!-- Mobile menu -->
      <div v-if="mobileMenuOpen" class="sm:hidden pb-4">
        <div class="space-y-1">
          <router-link
            v-for="item in navItems"
            :key="item.name"
            :to="item.path"
            class="block px-3 py-2 rounded-md text-base font-medium"
            :class="isActive(item.path) ? 'bg-blue-50 text-blue-600' : 'text-gray-700 hover:bg-gray-50 hover:text-blue-600'"
            @click="mobileMenuOpen = false"
          >
            {{ item.name }}
          </router-link>
        </div>
      </div>
    </nav>
  </header>
</template>

<script>
import { computed, ref } from 'vue'
import { useStore } from 'vuex'
import { useRouter, useRoute } from 'vue-router'

export default {
  name: 'Header',
  setup() {
    const store = useStore()
    const router = useRouter()
    const route = useRoute()
    const mobileMenuOpen = ref(false)

    const navItems = [
      { name: '首页', path: '/' },
      { name: '数据集', path: '/dataset' },
      { name: '模型', path: '/models' },
      { name: '排行榜', path: '/ranking' },
      { name: '评测试用', path: '/evaluation' }
    ]

    const isLoggedIn = computed(() => store.getters.isLoggedIn)

    const isActive = (path) => {
      if (path === '/') {
        return route.path === '/'
      }
      return route.path.startsWith(path)
    }

    const handleLogout = async () => {
      try {
        await store.dispatch('logout')
        router.push('/')
      } catch (error) {
        console.error('登出失败:', error)
      }
    }

    return {
      navItems,
      isLoggedIn,
      mobileMenuOpen,
      isActive,
      handleLogout
    }
  }
}
</script>
