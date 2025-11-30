<template>
  <div id="app">
    <Header v-if="!isAuthPage" />
    <router-view />
  </div>
</template>

<script>
import { computed, onMounted } from 'vue'
import { useRoute } from 'vue-router'
import { useStore } from 'vuex'
import Header from '@/components/Header.vue'

export default {
  name: 'App',
  components: {
    Header
  },
  setup() {
    const route = useRoute()
    const store = useStore()

    const isAuthPage = computed(() => {
      return route.path === '/login' || route.path === '/register'
    })

    onMounted(async () => {
      // 先获取CSRF token
      try {
        await fetch('/api/account/csrf/', {
          credentials: 'include'
        })
      } catch (error) {
        console.error('获取CSRF token失败:', error)
      }
      // 然后检查登录状态
      store.dispatch('checkLogin')
    })

    return {
      isAuthPage
    }
  }
}
</script>

<style>
* {
  margin: 0;
  padding: 0;
  box-sizing: border-box;
}

#app {
  font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, 'Helvetica Neue', Arial, sans-serif;
  -webkit-font-smoothing: antialiased;
  -moz-osx-font-smoothing: grayscale;
  color: #1f2937;
  background-color: #f9fafb;
  min-height: 100vh;
}
</style>
