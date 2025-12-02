import { createRouter, createWebHistory } from 'vue-router'
import { accountAPI } from '@/api'

const routes = [
  {
    path: '/',
    name: 'Home',
    component: () => import('@/views/Home.vue'),
    meta: { requiresAuth: false }
  },
  {
    path: '/dataset',
    name: 'Dataset',
    component: () => import('@/views/Dataset.vue'),
    meta: { requiresAuth: false }
  },
  {
    path: '/models',
    name: 'Models',
    component: () => import('@/views/Models.vue'),
    meta: { requiresAuth: false }
  },
  {
    path: '/model/:id',
    name: 'ModelDetail',
    component: () => import('@/views/ModelDetail.vue'),
    meta: { requiresAuth: false }
  },
  {
    path: '/ranking',
    name: 'Ranking',
    component: () => import('@/views/Ranking.vue'),
    meta: { requiresAuth: false }
  },
  {
    path: '/evaluation',
    name: 'Evaluation',
    component: () => import('@/views/Evaluation.vue'),
    meta: { requiresAuth: true }
  },
  {
    path: '/tasks',
    name: 'TaskList',
    component: () => import('@/views/TaskList.vue'),
    meta: { requiresAuth: true }
  },
  {
    path: '/task/:taskId',
    name: 'TaskModelDetail',
    component: () => import('@/views/TaskModelDetail.vue'),
    meta: { requiresAuth: true }
  },
  {
    path: '/profile',
    name: 'Profile',
    component: () => import('@/views/Profile.vue'),
    meta: { requiresAuth: true }
  },
  {
    path: '/login',
    name: 'Login',
    component: () => import('@/views/Login.vue'),
    meta: { requiresAuth: false }
  },
  {
    path: '/register',
    name: 'Register',
    component: () => import('@/views/Register.vue'),
    meta: { requiresAuth: false }
  }
]

const router = createRouter({
  history: createWebHistory(process.env.BASE_URL),
  routes
})

// 路由守卫 - 检查登录状态
router.beforeEach(async (to, from, next) => {
  if (to.meta.requiresAuth) {
    try {
      const res = await accountAPI.checkLogin()
      if (res.logged_in) {
        next()
      } else {
        next({ name: 'Login', query: { redirect: to.fullPath } })
      }
    } catch (error) {
      next({ name: 'Login', query: { redirect: to.fullPath } })
    }
  } else {
    next()
  }
})

export default router
