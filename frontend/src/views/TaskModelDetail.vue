<template>
  <div class="model-detail min-h-screen bg-gray-50 py-12">
    <div class="max-w-5xl mx-auto px-4 sm:px-6 lg:px-8">
      <!-- Loading -->
      <div v-if="loading" class="flex justify-center items-center py-20">
        <div class="animate-spin rounded-full h-12 w-12 border-b-2 border-blue-600"></div>
      </div>

      <!-- Content -->
      <div v-else-if="task" class="space-y-6">
        <!-- Header -->
        <div class="bg-white rounded-xl shadow-sm p-8">
          <div class="flex justify-between items-start mb-6">
            <div>
              <h1 class="text-3xl font-bold text-gray-900 mb-2">{{ task.model_name }}</h1>
              <p class="text-gray-600">{{ task.publisher || '未知发布者' }}</p>
            </div>
            <div v-if="task.status === 'completed'" class="text-right">
              <div class="text-4xl font-bold text-blue-600">{{ task.total_score?.toFixed(1) || '0.0' }}</div>
              <div class="text-sm text-gray-500">综合评分</div>
            </div>
            <div v-else class="text-right">
              <span 
                class="inline-flex items-center px-4 py-2 rounded-full text-sm font-medium"
                :class="getStatusClass(task.status)"
              >
                {{ task.status_display }}
              </span>
            </div>
          </div>

          <!-- Info Grid -->
          <div class="grid grid-cols-2 md:grid-cols-4 gap-4 mb-6">
            <div>
              <div class="text-sm text-gray-500">参数规模</div>
              <div class="font-semibold text-gray-900">{{ task.parameters || '未知' }}</div>
            </div>
            <div>
              <div class="text-sm text-gray-500">是否开源</div>
              <div class="font-semibold text-gray-900">{{ task.is_open_source ? '是' : '否' }}</div>
            </div>
            <div>
              <div class="text-sm text-gray-500">发布时间</div>
              <div class="font-semibold text-gray-900">{{ task.release_date || '未知' }}</div>
            </div>
            <div>
              <div class="text-sm text-gray-500">测试人</div>
              <div class="font-semibold text-gray-900">{{ task.username }}</div>
            </div>
          </div>

          <!-- Description -->
          <div class="border-t pt-6">
            <h3 class="font-semibold text-gray-900 mb-2">模型简介</h3>
            <p class="text-gray-600 leading-relaxed">{{ task.description || '暂无简介' }}</p>
          </div>

          <!-- Official Website -->
          <div v-if="task.official_website" class="border-t pt-6">
            <h3 class="font-semibold text-gray-900 mb-2">官方网站</h3>
            <a 
              :href="task.official_website" 
              target="_blank"
              class="text-blue-600 hover:text-blue-700 hover:underline"
            >
              {{ task.official_website }}
            </a>
          </div>
        </div>

        <!-- Scores (only for completed tasks) -->
        <div v-if="task.status === 'completed'" class="bg-white rounded-xl shadow-sm p-8">
          <h2 class="text-2xl font-bold text-gray-900 mb-6">评测得分</h2>
          
          <div class="grid grid-cols-2 md:grid-cols-4 gap-6 mb-8">
            <div class="text-center">
              <div class="text-3xl font-bold text-blue-600 mb-2">{{ task.semantic_score?.toFixed(1) || '0.0' }}</div>
              <div class="text-sm text-gray-600">语义一致性</div>
            </div>
            <div class="text-center">
              <div class="text-3xl font-bold text-green-600 mb-2">{{ task.temporal_score?.toFixed(1) || '0.0' }}</div>
              <div class="text-sm text-gray-600">时序一致性</div>
            </div>
            <div class="text-center">
              <div class="text-3xl font-bold text-purple-600 mb-2">{{ task.motion_score?.toFixed(1) || '0.0' }}</div>
              <div class="text-sm text-gray-600">运动属性</div>
            </div>
            <div class="text-center">
              <div class="text-3xl font-bold text-orange-600 mb-2">{{ task.reality_score?.toFixed(1) || '0.0' }}</div>
              <div class="text-sm text-gray-600">真实性</div>
            </div>
          </div>

          <!-- Score Chart -->
          <ScoreChart
            v-if="task.semantic_score"
            :scores="{
              semantic: task.semantic_score,
              temporal: task.temporal_score,
              motion: task.motion_score,
              reality: task.reality_score,
              total: task.total_score
            }"
            :model-name="task.model_name"
            chart-height="400px"
          />
        </div>

        <!-- Status Message (for non-completed tasks) -->
        <div v-else class="bg-white rounded-xl shadow-sm p-8">
          <h2 class="text-2xl font-bold text-gray-900 mb-6">任务状态</h2>
          
          <div class="text-center py-8">
            <div class="mb-4">
              <div v-if="task.status === 'pending'" class="w-16 h-16 bg-yellow-100 rounded-full flex items-center justify-center mx-auto mb-4">
                <svg class="w-8 h-8 text-yellow-600" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M12 8v4l3 3m6-3a9 9 0 11-18 0 9 9 0 0118 0z" />
                </svg>
              </div>
              <div v-else-if="task.status === 'processing'" class="w-16 h-16 bg-blue-100 rounded-full flex items-center justify-center mx-auto mb-4">
                <div class="animate-spin rounded-full h-8 w-8 border-b-2 border-blue-600"></div>
              </div>
              <div v-else-if="task.status === 'failed'" class="w-16 h-16 bg-red-100 rounded-full flex items-center justify-center mx-auto mb-4">
                <svg class="w-8 h-8 text-red-600" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M6 18L18 6M6 6l12 12" />
                </svg>
              </div>
            </div>
            
            <h3 class="text-xl font-semibold text-gray-900 mb-2">{{ task.status_display }}</h3>
            <p v-if="task.status !== 'failed'" class="text-gray-600">{{ task.message || '任务正在处理中，请稍候...' }}</p>
          </div>
        </div>

        <!-- Task Info -->
        <div class="bg-white rounded-xl shadow-sm p-8">
          <h2 class="text-2xl font-bold text-gray-900 mb-6">任务信息</h2>
          
          <div class="space-y-4">
            <div class="flex justify-between py-3 border-b">
              <span class="text-gray-600">任务ID</span>
              <span class="font-mono text-sm text-gray-900">{{ task.task_id }}</span>
            </div>
            <div class="flex justify-between py-3 border-b">
              <span class="text-gray-600">提交时间</span>
              <span class="text-gray-900">{{ formatDate(task.created_at) }}</span>
            </div>
            <div class="flex justify-between py-3 border-b">
              <span class="text-gray-600">完成时间</span>
              <span class="text-gray-900">{{ formatDate(task.completed_at) }}</span>
            </div>
            <div class="flex justify-between py-3 border-b">
              <span class="text-gray-600">任务状态</span>
              <span 
                class="inline-flex items-center px-3 py-1 rounded-full text-xs font-medium"
                :class="getStatusClass(task.status)"
              >
                {{ task.status_display }}
              </span>
            </div>
          </div>
        </div>

        <!-- Back Button -->
        <div class="flex justify-center">
          <button
            @click="$router.push('/tasks')"
            class="px-6 py-3 bg-gray-100 hover:bg-gray-200 text-gray-700 font-medium rounded-lg transition-colors"
          >
            返回任务列表
          </button>
        </div>
      </div>

      <!-- Error -->
      <div v-else class="text-center py-20">
        <p class="text-gray-600 mb-4">任务不存在或已被删除</p>
        <button
          @click="$router.push('/tasks')"
          class="px-6 py-3 bg-blue-600 hover:bg-blue-700 text-white font-medium rounded-lg transition-colors"
        >
          返回任务列表
        </button>
      </div>
    </div>
  </div>
</template>

<script>
import { ref, onMounted } from 'vue'
import { useRoute } from 'vue-router'
import { taskAPI } from '@/api'
import ScoreChart from '@/components/ScoreChart.vue'

export default {
  name: 'TaskModelDetail',
  components: {
    ScoreChart
  },
  setup() {
    const route = useRoute()
    const task = ref(null)
    const loading = ref(true)

    const loadTask = async () => {
      loading.value = true
      try {
        const taskId = route.params.taskId
        task.value = await taskAPI.getTaskDetail(taskId)
      } catch (error) {
        console.error('加载任务详情失败:', error)
        task.value = null
      } finally {
        loading.value = false
      }
    }

    const formatDate = (dateString) => {
      if (!dateString) return '-'
      const date = new Date(dateString)
      return date.toLocaleString('zh-CN', {
        year: 'numeric',
        month: '2-digit',
        day: '2-digit',
        hour: '2-digit',
        minute: '2-digit'
      })
    }

    const getStatusClass = (status) => {
      const classes = {
        pending: 'bg-yellow-100 text-yellow-800',
        processing: 'bg-blue-100 text-blue-800',
        completed: 'bg-green-100 text-green-800',
        failed: 'bg-red-100 text-red-800'
      }
      return classes[status] || 'bg-gray-100 text-gray-800'
    }

    onMounted(() => {
      loadTask()
    })

    return {
      task,
      loading,
      formatDate,
      getStatusClass
    }
  }
}
</script>
