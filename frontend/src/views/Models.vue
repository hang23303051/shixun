<template>
  <div class="models-page min-h-screen bg-gray-50 py-12">
    <div class="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
      <!-- Header -->
      <div class="mb-8">
        <h1 class="text-3xl font-bold text-gray-900 mb-2">视频生成模型</h1>
        <p class="text-gray-600">浏览已评测的视频生成模型及其性能表现</p>
      </div>

      <!-- Loading State -->
      <div v-if="loading" class="flex justify-center items-center py-20">
        <div class="animate-spin rounded-full h-12 w-12 border-b-2 border-blue-600"></div>
      </div>

      <!-- Model Grid -->
      <div v-else-if="models.length > 0" class="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
        <div
          v-for="model in models"
          :key="model.id"
          @click="goToDetail(model.id)"
          class="bg-white rounded-xl shadow-sm hover:shadow-lg transition-all cursor-pointer overflow-hidden border border-gray-100"
        >
          <div class="p-6">
            <!-- Model Name and Score -->
            <div class="flex justify-between items-start mb-4">
              <h3 class="text-xl font-bold text-gray-900 flex-1">{{ model.name }}</h3>
              <div class="ml-4 text-right">
                <div class="text-2xl font-bold text-blue-600">{{ model.total_score.toFixed(1) }}</div>
                <div class="text-xs text-gray-500">总分</div>
              </div>
            </div>

            <!-- Publisher -->
            <div class="text-sm text-gray-600 mb-4">{{ model.publisher }}</div>

            <!-- Score Badges -->
            <div class="grid grid-cols-2 gap-2 mb-4">
              <div class="bg-blue-50 rounded-lg px-3 py-2">
                <div class="text-xs text-gray-600">语义一致性</div>
                <div class="text-lg font-semibold text-blue-600">{{ model.semantic_score.toFixed(1) }}</div>
              </div>
              <div class="bg-blue-50 rounded-lg px-3 py-2">
                <div class="text-xs text-gray-600">时序一致性</div>
                <div class="text-lg font-semibold text-blue-600">{{ model.temporal_score.toFixed(1) }}</div>
              </div>
              <div class="bg-blue-50 rounded-lg px-3 py-2">
                <div class="text-xs text-gray-600">运动属性</div>
                <div class="text-lg font-semibold text-blue-600">{{ model.motion_score.toFixed(1) }}</div>
              </div>
              <div class="bg-blue-50 rounded-lg px-3 py-2">
                <div class="text-xs text-gray-600">真实性</div>
                <div class="text-lg font-semibold text-blue-600">{{ model.reality_score.toFixed(1) }}</div>
              </div>
            </div>

            <!-- View Detail Button -->
            <button class="w-full bg-blue-600 hover:bg-blue-700 text-white font-medium py-2 rounded-lg transition-colors">
              查看详情
            </button>
          </div>
        </div>
      </div>

      <!-- Empty State -->
      <div v-else class="text-center py-20">
        <div class="text-gray-400 mb-4">
          <svg class="mx-auto h-12 w-12" fill="none" viewBox="0 0 24 24" stroke="currentColor">
            <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M20 13V6a2 2 0 00-2-2H6a2 2 0 00-2 2v7m16 0v5a2 2 0 01-2 2H6a2 2 0 01-2-2v-5m16 0h-2.586a1 1 0 00-.707.293l-2.414 2.414a1 1 0 01-.707.293h-3.172a1 1 0 01-.707-.293l-2.414-2.414A1 1 0 006.586 13H4" />
          </svg>
        </div>
        <p class="text-gray-500">暂无模型数据</p>
      </div>
    </div>
  </div>
</template>

<script>
import { ref, onMounted } from 'vue'
import { useRouter } from 'vue-router'
import { modelAPI } from '@/api'

export default {
  name: 'Models',
  setup() {
    const router = useRouter()
    const models = ref([])
    const loading = ref(true)

    const loadModels = async () => {
      try {
        loading.value = true
        models.value = await modelAPI.getModelList()
      } catch (error) {
        console.error('加载模型列表失败:', error)
      } finally {
        loading.value = false
      }
    }

    const goToDetail = (id) => {
      router.push(`/model/${id}`)
    }

    onMounted(() => {
      loadModels()
    })

    return {
      models,
      loading,
      goToDetail
    }
  }
}
</script>
