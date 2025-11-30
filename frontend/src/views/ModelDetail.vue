<template>
  <div class="model-detail min-h-screen bg-gray-50 py-12">
    <div class="max-w-5xl mx-auto px-4 sm:px-6 lg:px-8">
      <!-- Loading -->
      <div v-if="loading" class="flex justify-center items-center py-20">
        <div class="animate-spin rounded-full h-12 w-12 border-b-2 border-blue-600"></div>
      </div>

      <!-- Content -->
      <div v-else-if="model" class="space-y-6">
        <!-- Header -->
        <div class="bg-white rounded-xl shadow-sm p-8">
          <div class="flex justify-between items-start mb-6">
            <div>
              <h1 class="text-3xl font-bold text-gray-900 mb-2">{{ model.name }}</h1>
              <p class="text-gray-600">{{ model.publisher }}</p>
            </div>
            <div class="text-right">
              <div class="text-4xl font-bold text-blue-600">{{ model.total_score.toFixed(1) }}</div>
              <div class="text-sm text-gray-500">综合评分</div>
            </div>
          </div>

          <!-- Info Grid -->
          <div class="grid grid-cols-2 md:grid-cols-4 gap-4 mb-6">
            <div>
              <div class="text-sm text-gray-500">参数规模</div>
              <div class="font-semibold text-gray-900">{{ model.parameters }}</div>
            </div>
            <div>
              <div class="text-sm text-gray-500">是否开源</div>
              <div class="font-semibold text-gray-900">{{ model.is_open_source ? '是' : '否' }}</div>
            </div>
            <div>
              <div class="text-sm text-gray-500">发布时间</div>
              <div class="font-semibold text-gray-900">{{ model.release_date }}</div>
            </div>
            <div>
              <div class="text-sm text-gray-500">测试人</div>
              <div class="font-semibold text-gray-900">{{ model.tester_name }}</div>
            </div>
          </div>

          <!-- Description -->
          <div class="border-t pt-6">
            <h3 class="font-semibold text-gray-900 mb-2">模型简介</h3>
            <p class="text-gray-600 leading-relaxed">{{ model.description || '暂无简介' }}</p>
          </div>

          <!-- Official Website -->
          <div class="mt-4">
            <a
              v-if="model.official_website"
              :href="model.official_website"
              target="_blank"
              class="inline-flex items-center text-blue-600 hover:text-blue-700 font-medium"
            >
              访问官网
              <svg class="ml-1 w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M10 6H6a2 2 0 00-2 2v10a2 2 0 002 2h10a2 2 0 002-2v-4M14 4h6m0 0v6m0-6L10 14" />
              </svg>
            </a>
          </div>
        </div>

        <!-- Score Chart -->
        <div class="bg-white rounded-xl shadow-sm p-8">
          <h2 class="text-2xl font-bold text-gray-900 mb-6">能力维度评分</h2>
          <ScoreChart
            :scores="{
              semantic: model.semantic_score,
              temporal: model.temporal_score,
              motion: model.motion_score,
              reality: model.reality_score
            }"
            :model-name="model.name"
            chart-height="450px"
          />
        </div>

        <!-- Back Button -->
        <div class="text-center">
          <button
            @click="$router.back()"
            class="inline-flex items-center px-6 py-3 bg-gray-100 hover:bg-gray-200 text-gray-700 font-medium rounded-lg transition-colors"
          >
            <svg class="mr-2 w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M10 19l-7-7m0 0l7-7m-7 7h18" />
            </svg>
            返回
          </button>
        </div>
      </div>
    </div>
  </div>
</template>

<script>
import { ref, onMounted } from 'vue'
import { useRoute } from 'vue-router'
import { modelAPI } from '@/api'
import ScoreChart from '@/components/ScoreChart.vue'

export default {
  name: 'ModelDetail',
  components: {
    ScoreChart
  },
  setup() {
    const route = useRoute()
    const model = ref(null)
    const loading = ref(true)

    const loadModel = async () => {
      try {
        loading.value = true
        const id = route.params.id
        model.value = await modelAPI.getModelDetail(id)
      } catch (error) {
        console.error('加载模型详情失败:', error)
      } finally {
        loading.value = false
      }
    }

    onMounted(() => {
      loadModel()
    })

    return {
      model,
      loading
    }
  }
}
</script>
