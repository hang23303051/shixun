<template>
  <div class="ranking-page min-h-screen bg-gray-50 py-12">
    <div class="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
      <h1 class="text-3xl font-bold text-gray-900 mb-8">模型排行榜</h1>

      <!-- Dimension Tabs -->
      <div class="flex space-x-2 mb-6 overflow-x-auto">
        <button
          v-for="dim in dimensions"
          :key="dim.value"
          @click="currentDimension = dim.value"
          class="px-4 py-2 rounded-lg font-medium whitespace-nowrap transition-colors"
          :class="currentDimension === dim.value 
            ? 'bg-blue-600 text-white' 
            : 'bg-white text-gray-700 hover:bg-gray-100'"
        >
          {{ dim.label }}
        </button>
      </div>

      <!-- Loading -->
      <div v-if="loading" class="flex justify-center py-20">
        <div class="animate-spin rounded-full h-12 w-12 border-b-2 border-blue-600"></div>
      </div>

      <!-- Ranking Table -->
      <div v-else class="bg-white rounded-xl shadow-sm overflow-hidden">
        <div class="overflow-x-auto">
          <table class="min-w-full divide-y divide-gray-200">
            <thead class="bg-gray-50">
              <tr>
                <th class="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">排名</th>
                <th class="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">模型名称</th>
                <th class="px-6 py-3 text-center text-xs font-medium text-gray-500 uppercase tracking-wider">总分</th>
                <th v-if="currentDimension !== 'total'" class="px-6 py-3 text-center text-xs font-medium text-gray-500 uppercase tracking-wider">
                  {{ getCurrentDimensionLabel() }}
                </th>
                <template v-else>
                  <th class="px-6 py-3 text-center text-xs font-medium text-gray-500 uppercase tracking-wider">语义</th>
                  <th class="px-6 py-3 text-center text-xs font-medium text-gray-500 uppercase tracking-wider">时序</th>
                  <th class="px-6 py-3 text-center text-xs font-medium text-gray-500 uppercase tracking-wider">运动</th>
                  <th class="px-6 py-3 text-center text-xs font-medium text-gray-500 uppercase tracking-wider">真实性</th>
                </template>
              </tr>
            </thead>
            <tbody class="bg-white divide-y divide-gray-200">
              <tr v-for="item in rankings" :key="item.model_id" class="hover:bg-gray-50 transition-colors">
                <td class="px-6 py-4 whitespace-nowrap">
                  <div class="flex items-center">
                    <span
                      class="inline-flex items-center justify-center w-8 h-8 rounded-full font-bold text-sm"
                      :class="getRankClass(item.rank)"
                    >
                      {{ item.rank }}
                    </span>
                  </div>
                </td>
                <td class="px-6 py-4">
                  <div
                    class="text-sm font-medium text-blue-600 hover:text-blue-700 cursor-pointer"
                    @click="$router.push(`/model/${item.model_id}`)"
                  >
                    {{ item.model_name }}
                  </div>
                </td>
                <td class="px-6 py-4 text-center">
                  <span class="text-lg font-bold text-gray-900">{{ item.total_score.toFixed(1) }}</span>
                </td>
                <td v-if="currentDimension !== 'total'" class="px-6 py-4 text-center">
                  <span class="text-lg font-semibold text-blue-600">
                    {{ getDimensionScore(item).toFixed(1) }}
                  </span>
                </td>
                <template v-else>
                  <td class="px-6 py-4 text-center text-sm text-gray-600">{{ item.semantic_score.toFixed(1) }}</td>
                  <td class="px-6 py-4 text-center text-sm text-gray-600">{{ item.temporal_score.toFixed(1) }}</td>
                  <td class="px-6 py-4 text-center text-sm text-gray-600">{{ item.motion_score.toFixed(1) }}</td>
                  <td class="px-6 py-4 text-center text-sm text-gray-600">{{ item.reality_score.toFixed(1) }}</td>
                </template>
              </tr>
            </tbody>
          </table>
        </div>
      </div>
    </div>
  </div>
</template>

<script>
import { ref, watch, onMounted } from 'vue'
import { modelAPI } from '@/api'

export default {
  name: 'Ranking',
  setup() {
    const currentDimension = ref('total')
    const rankings = ref([])
    const loading = ref(false)

    const dimensions = [
      { value: 'total', label: '总榜' },
      { value: 'semantic', label: '语义一致性' },
      { value: 'temporal', label: '时序一致性' },
      { value: 'motion', label: '运动属性' },
      { value: 'reality', label: '真实性' }
    ]

    const loadRanking = async () => {
      loading.value = true
      try {
        const res = await modelAPI.getRanking(currentDimension.value)
        rankings.value = res.rankings
      } catch (error) {
        console.error('加载排行榜失败:', error)
      } finally {
        loading.value = false
      }
    }

    const getRankClass = (rank) => {
      if (rank === 1) return 'bg-yellow-400 text-white'
      if (rank === 2) return 'bg-gray-400 text-white'
      if (rank === 3) return 'bg-orange-400 text-white'
      return 'bg-gray-100 text-gray-700'
    }

    const getCurrentDimensionLabel = () => {
      return dimensions.find(d => d.value === currentDimension.value)?.label || ''
    }

    const getDimensionScore = (item) => {
      return item[`${currentDimension.value}_score`] || 0
    }

    watch(currentDimension, () => {
      loadRanking()
    })

    onMounted(() => {
      loadRanking()
    })

    return {
      currentDimension,
      dimensions,
      rankings,
      loading,
      getRankClass,
      getCurrentDimensionLabel,
      getDimensionScore
    }
  }
}
</script>
