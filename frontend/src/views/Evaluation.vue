<template>
  <div class="evaluation-page min-h-screen bg-gray-50 py-12">
    <div class="max-w-4xl mx-auto px-4 sm:px-6 lg:px-8">
      <h1 class="text-3xl font-bold text-gray-900 mb-8">模型评测试用</h1>

      <!-- Form -->
      <div v-if="!taskId" class="bg-white rounded-xl shadow-sm p-8">
        <form @submit.prevent="handleSubmit" class="space-y-6">
          <div>
            <label class="block text-sm font-medium text-gray-700 mb-2">API地址 *</label>
            <input
              v-model="form.api_url"
              type="url"
              required
              placeholder="https://api.example.com/v1"
              class="w-full px-4 py-3 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-transparent"
            />
          </div>

          <div>
            <label class="block text-sm font-medium text-gray-700 mb-2">API密钥 *</label>
            <input
              v-model="form.api_key"
              type="password"
              required
              placeholder="输入您的API密钥"
              class="w-full px-4 py-3 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-transparent"
            />
          </div>

          <div>
            <label class="block text-sm font-medium text-gray-700 mb-2">模型名称 *</label>
            <input
              v-model="form.model_name"
              type="text"
              required
              placeholder="输入模型名称"
              class="w-full px-4 py-3 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-transparent"
            />
          </div>

          <div class="grid grid-cols-2 gap-4">
            <div>
              <label class="block text-sm font-medium text-gray-700 mb-2">发布者</label>
              <input
                v-model="form.publisher"
                type="text"
                placeholder="发布机构或个人"
                class="w-full px-4 py-3 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-transparent"
              />
            </div>
            <div>
              <label class="block text-sm font-medium text-gray-700 mb-2">参数规模</label>
              <input
                v-model="form.parameters"
                type="text"
                placeholder="如：7B"
                class="w-full px-4 py-3 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-transparent"
              />
            </div>
          </div>

          <div class="grid grid-cols-2 gap-4">
            <div>
              <label class="block text-sm font-medium text-gray-700 mb-2">发布时间</label>
              <input
                v-model="form.release_date"
                type="date"
                class="w-full px-4 py-3 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-transparent"
              />
            </div>
            <div>
              <label class="block text-sm font-medium text-gray-700 mb-2">是否开源</label>
              <select
                v-model="form.is_open_source"
                class="w-full px-4 py-3 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-transparent"
              >
                <option :value="false">否</option>
                <option :value="true">是</option>
              </select>
            </div>
          </div>

          <div>
            <label class="block text-sm font-medium text-gray-700 mb-2">官网链接</label>
            <input
              v-model="form.official_website"
              type="url"
              placeholder="https://example.com"
              class="w-full px-4 py-3 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-transparent"
            />
          </div>

          <div>
            <label class="block text-sm font-medium text-gray-700 mb-2">模型简介</label>
            <textarea
              v-model="form.description"
              rows="4"
              placeholder="简要描述模型的特点和功能"
              class="w-full px-4 py-3 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-transparent"
            ></textarea>
          </div>

          <div v-if="error" class="bg-red-50 border border-red-200 text-red-600 px-4 py-3 rounded-lg text-sm">
            {{ error }}
          </div>

          <button
            type="submit"
            :disabled="submitting"
            class="w-full bg-blue-600 hover:bg-blue-700 text-white font-medium py-3 rounded-lg transition-colors disabled:opacity-50 disabled:cursor-not-allowed"
          >
            {{ submitting ? '提交中...' : '提交评测' }}
          </button>
        </form>
      </div>

      <!-- Status -->
      <div v-else class="bg-white rounded-xl shadow-sm p-8">
        <div class="text-center">
          <div v-if="evalStatus.status !== 'completed'" class="mb-6">
            <div class="animate-spin rounded-full h-16 w-16 border-b-4 border-blue-600 mx-auto mb-4"></div>
            <h2 class="text-2xl font-bold text-gray-900 mb-2">{{ evalStatus.message }}</h2>
            <p class="text-gray-600">任务ID: {{ taskId }}</p>
          </div>

          <div v-else class="mb-6">
            <div class="w-16 h-16 bg-green-100 rounded-full flex items-center justify-center mx-auto mb-4">
              <svg class="w-8 h-8 text-green-600" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M5 13l4 4L19 7"></path>
              </svg>
            </div>
            <h2 class="text-2xl font-bold text-gray-900 mb-2">评测完成！</h2>

            <!-- Scores -->
            <div v-if="evalStatus.result" class="mt-8">
              <ScoreChart
                :scores="evalStatus.result.scores"
                :model-name="evalStatus.result.model_name"
                chart-height="400px"
              />

              <div class="mt-8 space-x-4">
                <button
                  @click="$router.push(`/model/${evalStatus.result.model_id}`)"
                  class="px-6 py-3 bg-blue-600 hover:bg-blue-700 text-white font-medium rounded-lg transition-colors"
                >
                  查看详情
                </button>
                <button
                  @click="reset"
                  class="px-6 py-3 bg-gray-100 hover:bg-gray-200 text-gray-700 font-medium rounded-lg transition-colors"
                >
                  继续评测
                </button>
              </div>
            </div>
          </div>
        </div>
      </div>
    </div>
  </div>
</template>

<script>
import { ref, onUnmounted } from 'vue'
import { evalAPI } from '@/api'
import ScoreChart from '@/components/ScoreChart.vue'

export default {
  name: 'Evaluation',
  components: {
    ScoreChart
  },
  setup() {
    const form = ref({
      api_url: '',
      api_key: '',
      model_name: '',
      publisher: '',
      parameters: '',
      is_open_source: false,
      release_date: '',
      official_website: '',
      description: ''
    })
    const submitting = ref(false)
    const error = ref('')
    const taskId = ref(null)
    const evalStatus = ref({})
    let statusTimer = null

    const handleSubmit = async () => {
      submitting.value = true
      error.value = ''

      try {
        const res = await evalAPI.submitEval(form.value)
        taskId.value = res.task_id
        checkStatus()
      } catch (err) {
        error.value = err.error || '提交失败，请重试'
      } finally {
        submitting.value = false
      }
    }

    const checkStatus = async () => {
      if (!taskId.value) return

      try {
        evalStatus.value = await evalAPI.getEvalStatus(taskId.value)
        
        if (evalStatus.value.status === 'completed' || evalStatus.value.status === 'failed') {
          if (statusTimer) clearInterval(statusTimer)
        } else {
          if (!statusTimer) {
            statusTimer = setInterval(checkStatus, 3000)
          }
        }
      } catch (error) {
        console.error('查询状态失败:', error)
      }
    }

    const reset = () => {
      taskId.value = null
      evalStatus.value = {}
      form.value = {
        api_url: '',
        api_key: '',
        model_name: '',
        publisher: '',
        parameters: '',
        is_open_source: false,
        release_date: '',
        official_website: '',
        description: ''
      }
    }

    onUnmounted(() => {
      if (statusTimer) clearInterval(statusTimer)
    })

    return {
      form,
      submitting,
      error,
      taskId,
      evalStatus,
      handleSubmit,
      reset
    }
  }
}
</script>
