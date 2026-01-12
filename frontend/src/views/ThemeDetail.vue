<template>
  <div class="theme-detail-page min-h-screen bg-gray-50 py-8">
    <div class="max-w-6xl mx-auto px-4 sm:px-6 lg:px-8">
      <!-- 返回按钮 -->
      <button
        @click="$router.push('/dataset')"
        class="mb-6 flex items-center text-blue-600 hover:text-blue-700 transition-colors"
      >
        <svg class="w-5 h-5 mr-2" fill="none" stroke="currentColor" viewBox="0 0 24 24">
          <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M15 19l-7-7 7-7"/>
        </svg>
        返回数据集
      </button>

      <!-- 主题标题 -->
      <div class="bg-white rounded-xl shadow-sm p-6 mb-6">
        <h1 class="text-3xl font-bold text-gray-900">{{ themeLabel }}</h1>
        <p class="text-gray-600 mt-2">{{ themeDescription }}</p>
        <div class="flex items-center gap-4 mt-4 text-sm text-gray-500">
          <span>总计 {{ totalCount }} 个视频</span>
          <span>•</span>
          <span>当前: 第 {{ currentIndex + 1 }} 个</span>
        </div>
      </div>

      <!-- 加载中 -->
      <div v-if="loading" class="bg-white rounded-xl shadow-sm p-12 text-center">
        <div class="inline-block animate-spin rounded-full h-12 w-12 border-b-2 border-blue-600 mb-4"></div>
        <p class="text-gray-600">加载中...</p>
      </div>

      <!-- 内容区域 -->
      <div v-else-if="currentVideo" class="bg-white rounded-xl shadow-sm overflow-hidden">
        <!-- 视频区域 -->
        <div class="bg-black aspect-video relative">
          <video
            :key="currentVideo.video_id"
            :src="`/media/${currentVideo.video_file}`"
            controls
            class="w-full h-full"
            @error="handleVideoError"
          >
            您的浏览器不支持视频播放
          </video>
        </div>

        <!-- 视频信息 -->
        <div class="p-6">
          <div class="flex items-center justify-between mb-4">
            <h2 class="text-xl font-semibold text-gray-900">{{ currentVideo.video_id }}</h2>
            <span class="px-3 py-1 bg-blue-100 text-blue-700 text-sm rounded-full">
              {{ currentVideo.shot_type_display }}
            </span>
          </div>

          <!-- Prompt信息 -->
          <div class="bg-gray-50 rounded-lg p-4 mb-6">
            <h3 class="font-semibold text-gray-900 mb-2 flex items-center">
              <svg class="w-5 h-5 mr-2 text-blue-600" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M7 8h10M7 12h4m1 8l-4-4H5a2 2 0 01-2-2V6a2 2 0 012-2h14a2 2 0 012 2v8a2 2 0 01-2 2h-3l-4 4z"/>
              </svg>
              Prompt描述
            </h3>
            <p class="text-gray-700 leading-relaxed">{{ currentVideo.prompt }}</p>
          </div>

          <!-- 导航按钮 -->
          <div class="flex items-center justify-between gap-4">
            <button
              @click="previousVideo"
              :disabled="currentIndex ===  0"
              :class="[
                'flex items-center px-6 py-3 rounded-lg font-medium transition-all',
                currentIndex === 0
                  ? 'bg-gray-100 text-gray-400 cursor-not-allowed'
                  : 'bg-blue-600 text-white hover:bg-blue-700 transform hover:scale-105'
              ]"
            >
              <svg class="w-5 h-5 mr-2" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M15 19l-7-7 7-7"/>
              </svg>
              上一个
            </button>

            <div class="text-center flex-1">
              <div class="text-sm text-gray-500">
                {{ currentIndex + 1 }} / {{ totalCount }}
              </div>
              <div class="mt-2 bg-gray-200 rounded-full h-2 overflow-hidden">
                <div
                  class="bg-blue-600 h-full transition-all duration-300"
                  :style="{ width: `${((currentIndex + 1) / totalCount) * 100}%` }"
                ></div>
              </div>
            </div>

            <button
              @click="nextVideo"
              :disabled="currentIndex === totalCount - 1"
              :class="[
                'flex items-center px-6 py-3 rounded-lg font-medium transition-all',
                currentIndex === totalCount - 1
                  ? 'bg-gray-100 text-gray-400 cursor-not-allowed'
                  : 'bg-blue-600 text-white hover:bg-blue-700 transform hover:scale-105'
              ]"
            >
              下一个
              <svg class="w-5 h-5 ml-2" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M9 5l7 7-7 7"/>
              </svg>
            </button>
          </div>

          <!-- 快速跳转 -->
          <div class="mt-6 pt-6 border-t border-gray-200">
            <label class="block text-sm font-medium text-gray-700 mb-2">
              快速跳转到第几个视频：
            </label>
            <div class="flex gap-2">
              <input
                v-model.number="jumpToIndex"
                type="number"
                min="1"
                :max="totalCount"
                class="flex-1 border border-gray-300 rounded-lg px-4 py-2 focus:outline-none focus:ring-2 focus:ring-blue-500"
                placeholder="输入视频序号"
              />
              <button
                @click="jumpTo"
                class="px-6 py-2 bg-gray-600 text-white rounded-lg hover:bg-gray-700 transition-colors"
              >
                跳转
              </button>
            </div>
          </div>
        </div>
      </div>

      <!-- 错误提示 -->
      <div v-else class="bg-white rounded-xl shadow-sm p-12 text-center">
        <svg class="w-16 h-16 text-gray-400 mx-auto mb-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
          <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M12 8v4m0 4h.01M21 12a9 9 0 11-18 0 9 9 0 0118 0z"/>
        </svg>
        <p class="text-gray-600">暂无数据</p>
      </div>
    </div>
  </div>
</template>

<script>
import { ref, onMounted, onBeforeUnmount, computed } from 'vue'
import { useRoute } from 'vue-router'
import { dataAPI } from '@/api'

export default {
  name: 'ThemeDetail',
  setup() {
    const route = useRoute()
    
    const loading = ref(true)
    const videos = ref([])
    const currentIndex = ref(0)
    const jumpToIndex = ref(null)
    
    const themeValue = route.params.theme
    
    // 主题配置
    const themeConfig = {
      'animals_and_ecology': { label: '动物与生态', description: '包含各种动物行为和生态场景' },
      'architecture': { label: '建筑', description: '涵盖各类建筑风格和场景' },
      'commercial_marketing': { label: '商业营销', description: '商业广告和营销相关内容' },
      'food': { label: '食物', description: '美食制作和展示场景' },
      'industrial_activity': { label: '工业活动', description: '工业生产和作业场景' },
      'landscape': { label: '风景', description: '自然景观和风光' },
      'people_daily': { label: '人物日常', description: '日常生活场景和人物活动' },
      'sports_competition': { label: '体育竞技', description: '各类体育运动和竞技场景' },
      'transportation': { label: '交通', description: '交通工具和运输场景' }
    }
    
    const themeLabel = computed(() => themeConfig[themeValue]?.label || themeValue)
    const themeDescription = computed(() => themeConfig[themeValue]?.description || '')
    const currentVideo = computed(() => videos.value[currentIndex.value])
    const totalCount = computed(() => videos.value.length)
    
    // 加载该主题的所有视频
    const loadVideos = async () => {
      loading.value = true
      try {
        const data = await dataAPI.getDataByTheme(themeValue)
        videos.value = data
      } catch (error) {
        console.error('加载视频失败:', error)
        alert('加载视频失败，请稍后重试')
      } finally {
        loading.value = false
      }
    }
    
    // 上一个视频
    const previousVideo = () => {
      if (currentIndex.value > 0) {
        currentIndex.value--
      }
    }
    
    // 下一个视频
    const nextVideo = () => {
      if (currentIndex.value < totalCount.value - 1) {
        currentIndex.value++
      }
    }
    
    // 跳转到指定视频
    const jumpTo = () => {
      const index = jumpToIndex.value - 1
      if (index >= 0 && index < totalCount.value) {
        currentIndex.value = index
        jumpToIndex.value = null
      } else {
        alert(`请输入1到${totalCount.value}之间的数字`)
      }
    }
    
    // 处理视频加载错误
    const handleVideoError = (e) => {
      console.error('视频加载失败:', e)
    }
    
    // 键盘快捷键
    const handleKeydown = (e) => {
      if (e.key === 'ArrowLeft') {
        previousVideo()
      } else if (e.key === 'ArrowRight') {
        nextVideo()
      }
    }
    
    onMounted(() => {
      loadVideos()
      window.addEventListener('keydown', handleKeydown)
    })
    
    onBeforeUnmount(() => {
      window.removeEventListener('keydown', handleKeydown)
    })
    
    return {
      loading,
      videos,
      currentIndex,
      currentVideo,
      totalCount,
      themeLabel,
      themeDescription,
      jumpToIndex,
      previousVideo,
      nextVideo,
      jumpTo,
      handleVideoError
    }
  }
}
</script>

<style scoped>
/* 自定义视频控件样式 */
video::-webkit-media-controls-panel {
  background-color: rgba(0, 0, 0, 0.8);
}
</style>
