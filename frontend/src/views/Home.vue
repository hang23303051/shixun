<template>
  <div class="home">
    <!-- Hero Section -->
    <section class="relative bg-gradient-to-br from-blue-50 via-white to-blue-50 py-20 overflow-hidden">
      <div class="absolute inset-0 opacity-10">
        <div class="absolute top-0 left-0 w-96 h-96 bg-blue-400 rounded-full mix-blend-multiply filter blur-3xl animate-blob"></div>
        <div class="absolute top-0 right-0 w-96 h-96 bg-blue-300 rounded-full mix-blend-multiply filter blur-3xl animate-blob animation-delay-2000"></div>
      </div>
      
      <div class="relative max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 text-center">
        <h1 class="text-5xl font-bold text-gray-900 mb-6">
          视频生成模型评测平台
        </h1>
        <p class="text-xl text-gray-600 mb-4">
          大规模在线人工智能生成视频评测体系
        </p>
        <p class="text-lg text-gray-500 mb-8">
          基于完整测试数据集，从语义一致性、时序一致性、运动属性和世界知识真实性四个维度全面评估视频生成模型
        </p>
        <button
          @click="$router.push('/models')"
          class="inline-flex items-center px-8 py-3 bg-blue-600 hover:bg-blue-700 text-white font-medium rounded-lg shadow-lg transition-all transform hover:scale-105"
        >
          查看模型
          <svg class="ml-2 w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
            <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M9 5l7 7-7 7"/>
          </svg>
        </button>
      </div>
    </section>

    <!-- Stats Section -->
    <section class="py-16 bg-white">
      <div class="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
        <div class="grid grid-cols-1 md:grid-cols-4 gap-8">
          <div v-for="stat in stats" :key="stat.label" class="text-center">
            <div class="bg-gradient-to-br from-blue-50 to-blue-100 rounded-2xl p-8 shadow-sm hover:shadow-md transition-shadow">
              <div class="inline-flex items-center justify-center w-16 h-16 mb-4">
                <img :src="stat.icon" :alt="stat.label" class="w-full h-full object-contain" v-if="false">
                <div class="w-full h-full bg-blue-500 rounded-lg opacity-80"></div>
              </div>
              <div class="text-4xl font-bold text-gray-900 mb-2">{{ stat.value }}</div>
              <div class="text-sm text-gray-600">{{ stat.label }}</div>
            </div>
          </div>
        </div>
      </div>
    </section>

    <!-- About Section -->
    <section class="py-20 bg-gray-50">
      <div class="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
        <div class="text-center mb-12">
          <h2 class="text-3xl font-bold text-gray-900 mb-4">关于 Ref4D</h2>
          <p class="text-lg text-gray-600 max-w-3xl mx-auto">
            Ref4D是一个专业的视频生成模型评测平台，为主流视频生成AI模型提供全方位的性能评估与对比分析
          </p>
        </div>

        <div class="grid md:grid-cols-2 gap-8 mb-12">
          <div class="bg-white rounded-xl p-8 shadow-sm hover:shadow-md transition-shadow">
            <h3 class="text-xl font-bold text-gray-900 mb-4">网站用途</h3>
            <p class="text-gray-600">
              展示主流视频生成模型（cogfun, cogvideo5b, jimeng, kling, sora2, videocrafter, vidu, wan21）
              的评测结果，并为用户提供新模型API评测接口，助力视频生成技术的发展与创新。
            </p>
          </div>

          <div class="bg-white rounded-xl p-8 shadow-sm hover:shadow-md transition-shadow">
            <h3 class="text-xl font-bold text-gray-900 mb-4">数据集规模</h3>
            <p class="text-gray-600">
              拥有600条高质量测试用例，覆盖9个主题领域：动物与生态、建筑、商业营销、食物、工业活动、
              风景、人物日常、体育竞技、交通运输。每个用例包含参考视频和详细的prompt描述。
            </p>
          </div>
        </div>

        <!-- Features Grid -->
        <div class="grid md:grid-cols-4 gap-6">
          <div v-for="feature in features" :key="feature.title" 
               class="bg-white rounded-lg p-6 text-center hover:shadow-md transition-shadow">
            <div class="w-12 h-12 bg-blue-100 rounded-lg mx-auto mb-4 flex items-center justify-center">
              <div class="w-6 h-6 bg-blue-500 rounded"></div>
            </div>
            <h4 class="font-semibold text-gray-900 mb-2">{{ feature.title }}</h4>
            <p class="text-sm text-gray-600">{{ feature.desc }}</p>
          </div>
        </div>
      </div>
    </section>

    <!-- CTA Section -->
    <section class="py-16 bg-gradient-to-r from-blue-600 to-blue-500">
      <div class="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 text-center">
        <h2 class="text-3xl font-bold text-white mb-4">准备好评测您的模型了吗？</h2>
        <p class="text-blue-100 mb-8 text-lg">立即注册，开始使用我们的专业评测服务</p>
        <div class="flex justify-center space-x-4">
          <button
            @click="$router.push('/register')"
            class="px-8 py-3 bg-white text-blue-600 font-medium rounded-lg shadow-lg hover:bg-gray-50 transition-colors"
          >
            立即注册
          </button>
          <button
            @click="$router.push('/dataset')"
            class="px-8 py-3 bg-blue-700 text-white font-medium rounded-lg shadow-lg hover:bg-blue-800 transition-colors"
          >
            了解数据集
          </button>
        </div>
      </div>
    </section>
  </div>
</template>

<script>
import { ref } from 'vue'

export default {
  name: 'Home',
  setup() {
    const stats = ref([
      { label: '主流模型', value: '8+', icon: '' },
      { label: '测试数据集', value: '600', icon: '' },
      { label: '评测维度', value: '4', icon: '' },
      { label: '主题分类', value: '9', icon: '' }
    ])

    const features = ref([
      { title: '语义一致性', desc: '评估生成视频与描述的语义匹配度' },
      { title: '时序一致性', desc: '检测视频帧间的连贯性与流畅度' },
      { title: '运动属性', desc: '分析物体运动的真实性与合理性' },
      { title: '世界知识', desc: '验证内容符合现实世界规律' }
    ])

    return {
      stats,
      features
    }
  }
}
</script>

<style scoped>
@keyframes blob {
  0%, 100% {
    transform: translate(0, 0) scale(1);
  }
  33% {
    transform: translate(30px, -50px) scale(1.1);
  }
  66% {
    transform: translate(-20px, 20px) scale(0.9);
  }
}

.animate-blob {
  animation: blob 7s infinite;
}

.animation-delay-2000 {
  animation-delay: 2s;
}
</style>
