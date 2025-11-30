<template>
  <div class="dataset-page min-h-screen bg-gray-50 py-12">
    <div class="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
      <h1 class="text-3xl font-bold text-gray-900 mb-8">测试数据集</h1>

      <div class="bg-white rounded-xl shadow-sm p-8 mb-8">
        <h2 class="text-2xl font-bold text-gray-900 mb-4">数据集介绍</h2>
        <p class="text-gray-600 leading-relaxed mb-4">
          Ref4D评测平台拥有一个专业的视频生成测试数据集，包含600条高质量测试用例。
          每个测试用例包含一个参考视频和对应的详细prompt描述，覆盖9个不同的主题领域。
        </p>
        <div class="grid md:grid-cols-2 gap-6 mt-6">
          <div class="bg-blue-50 rounded-lg p-6">
            <h3 class="font-semibold text-gray-900 mb-2">数据集规模</h3>
            <p class="text-3xl font-bold text-blue-600">600</p>
            <p class="text-sm text-gray-600">条测试用例</p>
          </div>
          <div class="bg-blue-50 rounded-lg p-6">
            <h3 class="font-semibold text-gray-900 mb-2">主题覆盖</h3>
            <p class="text-3xl font-bold text-blue-600">9</p>
            <p class="text-sm text-gray-600">个主题分类</p>
          </div>
        </div>
      </div>

      <div class="bg-white rounded-xl shadow-sm p-8">
        <h2 class="text-2xl font-bold text-gray-900 mb-6">主题分类</h2>
        <div class="grid md:grid-cols-3 gap-4">
          <div v-for="theme in themes" :key="theme.value" class="bg-gradient-to-br from-blue-50 to-white p-6 rounded-lg border border-blue-100 hover:shadow-md transition-shadow">
            <h3 class="font-semibold text-gray-900 mb-2">{{ theme.label }}</h3>
            <p class="text-sm text-gray-600">{{ theme.description }}</p>
          </div>
        </div>
      </div>

      <div class="mt-8 bg-gradient-to-r from-blue-600 to-blue-500 rounded-xl p-8 text-center text-white">
        <h3 class="text-2xl font-bold mb-2">想要了解更多？</h3>
        <p class="mb-4">查看已评测的模型，了解它们在各个维度上的表现</p>
        <button @click="$router.push('/models')" class="bg-white text-blue-600 px-6 py-3 rounded-lg font-medium hover:bg-gray-50 transition-colors">
          查看模型
        </button>
      </div>
    </div>
  </div>
</template>

<script>
import { ref } from 'vue'

export default {
  name: 'Dataset',
  setup() {
    const themes = ref([
      { value: 'animals_and_ecology', label: '动物与生态', description: '包含各种动物行为和生态场景' },
      { value: 'architecture', label: '建筑', description: '涵盖各类建筑风格和场景' },
      { value: 'commercial_marketing', label: '商业营销', description: '商业广告和营销相关内容' },
      { value: 'food', label: '食物', description: '美食制作和展示场景' },
      { value: 'industrial_activity', label: '工业活动', description: '工业生产和作业场景' },
      { value: 'landscape', label: '风景', description: '自然景观和风光' },
      { value: 'people_daily', label: '人物日常', description: '日常生活场景和人物活动' },
      { value: 'sports_competition', label: '体育竞技', description: '各类体育运动和竞技场景' },
      { value: 'transportation', label: '交通', description: '交通工具和运输场景' }
    ])

    return { themes }
  }
}
</script>
