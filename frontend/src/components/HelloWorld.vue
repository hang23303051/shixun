<template>
  <div class="max-w-4xl mx-auto px-6 py-12">
    <!-- 标题区域 - 使用渐变色和阴影测试 TailwindCSS -->
    <div class="text-center mb-12">
      <h1 class="text-5xl font-bold bg-gradient-to-r from-blue-600 to-purple-600 bg-clip-text text-transparent mb-4">
        {{ msg }}
      </h1>
      <div class="inline-block bg-green-100 text-green-800 px-4 py-2 rounded-full text-sm font-semibold shadow-md">
        ✅ TailwindCSS 已成功导入！
      </div>
    </div>

    <!-- 介绍卡片 -->
    <div class="bg-white rounded-xl shadow-lg p-6 mb-8 border border-gray-100 hover:shadow-2xl transition-shadow duration-300">
      <p class="text-gray-600 leading-relaxed">
        For a guide and recipes on how to configure / customize this project,<br>
        check out the
        <a href="https://cli.vuejs.org" target="_blank" rel="noopener" 
           class="text-blue-600 hover:text-blue-800 font-semibold underline decoration-2 underline-offset-2 hover:decoration-blue-800 transition-colors">
          vue-cli documentation
        </a>.
      </p>
    </div>

    <!-- Installed CLI Plugins -->
    <div class="mb-8">
      <h3 class="text-2xl font-bold text-gray-800 mb-4 flex items-center">
        <span class="w-2 h-8 bg-blue-500 rounded-full mr-3"></span>
        Installed CLI Plugins
      </h3>
      <ul class="flex flex-wrap gap-3">
        <li><a href="https://github.com/vuejs/vue-cli/tree/dev/packages/%40vue/cli-plugin-babel" target="_blank" rel="noopener"
               class="inline-block bg-blue-50 text-blue-700 px-4 py-2 rounded-lg hover:bg-blue-100 hover:scale-105 transition-all duration-200 font-medium shadow-sm">
          babel
        </a></li>
        <li><a href="https://github.com/vuejs/vue-cli/tree/dev/packages/%40vue/cli-plugin-router" target="_blank" rel="noopener"
               class="inline-block bg-blue-50 text-blue-700 px-4 py-2 rounded-lg hover:bg-blue-100 hover:scale-105 transition-all duration-200 font-medium shadow-sm">
          router
        </a></li>
        <li><a href="https://github.com/vuejs/vue-cli/tree/dev/packages/%40vue/cli-plugin-vuex" target="_blank" rel="noopener"
               class="inline-block bg-blue-50 text-blue-700 px-4 py-2 rounded-lg hover:bg-blue-100 hover:scale-105 transition-all duration-200 font-medium shadow-sm">
          vuex
        </a></li>
      </ul>
    </div>

    <!-- Essential Links -->
    <div class="mb-8">
      <h3 class="text-2xl font-bold text-gray-800 mb-4 flex items-center">
        <span class="w-2 h-8 bg-purple-500 rounded-full mr-3"></span>
        Essential Links
      </h3>
      <ul class="flex flex-wrap gap-3">
        <li><a href="https://vuejs.org" target="_blank" rel="noopener"
               class="inline-block bg-purple-50 text-purple-700 px-4 py-2 rounded-lg hover:bg-purple-100 hover:scale-105 transition-all duration-200 font-medium shadow-sm">
          Core Docs
        </a></li>
        <li><a href="https://forum.vuejs.org" target="_blank" rel="noopener"
               class="inline-block bg-purple-50 text-purple-700 px-4 py-2 rounded-lg hover:bg-purple-100 hover:scale-105 transition-all duration-200 font-medium shadow-sm">
          Forum
        </a></li>
        <li><a href="https://chat.vuejs.org" target="_blank" rel="noopener"
               class="inline-block bg-purple-50 text-purple-700 px-4 py-2 rounded-lg hover:bg-purple-100 hover:scale-105 transition-all duration-200 font-medium shadow-sm">
          Community Chat
        </a></li>
        <li><a href="https://twitter.com/vuejs" target="_blank" rel="noopener"
               class="inline-block bg-purple-50 text-purple-700 px-4 py-2 rounded-lg hover:bg-purple-100 hover:scale-105 transition-all duration-200 font-medium shadow-sm">
          Twitter
        </a></li>
        <li><a href="https://news.vuejs.org" target="_blank" rel="noopener"
               class="inline-block bg-purple-50 text-purple-700 px-4 py-2 rounded-lg hover:bg-purple-100 hover:scale-105 transition-all duration-200 font-medium shadow-sm">
          News
        </a></li>
      </ul>
    </div>

    <!-- Ecosystem -->
    <div class="mb-8">
      <h3 class="text-2xl font-bold text-gray-800 mb-4 flex items-center">
        <span class="w-2 h-8 bg-green-500 rounded-full mr-3"></span>
        Ecosystem
      </h3>
      <ul class="flex flex-wrap gap-3">
        <li><a href="https://router.vuejs.org" target="_blank" rel="noopener"
               class="inline-block bg-green-50 text-green-700 px-4 py-2 rounded-lg hover:bg-green-100 hover:scale-105 transition-all duration-200 font-medium shadow-sm">
          vue-router
        </a></li>
        <li><a href="https://vuex.vuejs.org" target="_blank" rel="noopener"
               class="inline-block bg-green-50 text-green-700 px-4 py-2 rounded-lg hover:bg-green-100 hover:scale-105 transition-all duration-200 font-medium shadow-sm">
          vuex
        </a></li>
        <li><a href="https://github.com/vuejs/vue-devtools#vue-devtools" target="_blank" rel="noopener"
               class="inline-block bg-green-50 text-green-700 px-4 py-2 rounded-lg hover:bg-green-100 hover:scale-105 transition-all duration-200 font-medium shadow-sm">
          vue-devtools
        </a></li>
        <li><a href="https://vue-loader.vuejs.org" target="_blank" rel="noopener"
               class="inline-block bg-green-50 text-green-700 px-4 py-2 rounded-lg hover:bg-green-100 hover:scale-105 transition-all duration-200 font-medium shadow-sm">
          vue-loader
        </a></li>
        <li><a href="https://github.com/vuejs/awesome-vue" target="_blank" rel="noopener"
               class="inline-block bg-green-50 text-green-700 px-4 py-2 rounded-lg hover:bg-green-100 hover:scale-105 transition-all duration-200 font-medium shadow-sm">
          awesome-vue
        </a></li>
      </ul>
    </div>
  </div>
</template>

<script>
export default {
  name: 'HelloWorld',
  props: {
    msg: String
  }
}
</script>
