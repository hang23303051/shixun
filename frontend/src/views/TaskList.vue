<template>
  <div class="task-list-page min-h-screen bg-gray-50 py-12">
    <div class="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
      <div class="flex justify-between items-center mb-8">
        <h1 class="text-3xl font-bold text-gray-900">任务列表</h1>
        <button
          @click="loadTasks"
          class="px-4 py-2 bg-blue-600 hover:bg-blue-700 text-white rounded-lg text-sm font-medium transition-colors"
        >
          刷新
        </button>
      </div>

      <!-- Statistics -->
      <div v-if="statistics" class="grid grid-cols-1 md:grid-cols-5 gap-4 mb-6">
        <div class="bg-white rounded-lg shadow-sm p-4">
          <div class="text-sm text-gray-500 mb-1">总任务数</div>
          <div class="text-2xl font-bold text-gray-900">{{ statistics.total }}</div>
        </div>
        <div class="bg-white rounded-lg shadow-sm p-4">
          <div class="text-sm text-gray-500 mb-1">等待处理</div>
          <div class="text-2xl font-bold text-yellow-600">{{ statistics.pending }}</div>
        </div>
        <div class="bg-white rounded-lg shadow-sm p-4">
          <div class="text-sm text-gray-500 mb-1">处理中</div>
          <div class="text-2xl font-bold text-blue-600">{{ statistics.processing }}</div>
        </div>
        <div class="bg-white rounded-lg shadow-sm p-4">
          <div class="text-sm text-gray-500 mb-1">已完成</div>
          <div class="text-2xl font-bold text-green-600">{{ statistics.completed }}</div>
        </div>
        <div class="bg-white rounded-lg shadow-sm p-4">
          <div class="text-sm text-gray-500 mb-1">失败</div>
          <div class="text-2xl font-bold text-red-600">{{ statistics.failed }}</div>
        </div>
      </div>

      <!-- Loading -->
      <div v-if="loading" class="flex justify-center py-20">
        <div class="animate-spin rounded-full h-12 w-12 border-b-2 border-blue-600"></div>
      </div>

      <!-- Task Table -->
      <div v-else class="bg-white rounded-xl shadow-sm overflow-hidden">
        <div class="overflow-x-auto">
          <table class="min-w-full divide-y divide-gray-200">
            <thead class="bg-gray-50">
              <tr>
                <th class="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">序号</th>
                <th class="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">模型名称</th>
                <th class="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">提交用户</th>
                <th class="px-6 py-3 text-center text-xs font-medium text-gray-500 uppercase tracking-wider">状态</th>
                <th class="px-6 py-3 text-center text-xs font-medium text-gray-500 uppercase tracking-wider">总分</th>
                <th class="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">提交时间</th>
                <th class="px-6 py-3 text-center text-xs font-medium text-gray-500 uppercase tracking-wider">操作</th>
              </tr>
            </thead>
            <tbody class="bg-white divide-y divide-gray-200">
              <tr v-if="tasks.length === 0">
                <td colspan="7" class="px-6 py-8 text-center text-gray-500">
                  暂无任务记录
                </td>
              </tr>
              <tr v-for="(task, index) in tasks" :key="task.task_id" class="hover:bg-gray-50 transition-colors">
                <td class="px-6 py-4 whitespace-nowrap text-sm text-gray-900">
                  {{ index + 1 }}
                </td>
                <td class="px-6 py-4 whitespace-nowrap">
                  <div class="text-sm font-medium text-gray-900">{{ task.model_name }}</div>
                </td>
                <td class="px-6 py-4 whitespace-nowrap text-sm text-gray-600">
                  {{ task.username }}
                </td>
                <td class="px-6 py-4 whitespace-nowrap text-center">
                  <span
                    class="inline-flex items-center px-3 py-1 rounded-full text-xs font-medium"
                    :class="getStatusClass(task.status)"
                  >
                    {{ task.status_display }}
                  </span>
                </td>
                <td class="px-6 py-4 whitespace-nowrap text-center">
                  <span v-if="task.total_score" class="text-sm font-semibold text-gray-900">
                    {{ task.total_score.toFixed(1) }}
                  </span>
                  <span v-else class="text-sm text-gray-400">-</span>
                </td>
                <td class="px-6 py-4 whitespace-nowrap text-sm text-gray-600">
                  {{ formatDate(task.created_at) }}
                </td>
                <td class="px-6 py-4 whitespace-nowrap text-center text-sm">
                  <button
                    @click="$router.push(`/task/${task.task_id}`)"
                    class="text-blue-600 hover:text-blue-700 font-medium mr-3"
                  >
                    详情
                  </button>
                  <button
                    @click="confirmDelete(task)"
                    class="text-red-600 hover:text-red-700 font-medium"
                  >
                    删除
                  </button>
                </td>
              </tr>
            </tbody>
          </table>
        </div>
      </div>

      <!-- Delete Confirmation Modal -->
      <div
        v-if="deleteConfirmTask"
        class="fixed inset-0 bg-black bg-opacity-50 flex items-center justify-center z-50 p-4"
        @click.self="deleteConfirmTask = null"
      >
        <div class="bg-white rounded-xl shadow-xl max-w-md w-full p-6">
          <div class="flex items-center justify-center w-12 h-12 mx-auto bg-red-100 rounded-full mb-4">
            <svg class="w-6 h-6 text-red-600" fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M12 9v2m0 4h.01m-6.938 4h13.856c1.54 0 2.502-1.667 1.732-3L13.732 4c-.77-1.333-2.694-1.333-3.464 0L3.34 16c-.77 1.333.192 3 1.732 3z" />
            </svg>
          </div>
          <h3 class="text-lg font-bold text-gray-900 text-center mb-2">确认删除</h3>
          <p class="text-sm text-gray-600 text-center mb-6">
            删除后将不再进行评测并永久删除相关内容，是否确定删除？
          </p>
          <div class="flex space-x-3">
            <button
              @click="deleteConfirmTask = null"
              class="flex-1 px-4 py-2 bg-gray-100 hover:bg-gray-200 text-gray-700 rounded-lg font-medium transition-colors"
            >
              取消
            </button>
            <button
              @click="handleDelete"
              :disabled="deleting"
              class="flex-1 px-4 py-2 bg-red-600 hover:bg-red-700 text-white rounded-lg font-medium transition-colors disabled:opacity-50 disabled:cursor-not-allowed"
            >
              {{ deleting ? '删除中...' : '确认' }}
            </button>
          </div>
        </div>
      </div>
    </div>
  </div>
</template>

<script>
import { ref, onMounted } from 'vue'
import { taskAPI } from '@/api'

export default {
  name: 'TaskList',
  setup() {
    const tasks = ref([])
    const statistics = ref(null)
    const loading = ref(false)
    const deleteConfirmTask = ref(null)
    const deleting = ref(false)

    const loadTasks = async () => {
      loading.value = true
      try {
        tasks.value = await taskAPI.getTasks()
      } catch (error) {
        console.error('加载任务列表失败:', error)
      } finally {
        loading.value = false
      }
    }

    const loadStatistics = async () => {
      try {
        statistics.value = await taskAPI.getStatistics()
      } catch (error) {
        console.error('加载统计信息失败:', error)
      }
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

    const confirmDelete = (task) => {
      deleteConfirmTask.value = task
    }

    const handleDelete = async () => {
      if (!deleteConfirmTask.value) return
      
      deleting.value = true
      try {
        await taskAPI.deleteTask(deleteConfirmTask.value.task_id)
        // 删除成功，关闭弹窗
        deleteConfirmTask.value = null
        // 重新加载任务列表和统计信息
        await loadTasks()
        await loadStatistics()
      } catch (error) {
        console.error('删除任务失败:', error)
        alert('删除失败，请重试')
      } finally {
        deleting.value = false
      }
    }

    onMounted(() => {
      loadTasks()
      loadStatistics()
    })

    return {
      tasks,
      statistics,
      loading,
      deleteConfirmTask,
      deleting,
      loadTasks,
      getStatusClass,
      formatDate,
      confirmDelete,
      handleDelete
    }
  }
}
</script>
