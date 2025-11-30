<template>
  <div class="score-chart">
    <!-- Tabs -->
    <div class="flex border-b border-gray-200 mb-6">
      <button
        v-for="tab in tabs"
        :key="tab.type"
        @click="currentTab = tab.type"
        class="px-4 py-2 text-sm font-medium transition-colors"
        :class="currentTab === tab.type 
          ? 'text-blue-600 border-b-2 border-blue-600' 
          : 'text-gray-600 hover:text-blue-600'"
      >
        {{ tab.label }}
      </button>
    </div>

    <!-- Chart Container -->
    <div class="chart-container" :style="{ height: chartHeight }">
      <v-chart :option="chartOption" autoresize />
    </div>
  </div>
</template>

<script>
import { ref, computed } from 'vue'
import VChart from 'vue-echarts'
import { use } from 'echarts/core'
import { CanvasRenderer } from 'echarts/renderers'
import { RadarChart, BarChart } from 'echarts/charts'
import {
  TitleComponent,
  TooltipComponent,
  LegendComponent,
  GridComponent
} from 'echarts/components'

use([
  CanvasRenderer,
  RadarChart,
  BarChart,
  TitleComponent,
  TooltipComponent,
  LegendComponent,
  GridComponent
])

export default {
  name: 'ScoreChart',
  components: {
    VChart
  },
  props: {
    scores: {
      type: Object,
      required: true
    },
    modelName: {
      type: String,
      default: ''
    },
    chartHeight: {
      type: String,
      default: '400px'
    }
  },
  setup(props) {
    const currentTab = ref('radar')
    const tabs = [
      { type: 'radar', label: '雷达图' },
      { type: 'bar', label: '柱状图' }
    ]

    const dimensions = [
      { key: 'semantic', label: '语义一致性', max: 100 },
      { key: 'temporal', label: '时序一致性', max: 100 },
      { key: 'motion', label: '运动属性', max: 100 },
      { key: 'reality', label: '真实性', max: 100 }
    ]

    const chartOption = computed(() => {
      const scoreData = dimensions.map(d => props.scores[d.key] || 0)

      if (currentTab.value === 'radar') {
        return {
          tooltip: {
            trigger: 'item'
          },
          radar: {
            indicator: dimensions.map(d => ({
              name: d.label,
              max: d.max
            })),
            radius: '65%',
            axisName: {
              color: '#6b7280',
              fontSize: 12
            },
            splitArea: {
              areaStyle: {
                color: ['rgba(59, 130, 246, 0.05)', 'rgba(59, 130, 246, 0.1)']
              }
            },
            splitLine: {
              lineStyle: {
                color: 'rgba(59, 130, 246, 0.2)'
              }
            }
          },
          series: [
            {
              type: 'radar',
              data: [
                {
                  value: scoreData,
                  name: props.modelName || '评分',
                  areaStyle: {
                    color: 'rgba(59, 130, 246, 0.2)'
                  },
                  lineStyle: {
                    color: '#3b82f6',
                    width: 2
                  },
                  itemStyle: {
                    color: '#3b82f6'
                  }
                }
              ]
            }
          ]
        }
      } else {
        return {
          tooltip: {
            trigger: 'axis',
            axisPointer: {
              type: 'shadow'
            }
          },
          grid: {
            left: '3%',
            right: '4%',
            bottom: '3%',
            containLabel: true
          },
          xAxis: {
            type: 'category',
            data: dimensions.map(d => d.label),
            axisLabel: {
              color: '#6b7280',
              fontSize: 11,
              rotate: 15
            }
          },
          yAxis: {
            type: 'value',
            max: 100,
            axisLabel: {
              color: '#6b7280'
            },
            splitLine: {
              lineStyle: {
                color: 'rgba(59, 130, 246, 0.1)'
              }
            }
          },
          series: [
            {
              name: '评分',
              type: 'bar',
              data: scoreData,
              itemStyle: {
                color: '#3b82f6',
                borderRadius: [4, 4, 0, 0]
              },
              barWidth: '40%',
              label: {
                show: true,
                position: 'top',
                color: '#3b82f6',
                fontWeight: 'bold'
              }
            }
          ]
        }
      }
    })

    return {
      currentTab,
      tabs,
      chartOption
    }
  }
}
</script>

<style scoped>
.chart-container {
  width: 100%;
}
</style>
