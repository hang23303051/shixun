import { createStore } from 'vuex'
import { accountAPI } from '@/api'

export default createStore({
  state: {
    user: null,
    isLoggedIn: false
  },
  getters: {
    user: state => state.user,
    isLoggedIn: state => state.isLoggedIn
  },
  mutations: {
    SET_USER(state, user) {
      state.user = user
      state.isLoggedIn = !!user
    },
    CLEAR_USER(state) {
      state.user = null
      state.isLoggedIn = false
    }
  },
  actions: {
    async checkLogin({ commit }) {
      try {
        const res = await accountAPI.checkLogin()
        if (res.logged_in) {
          commit('SET_USER', res.user)
          return true
        } else {
          commit('CLEAR_USER')
          return false
        }
      } catch (error) {
        commit('CLEAR_USER')
        return false
      }
    },
    async login({ commit }, credentials) {
      const res = await accountAPI.login(credentials)
      commit('SET_USER', res.user)
      return res
    },
    async logout({ commit }) {
      await accountAPI.logout()
      commit('CLEAR_USER')
    },
    async register(_, userData) {
      return await accountAPI.register(userData)
    }
  },
  modules: {
  }
})
