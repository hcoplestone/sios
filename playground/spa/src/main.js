import Vue from 'vue'

import VueRouter from 'vue-router'

Vue.use(VueRouter)

import BootstrapVue from 'bootstrap-vue'

Vue.use(BootstrapVue)

import 'bootstrap/dist/css/bootstrap.css'
import 'bootstrap-vue/dist/bootstrap-vue.css'

Vue.config.productionTip = false
Vue.config.devtools = true

// Routes

import App from './App.vue'
import Examples from './screens/Examples.vue'
import Example from './screens/Example.vue'
import Home from './screens/Home.vue'


const routes = [
    {path: '/', component: Home},
    {path: '/examples', component: Examples},
    {path: '/examples/:key', name: 'example', component: Example}
]

const router = new VueRouter({
    routes // short for `routes: routes`
})

new Vue({
    router,
    render: h => h(App),
}).$mount('#app')
