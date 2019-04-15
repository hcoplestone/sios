<template>
    <div class="examples-screen">
        <h1>Examples</h1>

        <loading-screen :is-loading="isLoading"></loading-screen>
        <b-alert :show="error" variant="danger">{{ error }}</b-alert>

    </div>
</template>

<script>
    import LoadingScreen from "../components/LoadingScreen"
    const axios = require('axios')

    export default {
        name: "Examples",

        data() {
            return {
                isLoading: false,
                examples: null,
                error: null
            }
        },

        created() {
            this.fetchExamples()
        },

        watch: {
            '$route': 'fetchExamples'
        },

        methods: {
            fetchExamples() {
                this.examples = this.error = null

                this.isLoading = true
                axios.get('/examples').then(r => {
                    this.examples = r.data
                    this.isLoading = false
                }).catch(e => {
                    this.error = e
                    this.isLoading = false
                })
            }
        },


        components: {LoadingScreen}
    }
</script>

<style scoped>
    .examples-screen {
        padding-top: 50px;
    }
</style>