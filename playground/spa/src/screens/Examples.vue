<template>
    <div class="examples-screen">
        <b-jumbotron header="Examples" lead="Click on an example and have a play!">
        </b-jumbotron>

        <loading-screen :is-loading="isLoading"></loading-screen>

        <b-container fluid>
            <b-alert :show="error" variant="danger">{{ error }}</b-alert>
            <b-table v-if="examples" striped bordered hover
                     :items="examples"
                     :fields="['name', 'description']"
                     selectable
                     @row-selected="selectExample"></b-table>
        </b-container>

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
            },

            selectExample(example) {
                this.$router.push({name: 'example', params: {key: example[0].key}})
            }
        },


        components: {LoadingScreen}
    }
</script>

<style scoped>
    .examples-screen {
    }
</style>