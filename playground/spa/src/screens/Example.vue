<template>
    <div class="example-screen">
        <loading-screen :is-loading="isLoading"></loading-screen>

        <b-container fluid>
            <b-alert :show="error" variant="danger">{{ error }}</b-alert>

            <div v-if="example" class="mb-3">
                <b-card class="mb-3" :title="example.name" :sub-title="example.description"></b-card>

                <b-card class="mb-3" title="Parameters">
                        <b-row class="my-1" v-for="param in example.params" :key="param.key">
                            <b-col sm="3">
                                <label :for="`param-${param.key}`">{{ param.name }}:</label>
                            </b-col>
                            <b-col sm="9">
                                <b-form-input v-model="param.value" :id="`param-${param.key}`"
                                              type="number"></b-form-input>
                            </b-col>
                        </b-row>
                </b-card>

                <b-button block variant="primary" @click="integrate">Integrate</b-button>

            </div>

            <oscillator-results v-if="example.key == 'harm-osc-2d'" :results="results"></oscillator-results>

        </b-container>

    </div>
</template>

<script>
    import LoadingScreen from "../components/LoadingScreen"
    import OscillatorResults from "../components/examples/OscillatorResults"

    const axios = require('axios')
    const _ = require('lodash');

    export default {
        name: "Examples",

        data() {
            return {
                isLoading: false,
                example: null,
                error: null,
                results: null
            }
        },

        created() {
            this.fetchExample()
        },

        watch: {
            '$route': 'fetchExample'
        },

        computed: {},

        methods: {
            fetchExample() {
                this.example = this.error = null

                this.isLoading = true
                axios.get('/examples/' + this.$route.params['key']).then(r => {
                    this.example = r.data
                    this.isLoading = false
                }).catch(e => {
                    this.error = e
                    this.isLoading = false
                })
            },

            integrate() {
                this.results = this.error = null
                this.isLoading = true;
                axios.post('/examples/' + this.$route.params['key'] + '/integrate', _.map(this.example.params, p => {
                    return {key: p.key, value: p.value}
                })).then(r => {
                    this.results = r.data
                    this.isLoading = false
                }).catch(e => {
                    this.error = e
                    this.isLoading = false
                })
            }
        },


        components: {OscillatorResults, LoadingScreen}
    }
</script>

<style scoped>
    .example-screen {
        margin-top: 15px;
    }
</style>