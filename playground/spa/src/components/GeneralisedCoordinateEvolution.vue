<script>
    import {Line} from 'vue-chartjs'

    const _ = require('lodash')

    export default {
        name: 'GeneralisedCoordinateEvolution',
        extends: Line,

        props: ['t', 'q', 'variable'],

        mounted() {
            this.renderChart(this.graphData, this.options)
        },

        data() {
            return {
                options: {
                    responsive: true,
                    maintainAspectRatio: false,
                    scales: {
                        xAxes: [{
                            type: 'linear',
                            position: 'bottom'
                        }]
                    }
                }
            }
        },

        computed: {
            dataset() {
                return _.map(this.t, (t, index) => {
                    return {x: t, y: this.q[index]}
                })
            },

            graphData() {
                return {
                    datasets: [{
                        label: 'Evolution of ' + this.variable,
                        data: this.dataset
                    }]
                }
            }
        }
    }
</script>

<style>
</style>
