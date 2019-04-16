<script>
    import {Line} from 'vue-chartjs'

    const _ = require('lodash')

    export default {
        name: 'Trajectory',
        extends: Line,

        props: ['x', 'y', 'xLabel', 'yLabel'],

        mounted() {
            this.renderChart(this.graphData, this.options)
        },

        data() {
            return {
                options: {
                    legend: {
                        display: false
                    },
                    responsive: true,
                    maintainAspectRatio: false,
                    scales: {
                        xAxes: [{
                            type: 'linear',
                            position: 'bottom',
                            scaleLabel: {
                                display: true,
                                labelString: this.xLabel
                            }
                        }],
                        yAxes: [{
                            scaleLabel: {
                                display: true,
                                labelString: this.yLabel
                            }
                        }]
                    }
                }
            }
        },

        computed: {
            dataset() {
                return _.map(this.x, (x, index) => {
                    return {x: x, y: this.y[index]}
                })
            },

            graphData() {
                return {
                    datasets: [{
                        label: 'Trajectory',
                        data: this.dataset,
                        fill: false,
                        pointBackgroundColor: '#2b8cbf',
                        pointRadius: 1,
                        showLine: false
                    }]
                }
            }
        }
    }
</script>

<style>
</style>
