<template>
    <div v-if="results" class="mb-3">
        <b-card no-body>
            <b-tabs card v-model="tabIndex">
                <b-tab title="Coordinate Evolution" active>
                    <b-row>
                        <b-col sm="6">
                            <generalised-coordinate-evolution
                                    v-if="results"
                                    variable="x"
                                    :t="results.t_list"
                                    :q="results.x_solutions"></generalised-coordinate-evolution>
                        </b-col>
                        <b-col sm="6">
                            <generalised-coordinate-evolution
                                    variable="y"
                                    :t="results.t_list"
                                    :q="results.y_solutions"></generalised-coordinate-evolution>
                        </b-col>
                    </b-row>
                </b-tab>
                <b-tab title="Trajectory">
                    <trajectory v-if="tabIndex==1"
                            :x="results.x_solutions"
                            :y="results.y_solutions"
                            x-label="x"
                            y-label="y"
                    ></trajectory>
                </b-tab>
            </b-tabs>
        </b-card>

    </div>
</template>

<script>
    import GeneralisedCoordinateEvolution from "../GeneralisedCoordinateEvolution"
    import Trajectory from "../Trajectory"

    export default {
        props: ['results'],

        name: "OscillatorResults",

        components: {GeneralisedCoordinateEvolution, Trajectory},

        data() {
            return {
                tabIndex: 0
            }
        },

        computed: {}
    }
</script>

<style scoped>
    .card {
        min-height: 500px;
    }
</style>