<template>
    <div v-if="results">
        <b-row>
            <b-col sm="6">
                <b-card subtitle="x = x(t)">
                    <generalised-coordinate-evolution
                            v-if="results"
                            variable="x"
                            :t="results.t_list"
                            :q="results.x_solutions"></generalised-coordinate-evolution>
                </b-card>
            </b-col>
             <b-col sm="6">
                <b-card subtitle="y = y(t)">
                    <generalised-coordinate-evolution
                            v-if="results"
                            variable="y"
                            :t="results.t_list"
                            :q="results.y_solutions"></generalised-coordinate-evolution>
                </b-card>
            </b-col>
        </b-row>
    </div>
</template>

<script>
    import GeneralisedCoordinateEvolution from "../GeneralisedCoordinateEvolution"

    export default {
        props: ['results'],

        name: "OscillatorResults",

        components: {GeneralisedCoordinateEvolution},

        computed: {}
    }
</script>

<style scoped>
.card {
    max-height: 500px;
}
</style>