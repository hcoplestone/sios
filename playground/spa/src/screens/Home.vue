<template>
    <div class="hello">
        <h1>Slimplectic Integrator on Steroids</h1>
        <p>
            Authors: Harvey Coplestone, Ellis Lukas, David Tsang <br>
            For documentation, please visit
            <a href="https://github.com/hcoplestone/sios" target="_blank" rel="noopener">Github</a>.
        </p>
    </div>
</template>

<script>
    export default {
        name: 'Home',
        props: {
            msg: String
        }
    }
</script>

<!-- Add "scoped" attribute to limit CSS to this component only -->
<style scoped>
    .hello {
        margin-top: 100px;
    }

    h3 {
        margin: 40px 0 0;
    }

    ul {
        list-style-type: none;
        padding: 0;
    }

    li {
        display: inline-block;
        margin: 0 10px;
    }

    a {
        color: #42b983;
    }
</style>
