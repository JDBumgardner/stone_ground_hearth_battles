<script>
   import {default as vegaEmbed} from 'vega-embed';
   export let promise;

   let charts = [];
   $: (async() => charts = await promise)();
   $: stepToChart = Object.fromEntries(charts.map((c) => [c.step, c.vega_spec]));

   $: steps = Object.keys(stepToChart);
   $: maxStep = Math.max(...steps);

   let sliderStep = 0;
   $: selectedStep = steps.reduce((prev, curr) => Math.abs(curr - sliderStep) < Math.abs(prev - sliderStep) ? curr : prev, maxStep || undefined) || 0;

   $: {
      if (selectedStep in stepToChart) {
      	 vegaEmbed("#chrt", JSON.parse(stepToChart[selectedStep]), {actions: false}).catch(error => console.log("VegaEmbedError: ", error));
      }
   }

</script>

<main>
<input class="stepSlider" type=range bind:value={sliderStep} min=0 max={maxStep} list="steplist">

<datalist id="steplist">
    {#each steps as step}
    <option>{step}</option>
    {/each}
</datalist>
{#await promise}
	<p>...waiting</p>
{:then number}
	<p>Chart ready. Step {selectedStep} </p>
	<div id="chrt"></div>
{:catch error}
	<p style="color: red">{error.message}</p>
{/await}
</main>

<style>
    .stepSlider {
		width: 100%;
	}
</style>