
<script lang="ts">
    import { onMount } from 'svelte';
    import {default as vegaEmbed} from 'vega-embed';
    import PlotWidget from './Plot.svelte';
    
    export let name;
    let selectedRun;
    
    let runToTags = {};
    onMount(async () => {
        runToTags = await fetch('./tags').then((response) => response.json());
    });

    $: tags = Object.keys(runToTags[selectedRun] || {});
    $: spec_promises = tags.map((tag) => ({tag, spec: fetch('./plot_specs?' + new URLSearchParams({run: selectedRun, tag})).then((response) => response.json())}));
</script>

<main>
	<select bind:value={selectedRun}>
		{#each Object.keys(runToTags) as run}
			<option value={run}>
				{run}
			</option>
		{/each}
	</select>
	{#if spec_promises.length == 0}
	   <div>No Tags Found for this Run</div>
	{/if}
	{#each spec_promises as spec_promise}
	       <div>Tag {spec_promise.tag}</div>
	       <PlotWidget promise={spec_promise.spec} />
	{/each}
</main>

<style>
	main {
		text-align: center;
		padding: 1em;
		max-width: 240px;
		margin: 0 auto;
	}

	@media (min-width: 640px) {
		main {
			max-width: none;
		}
	}
</style>