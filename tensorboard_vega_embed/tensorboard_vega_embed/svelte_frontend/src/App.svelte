
<script lang="ts">
    import { onMount } from 'svelte';
    import {default as vegaEmbed} from 'vega-embed';
    import PlotWidget from './Plot.svelte';
    
    export let name;
    let run;
    
    let runToTags = {};
    onMount(async () => {
        runToTags = await fetch('./tags').then((response) => response.json());
    });

    $: tags = Object.keys(runToTags[run] || {});
    $: spec_promises = tags.map((tag) => ({tag, spec: fetch('./plot_specs?' + new URLSearchParams({run, tag})).then((response) => response.json())}));
</script>


<main>
	<h1>Hello {name}!</h1>
	<select bind:value={run}>
		{#each Object.keys(runToTags) as run}
			<option value={run}>
				{run}
			</option>
		{/each}
	</select>
	{#each spec_promises as spec_promise }
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

	h1 {
		color: #ff3e00;
		text-transform: uppercase;
		font-size: 4em;
		font-weight: 100;
	}

	@media (min-width: 640px) {
		main {
			max-width: none;
		}
	}
</style>