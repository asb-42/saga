<script lang="ts">
	import { onMount } from 'svelte';
	import { apiFetch } from '$lib/api';

	let images = $state<Array<{ name: string; url: string; size: number }>>([]);
	let error = $state('');
	let selected = $state<string | null>(null);

	const apiBase = `http://${typeof window !== 'undefined' ? window.location.hostname : 'localhost'}:8420`;

	onMount(async () => {
		try {
			const resp = await apiFetch('/api/alignment/tsne');
			if (resp.ok) {
				const data = await resp.json();
				images = data.images;
				if (images.length > 0) selected = images[0].url;
			} else {
				error = `HTTP ${resp.status}`;
			}
		} catch (e) {
			error = String(e);
		}
	});

	function formatSize(bytes: number): string {
		if (bytes < 1024) return `${bytes} B`;
		if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(1)} KB`;
		return `${(bytes / (1024 * 1024)).toFixed(1)} MB`;
	}

	function describeImage(name: string): string {
		if (name.includes('v1')) return 'v1 InfoNCE-only — three separate model clouds';
		if (name.includes('v2')) return 'v2 Structured — blended clouds, weaker semantic clustering';
		return name;
	}
</script>

<div class="bg-[#1a1a2e] rounded-lg border border-gray-800 p-4">
	<div class="mb-4">
		<div class="text-xs text-gray-500 mb-1">t-SNE Visualizations</div>
		<div class="text-sm text-gray-400">Shared space geometry — color by model and semantic category</div>
	</div>

	{#if error}
		<div class="text-[#ff0040] text-sm py-4">{error}</div>
	{:else if images.length === 0}
		<div class="text-gray-600 text-sm py-4">No t-SNE images found</div>
	{:else}
		<!-- Image selector tabs -->
		<div class="flex gap-2 mb-4">
			{#each images as img}
				{@const isSelected = selected === img.url}
				<button
					class="px-3 py-1.5 rounded text-xs font-mono transition-all {isSelected ? 'bg-[#00d4ff20] text-[#00d4ff]' : 'bg-black/20 text-gray-500'}"
					onclick={() => selected = img.url}
				>
					{img.name.replace('.png', '').replace('tsne_', '')}
				</button>
			{/each}
		</div>

		<!-- Selected image -->
		{#if selected}
			{@const img = images.find(i => i.url === selected)}
			{#if img}
				<div class="rounded-lg overflow-hidden border border-gray-800 bg-black/30">
					<img
						src="{apiBase}{img.url}"
						alt={img.name}
						class="w-full h-auto"
						loading="lazy"
					/>
				</div>
				<div class="mt-2 text-xs text-gray-500">
					{describeImage(img.name)} — {formatSize(img.size)}
				</div>
			{/if}
		{/if}
	{/if}
</div>
