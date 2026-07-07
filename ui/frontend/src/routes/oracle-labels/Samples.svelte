<script lang="ts">
	import { onMount } from 'svelte';
	import { apiFetch } from '$lib/api';

	interface SampleEntry {
		source: string;
		best_model: string;
		scores: Record<string, number>;
		judge_mode: string;
		prompt_preview: string;
	}

	let entries = $state<SampleEntry[]>([]);
	let total = $state(0);
	let loading = $state(true);

	function modelColor(model: string): string {
		const c: Record<string, string> = {
			qwen: '#00d4ff', falcon: '#ff6b6b', smollm: '#4ecdc4',
			phi2: '#ffaa00', codeqwen: '#a78bfa',
		};
		return c[model] || '#888';
	}

	function judgeModeColor(mode: string): string {
		const c: Record<string, string> = {
			judge: '#00ff88', judge_ppl_blend: '#ffaa00',
			ppl_fallback: '#ff6b6b', judge_failed: '#ff0040',
		};
		return c[mode] || '#888';
	}

	onMount(async () => {
		try {
			const res = await apiFetch('/api/oracle-labels/sample?n=20');
			if (res.ok) {
				const data = await res.json();
				entries = data.entries || [];
				total = data.total || 0;
			}
		} catch (e) {
			console.error('Failed to fetch samples:', e);
		} finally {
			loading = false;
		}
	});
</script>

{#if loading}
	<div class="text-center py-8 text-gray-500">Loading samples...</div>
{:else if entries.length === 0}
	<div class="text-center py-8 text-gray-500">No samples available</div>
{:else}
	<div class="bg-[#1a1a2e] rounded-lg border border-gray-800 p-4 space-y-4">
		<div class="flex items-center justify-between">
			<h3 class="text-sm font-semibold text-white">Sample Entries (showing {entries.length} of {total.toLocaleString()})</h3>
		</div>

		{#each entries as entry, i}
			<div class="bg-[#0a0a0f] rounded-lg border border-gray-800/50 p-3 space-y-2">
				<div class="flex items-center gap-3 text-xs">
					<span class="font-mono text-gray-500">#{i + 1}</span>
					<span class="px-1.5 py-0.5 rounded bg-gray-800 text-gray-300">{entry.source.toUpperCase()}</span>
					<span class="px-1.5 py-0.5 rounded font-mono" style="background-color: {modelColor(entry.best_model)}20; color: {modelColor(entry.best_model)}">
						best: {entry.best_model}
					</span>
					<span class="px-1.5 py-0.5 rounded text-xs" style="background-color: {judgeModeColor(entry.judge_mode)}20; color: {judgeModeColor(entry.judge_mode)}">
						{entry.judge_mode}
					</span>
				</div>
				<div class="text-xs text-gray-400 font-mono line-clamp-2">{entry.prompt_preview}...</div>
				<div class="flex gap-3 text-xs">
					{#each Object.entries(entry.scores ?? {}).sort((a, b) => b[1] - a[1]) as [model, score]}
						<span class="font-mono" style="color: {modelColor(model)}">
							{model}: {score.toFixed(2)}
						</span>
					{/each}
				</div>
			</div>
		{/each}
	</div>
{/if}
