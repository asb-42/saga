<script lang="ts">
	import { onMount } from 'svelte';
	import { apiFetch } from '$lib/api';

	let data = $state<any>(null);
	let error = $state('');

	onMount(async () => {
		try {
			const resp = await apiFetch('/api/alignment/lambda-ablation');
			if (resp.ok) {
				data = await resp.json();
			} else {
				error = `HTTP ${resp.status}`;
			}
		} catch (e) {
			error = String(e);
		}
	});

	const metrics = [
		{ key: 'retrieval', label: 'Retrieval Accuracy', color: '#00d4ff', format: (v: number) => `${(v * 100).toFixed(1)}%` },
		{ key: 'spearman', label: 'Spearman ρ', color: '#00ff88', format: (v: number) => v.toFixed(3) },
		{ key: 'anti_collapse', label: 'Anti-Collapse Ratio', color: '#ffaa00', format: (v: number) => `${v.toFixed(2)}x` },
	];

	function barWidth(value: number, maxVal: number): number {
		return Math.max(2, (value / maxVal) * 100);
	}
</script>

<div class="bg-[#1a1a2e] rounded-lg border border-gray-800 p-4">
	<div class="flex items-center justify-between mb-4">
		<div>
			<div class="text-xs text-gray-500 mb-1">λ Ablation Study</div>
			<div class="text-sm text-gray-400">Structure loss weight vs alignment quality</div>
		</div>
		{#if data}
			<span class="text-xs text-gray-600">{data.lambdas.length} λ values tested</span>
		{/if}
	</div>

	{#if error}
		<div class="text-[#ff0040] text-sm py-4">{error}</div>
	{:else if !data}
		<div class="text-gray-600 text-sm py-4">Loading...</div>
	{:else}
		{#each metrics as m}
			{@const values = data.lambdas.map((l: number) => data.results[String(l)][m.key])}
			{@const maxVal = Math.max(...values) * 1.1}
			<div class="mb-4">
				<div class="text-xs text-gray-500 mb-2">{m.label}</div>
				<div class="space-y-1.5">
					{#each data.lambdas as lam, i}
						<div class="flex items-center gap-3">
							<span class="text-xs text-gray-500 w-12 font-mono text-right">λ={lam}</span>
							<div class="flex-1 h-5 bg-black/30 rounded overflow-hidden relative">
								<div
									class="h-full rounded transition-all duration-300"
									style="width: {barWidth(values[i], maxVal)}%; background: {m.color}; opacity: 0.8;"
								></div>
								<span class="absolute inset-0 flex items-center px-2 text-xs font-mono text-white">
									{m.format(values[i])}
								</span>
							</div>
						</div>
					{/each}
				</div>
			</div>
		{/each}

		<!-- Cross-model cosine -->
		<div class="mt-4 pt-3 border-t border-gray-800">
			<div class="text-xs text-gray-500 mb-2">Cross-Model Cosine Similarity</div>
			<div class="grid grid-cols-2 gap-3 text-xs">
				<div>
					<span class="text-gray-500">Same prompt:</span>
					{#each data.lambdas as lam}
						{@const v = data.results[String(lam)].same_cos}
						<div class="font-mono text-[#00ff88]">λ={lam}: {v.toFixed(4)}</div>
					{/each}
				</div>
				<div>
					<span class="text-gray-500">Diff prompt:</span>
					{#each data.lambdas as lam}
						{@const v = data.results[String(lam)].diff_cos}
						<div class="font-mono text-[#ff0040]">λ={lam}: {v.toFixed(4)}</div>
					{/each}
				</div>
			</div>
		</div>
	{/if}
</div>
