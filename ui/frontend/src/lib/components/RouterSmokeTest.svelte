<script lang="ts">
	import { onMount } from 'svelte';
	import { apiFetch } from '$lib/api';

	let data = $state<any>(null);
	let error = $state('');

	onMount(async () => {
		try {
			const resp = await apiFetch('/api/alignment/router-smoke-test');
			if (resp.ok) {
				data = await resp.json();
			} else {
				error = `HTTP ${resp.status}`;
			}
		} catch (e) {
			error = String(e);
		}
	});

	const strategyLabels: Record<string, string> = {
		avg_lr: 'Avg Embedding → LR',
		avg_mlp: 'Avg Embedding → MLP',
		concat_lr: 'Concat Per-Model → LR',
		cross_cos_lr: 'Cross-Model Cosine → LR',
	};

	function getVerdictColor(verdict: string): string {
		if (verdict === 'navigable') return '#00ff88';
		if (verdict === 'weak') return '#ffaa00';
		return '#ff0040';
	}
</script>

<div class="bg-[#1a1a2e] rounded-lg border border-gray-800 p-4">
	<div class="mb-4">
		<div class="text-xs text-gray-500 mb-1">Router Smoke Test</div>
		<div class="text-sm text-gray-400">Can a trivial router beat random chance?</div>
	</div>

	{#if error}
		<div class="text-[#ff0040] text-sm py-4">{error}</div>
	{:else if !data}
		<div class="text-gray-600 text-sm py-4">Loading...</div>
	{:else}
		<!-- Verdict banner -->
		<div class="rounded-lg border p-3 mb-4 text-center"
			style="border-color: {getVerdictColor(data.verdict)}40; background: {getVerdictColor(data.verdict)}10;">
			<div class="text-2xl font-bold font-mono" style="color: {getVerdictColor(data.verdict)}">
				{(data.best_accuracy * 100).toFixed(1)}%
			</div>
			<div class="text-xs" style="color: {getVerdictColor(data.verdict)}">
				{#if data.verdict === 'navigable'}
					✅ SHARED SPACE IS NAVIGABLE
				{:else if data.verdict === 'weak'}
					⚠️ Weak signal
				{:else}
					❌ Not navigable
				{/if}
			</div>
			<div class="text-xs text-gray-500 mt-1">
				Best: {strategyLabels[data.best_strategy] || data.best_strategy}
			</div>
		</div>

		<!-- Random chance baseline -->
		<div class="flex items-center gap-2 mb-3 text-xs">
			<span class="text-gray-500">Random chance:</span>
			<span class="text-white font-mono">{(data.random_chance * 100).toFixed(1)}%</span>
		</div>

		<!-- Strategy bars -->
		<div class="space-y-2">
			{#each Object.entries(data.strategies) as [key, strat]}
				{@const s = strat as { accuracy: number; beats_random: boolean }}
				<div class="flex items-center gap-3">
					<span class="text-xs text-gray-500 w-44 truncate">{strategyLabels[key] || key}</span>
					<div class="flex-1 h-5 bg-black/30 rounded overflow-hidden relative">
						<!-- Random chance baseline line -->
						<div class="absolute top-0 bottom-0 w-px bg-gray-600"
							style="left: {(data.random_chance * 100 / (s.accuracy * 100 + 5)) * 100}%"></div>
						<!-- Accuracy bar -->
						<div
							class="h-full rounded transition-all duration-300"
							class:bg-[#00ff88]={s.beats_random}
							class:bg-[#ff0040]={!s.beats_random}
							style="width: {(s.accuracy / (s.accuracy + 0.05)) * 100}%; opacity: 0.8;"
						></div>
						<span class="absolute inset-0 flex items-center px-2 text-xs font-mono text-white">
							{(s.accuracy * 100).toFixed(1)}% {s.beats_random ? '✅' : '❌'}
						</span>
					</div>
				</div>
			{/each}
		</div>

		<!-- Oracle distribution -->
		{#if data.oracle_distribution}
			<div class="mt-4 pt-3 border-t border-gray-800">
				<div class="text-xs text-gray-500 mb-2">Oracle Label Distribution (train)</div>
				<div class="flex gap-4 text-xs">
					{#each Object.entries(data.oracle_distribution) as [mid, dist]}
						{@const d = dist as { train: number; val: number }}
						<span class="text-gray-400">
							{mid}: <span class="text-white font-mono">{d.train}</span>
							<span class="text-gray-600">({(d.train / data.n_train * 100).toFixed(0)}%)</span>
						</span>
					{/each}
				</div>
			</div>
		{/if}

		<div class="mt-3 text-xs text-gray-600">
			{data.n_train} train / {data.n_val} val prompts • checkpoint step {data.checkpoint_step}
		</div>
	{/if}
</div>
