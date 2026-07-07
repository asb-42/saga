<script lang="ts">
	import { onMount } from 'svelte';
	import { apiFetch } from '$lib/api';

	interface PairStats {
		mean: number;
		std: number;
		min: number;
		max: number;
	}

	interface SmokeTestResult {
		timestamp: string;
		num_prompts: number;
		models: string[];
		model_dims: Record<string, number>;
		projector_dim: number;
		seed: number;
		passed: boolean;
		ttest: {
			t_stat: number;
			p_value: number;
			mean_same: number;
			mean_diff: number;
			mean_delta: number;
			passed: boolean;
		};
		pair_stats: Record<string, PairStats>;
	}

	let result = $state<SmokeTestResult | null>(null);
	let history = $state<any[]>([]);
	let loading = $state(true);

	onMount(async () => {
		await Promise.all([fetchResult(), fetchHistory()]);
	});

	async function fetchResult() {
		try {
			const res = await apiFetch('/api/smoke-test');
			if (res.ok) result = await res.json();
		} catch (e) {
			console.error('Failed to fetch smoke test results:', e);
		} finally {
			loading = false;
		}
	}

	async function fetchHistory() {
		try {
			const res = await apiFetch('/api/smoke-test/history');
			if (res.ok) {
				const data = await res.json();
				history = data.history || [];
			}
		} catch (e) {
			console.error('Failed to fetch history:', e);
		}
	}

	function formatTs(ts: string): string {
		if (!ts) return '-';
		const m = ts.match(/^(\d{4})(\d{2})(\d{2})_(\d{2})(\d{2})(\d{2})$/);
		if (m) return `${m[1]}-${m[2]}-${m[3]} ${m[4]}:${m[5]}:${m[6]}`;
		return ts;
	}

	function pValueColor(p: number): string {
		if (p < 0.01) return 'text-[#00ff88]';
		if (p < 0.05) return 'text-[#ffaa00]';
		return 'text-[#ff0040]';
	}

	function modelColor(model: string): string {
		const c: Record<string, string> = {
			qwen: '#00d4ff', falcon: '#ff6b6b', smollm: '#4ecdc4',
			phi2: '#ffaa00', codeqwen: '#a78bfa',
		};
		return c[model] || '#888';
	}

	function pairKey(a: string, b: string): string {
		return `${a}_${b}`;
	}
</script>

<svelte:head>
	<title>Smoke Test — SAGA Research Lab</title>
</svelte:head>

<div class="space-y-6">
	<div class="flex items-center justify-between">
		<div>
			<h2 class="text-2xl font-bold text-white">Alignment Smoke Test</h2>
			<p class="text-gray-400">Random projector baseline — proves alignment is required</p>
		</div>
		{#if result?.timestamp}
			<span class="text-xs text-gray-500">Last run: {formatTs(result.timestamp)}</span>
		{/if}
	</div>

	{#if loading}
		<div class="text-center py-12 text-gray-500">Loading...</div>
	{:else if !result}
		<div class="bg-[#0a0a0f] rounded-lg border border-gray-800 p-8 text-center">
			<div class="text-gray-500 text-lg mb-2">No smoke test results yet</div>
			<div class="text-gray-600 text-sm">Run the Smoke Test from the Pipeline page.</div>
		</div>
	{:else}
		<!-- Verdict Banner -->
		<div class="rounded-lg border p-6 {result.passed
			? 'bg-[#ffaa00]/10 border-[#ffaa00]/30'
			: 'bg-[#00ff88]/10 border-[#00ff88]/30'}">
			<div class="flex items-center gap-4">
				<div class="text-4xl">{result.passed ? '⚠️' : '✅'}</div>
				<div>
					<div class="text-lg font-bold {result.passed ? 'text-[#ffaa00]' : 'text-[#00ff88]'}">
						{result.passed ? 'PASSED (Unexpected)' : 'FAILED — Alignment Required'}
					</div>
					<div class="text-sm text-gray-400 mt-1">
						{result.passed
							? 'Random projectors achieved alignment — test may be too easy.'
							: 'Random projectors cannot align cross-model embeddings. Trained projectors are needed.'}
					</div>
				</div>
			</div>
		</div>

		<!-- Config Summary -->
		<div class="bg-[#1a1a2e] rounded-lg border border-gray-800 p-4">
			<h3 class="text-sm font-semibold text-white mb-3">Configuration</h3>
			<div class="grid grid-cols-2 md:grid-cols-4 gap-4 text-sm">
				<div>
					<span class="text-gray-400">Prompts</span>
					<div class="text-white font-mono">{result.num_prompts}</div>
				</div>
				<div>
					<span class="text-gray-400">Models</span>
					<div class="text-white font-mono">{result.models.length}</div>
				</div>
				<div>
					<span class="text-gray-400">Projector Dim</span>
					<div class="text-white font-mono">{result.projector_dim}</div>
				</div>
				<div>
					<span class="text-gray-400">Seed</span>
					<div class="text-white font-mono">{result.seed}</div>
				</div>
			</div>
			<div class="mt-3 flex gap-2 flex-wrap">
				{#each result.models as model}
					<span class="px-2 py-0.5 rounded text-xs font-mono" style="background-color: {modelColor(model)}20; color: {modelColor(model)}">
						{model} ({result.model_dims[model]}d)
					</span>
				{/each}
			</div>
		</div>

		<!-- T-Test Results -->
		<div class="bg-[#1a1a2e] rounded-lg border border-gray-800 p-4">
			<h3 class="text-sm font-semibold text-white mb-3">Paired T-Test</h3>
			<div class="grid grid-cols-2 md:grid-cols-5 gap-4 text-sm">
				<div>
					<span class="text-gray-400">t-statistic</span>
					<div class="text-white font-mono">{result.ttest.t_stat?.toFixed(4)}</div>
				</div>
				<div>
					<span class="text-gray-400">p-value</span>
					<div class="font-mono font-bold {pValueColor(result.ttest.p_value)}">
						{result.ttest.p_value?.toFixed(6)}
					</div>
				</div>
				<div>
					<span class="text-gray-400">Mean Same-Prompt</span>
					<div class="text-[#00d4ff] font-mono">{result.ttest.mean_same?.toFixed(4)}</div>
				</div>
				<div>
					<span class="text-gray-400">Mean Diff-Prompt</span>
					<div class="text-[#ff6b6b] font-mono">{result.ttest.mean_diff?.toFixed(4)}</div>
				</div>
				<div>
					<span class="text-gray-400">Delta (same - diff)</span>
					<div class="text-[#ffaa00] font-mono">{result.ttest.mean_delta?.toFixed(4)}</div>
				</div>
			</div>
		</div>

		<!-- Cosine Similarity Pairs -->
		<div class="bg-[#1a1a2e] rounded-lg border border-gray-800 p-4">
			<h3 class="text-sm font-semibold text-white mb-3">Cosine Similarity by Model Pair</h3>
			<div class="overflow-x-auto">
				<table class="w-full text-sm">
					<thead>
						<tr class="border-b border-gray-800">
							<th class="text-left py-2 px-3 text-gray-400 font-medium">Pair</th>
							<th class="text-right py-2 px-3 text-gray-400 font-medium">Same (mean ± std)</th>
							<th class="text-right py-2 px-3 text-gray-400 font-medium">Diff (mean ± std)</th>
							<th class="text-right py-2 px-3 text-gray-400 font-medium">Same Range</th>
							<th class="text-right py-2 px-3 text-gray-400 font-medium">Diff Range</th>
						</tr>
					</thead>
					<tbody>
						{#each [...result.models].sort() as m1, i}
							{#each [...result.models].sort().slice(i + 1) as m2}
								{@const sameKey = `${m1}_${m2}_same`}
								{@const diffKey = `${m1}_${m2}_diff`}
								{@const same = result.pair_stats[sameKey]}
								{@const diff = result.pair_stats[diffKey]}
								<tr class="border-b border-gray-800/50 hover:bg-gray-800/20">
									<td class="py-2 px-3 font-mono text-xs">
										<span style="color: {modelColor(m1)}">{m1}</span>
										<span class="text-gray-600"> × </span>
										<span style="color: {modelColor(m2)}">{m2}</span>
									</td>
									<td class="py-2 px-3 text-right font-mono text-xs text-[#00d4ff]">
										{same ? `${same.mean.toFixed(4)} ± ${same.std.toFixed(4)}` : '-'}
									</td>
									<td class="py-2 px-3 text-right font-mono text-xs text-[#ff6b6b]">
										{diff ? `${diff.mean.toFixed(4)} ± ${diff.std.toFixed(4)}` : '-'}
									</td>
									<td class="py-2 px-3 text-right font-mono text-xs text-gray-500">
										{same ? `[${same.min.toFixed(3)}, ${same.max.toFixed(3)}]` : '-'}
									</td>
									<td class="py-2 px-3 text-right font-mono text-xs text-gray-500">
										{diff ? `[${diff.min.toFixed(3)}, ${diff.max.toFixed(3)}]` : '-'}
									</td>
								</tr>
							{/each}
						{/each}
					</tbody>
				</table>
			</div>
		</div>

		<!-- History -->
		{#if history.length > 0}
			<div class="bg-[#1a1a2e] rounded-lg border border-gray-800 p-4">
				<h3 class="text-sm font-semibold text-white mb-3">Run History</h3>
				<div class="space-y-1 max-h-48 overflow-y-auto">
					{#each history.slice().reverse() as run}
						<div class="flex items-center justify-between py-1 px-2 rounded hover:bg-gray-800/50 text-xs">
							<span class="font-mono text-gray-400">{formatTs(run.timestamp)}</span>
							<span class="text-gray-500">{run.models?.length || 0} models</span>
							<span class={run.passed ? 'text-[#ffaa00]' : 'text-[#00ff88]'}>
								{run.passed ? 'PASSED' : 'FAILED'}
							</span>
						</div>
					{/each}
				</div>
			</div>
		{/if}
	{/if}
</div>
