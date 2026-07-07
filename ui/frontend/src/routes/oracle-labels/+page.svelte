<script lang="ts">
	import { onMount } from 'svelte';
	import { apiFetch } from '$lib/api';
	import Samples from './Samples.svelte';

	interface OracleStats {
		total: number;
		models: string[];
		win_rates: Record<string, number>;
		avg_scores: Record<string, number>;
		judge_modes: Record<string, number>;
		sources: Record<string, number>;
		per_source_win_rates: Record<string, Record<string, number>>;
		score_distribution: Record<string, number>;
	}

	interface OracleResult {
		filename: string;
		oracle_mode: string;
		stats: OracleStats;
	}

	let result = $state<OracleResult | null>(null);
	let history = $state<any[]>([]);
	let loading = $state(true);
	let activeTab = $state<'overview' | 'per-source' | 'samples'>('overview');

	onMount(async () => {
		await Promise.all([fetchResult(), fetchHistory()]);
	});

	async function fetchResult() {
		try {
			const res = await apiFetch('/api/oracle-labels');
			if (res.ok) result = await res.json();
		} catch (e) {
			console.error('Failed to fetch oracle labels:', e);
		} finally {
			loading = false;
		}
	}

	async function fetchHistory() {
		try {
			const res = await apiFetch('/api/oracle-labels/history');
			if (res.ok) {
				const data = await res.json();
				history = data.history || [];
			}
		} catch (e) {
			console.error('Failed to fetch oracle labels history:', e);
		}
	}

	function modelColor(model: string): string {
		const c: Record<string, string> = {
			qwen: '#00d4ff', falcon: '#ff6b6b', smollm: '#4ecdc4',
			phi2: '#ffaa00', codeqwen: '#a78bfa',
		};
		return c[model] || '#888';
	}

	function judgeModeColor(mode: string): string {
		const c: Record<string, string> = {
			judge: '#00ff88',
			judge_ppl_blend: '#ffaa00',
			ppl_fallback: '#ff6b6b',
			exact_match: '#00d4ff',
			judge_failed: '#ff0040',
		};
		return c[mode] || '#888';
	}

	function judgeModeLabel(mode: string): string {
		const labels: Record<string, string> = {
			judge: 'Judge Only',
			judge_ppl_blend: 'Judge + PPL Blend',
			ppl_fallback: 'PPL Fallback',
			exact_match: 'Exact Match',
			judge_failed: 'Judge Failed',
		};
		return labels[mode] || mode;
	}

	function sourceIcon(src: string): string {
		return src === 'mmlu' ? '📚' : src === 'gsm8k' ? '🧮' : '❓';
	}
</script>

<svelte:head>
	<title>Oracle Labels — SAGA Research Lab</title>
</svelte:head>

<div class="space-y-6">
	<div class="flex items-center justify-between">
		<div>
			<h2 class="text-2xl font-bold text-white">Oracle Label Generation</h2>
			<p class="text-gray-400">Model ranking quality — judge consensus, win rates, source balance</p>
		</div>
		{#if result?.filename}
			<span class="text-xs text-gray-500 font-mono">{result.filename}</span>
		{/if}
	</div>

	{#if loading}
		<div class="text-center py-12 text-gray-500">Loading...</div>
	{:else if !result}
		<div class="bg-[#0a0a0f] rounded-lg border border-gray-800 p-8 text-center">
			<div class="text-gray-500 text-lg mb-2">No oracle label results yet</div>
			<div class="text-gray-600 text-sm">Run Oracle Label Generation from the Pipeline page.</div>
		</div>
	{:else}
		<!-- Summary Cards -->
		<div class="grid grid-cols-2 md:grid-cols-4 gap-4">
			<div class="bg-[#1a1a2e] rounded-lg border border-gray-800 p-4">
				<div class="text-xs text-gray-400 mb-1">Total Prompts</div>
				<div class="text-2xl font-bold text-white font-mono">{result.stats.total.toLocaleString()}</div>
			</div>
			<div class="bg-[#1a1a2e] rounded-lg border border-gray-800 p-4">
				<div class="text-xs text-gray-400 mb-1">Oracle Mode</div>
				<div class="text-sm font-mono font-bold" style="color: {judgeModeColor(result.oracle_mode)}">
					{judgeModeLabel(result.oracle_mode)}
				</div>
			</div>
			<div class="bg-[#1a1a2e] rounded-lg border border-gray-800 p-4">
				<div class="text-xs text-gray-400 mb-1">Models</div>
				<div class="flex gap-1 flex-wrap mt-1">
					{#each result.stats.models as m}
						<span class="px-1.5 py-0.5 rounded text-xs font-mono" style="background-color: {modelColor(m)}20; color: {modelColor(m)}">
							{m}
						</span>
					{/each}
				</div>
			</div>
			<div class="bg-[#1a1a2e] rounded-lg border border-gray-800 p-4">
				<div class="text-xs text-gray-400 mb-1">Sources</div>
				<div class="flex gap-2 mt-1">
					{#each Object.entries(result.stats.sources) as [src, count]}
						<span class="text-sm text-gray-300">
							{sourceIcon(src)} {src} ({count.toLocaleString()})
						</span>
					{/each}
				</div>
			</div>
		</div>

		<!-- Tabs -->
		<div class="flex gap-2 border-b border-gray-800 pb-2">
			{#each [
				{ key: 'overview', label: 'Overview' },
				{ key: 'per-source', label: 'Per-Source Breakdown' },
				{ key: 'samples', label: 'Sample Entries' },
			] as tab}
				<button
					class="px-3 py-1.5 rounded text-sm font-medium transition-all {activeTab === tab.key
						? 'bg-[#00d4ff]/20 text-[#00d4ff] border border-[#00d4ff]/30'
						: 'text-gray-400 hover:text-white hover:bg-gray-800/50 border border-transparent'}"
					onclick={() => activeTab = tab.key as typeof activeTab}
				>
					{tab.label}
				</button>
			{/each}
		</div>

		{#if activeTab === 'overview'}
			<!-- Model Win Rates -->
			<div class="bg-[#1a1a2e] rounded-lg border border-gray-800 p-4">
				<h3 class="text-sm font-semibold text-white mb-3">Model Win Rates</h3>
				<p class="text-xs text-gray-500 mb-4">How often each model is ranked best by the judge</p>
				<div class="space-y-3">
					{#each Object.entries(result.stats.win_rates).sort((a, b) => b[1] - a[1]) as [model, rate]}
						{@const pct = (rate * 100).toFixed(1)}
						<div class="flex items-center gap-3">
							<span class="w-20 text-sm font-mono font-medium" style="color: {modelColor(model)}">{model}</span>
							<div class="flex-1 h-6 bg-gray-800 rounded-full overflow-hidden">
								<div
									class="h-full rounded-full transition-all duration-500"
									style="width: {pct}%; background-color: {modelColor(model)}"
								></div>
							</div>
							<span class="w-14 text-right text-sm font-mono text-gray-300">{pct}%</span>
						</div>
					{/each}
				</div>
			</div>

			<!-- Judge Mode Distribution + Score Distribution -->
			<div class="grid grid-cols-1 md:grid-cols-2 gap-4">
				<!-- Judge Modes -->
				<div class="bg-[#1a1a2e] rounded-lg border border-gray-800 p-4">
					<h3 class="text-sm font-semibold text-white mb-3">Judge Mode Distribution</h3>
					<div class="space-y-2">
						{#each Object.entries(result.stats.judge_modes).sort((a, b) => b[1] - a[1]) as [mode, count]}
							{@const pct = ((count / result.stats.total) * 100).toFixed(1)}
							<div class="flex items-center gap-3">
								<div class="flex items-center gap-2 w-36">
									<div class="w-2 h-2 rounded-full" style="background-color: {judgeModeColor(mode)}"></div>
									<span class="text-xs text-gray-300 truncate">{judgeModeLabel(mode)}</span>
								</div>
								<div class="flex-1 h-4 bg-gray-800 rounded-full overflow-hidden">
									<div
										class="h-full rounded-full"
										style="width: {pct}%; background-color: {judgeModeColor(mode)}"
									></div>
								</div>
								<span class="w-14 text-right text-xs font-mono text-gray-400">{count.toLocaleString()}</span>
							</div>
						{/each}
					</div>
				</div>

				<!-- Score Distribution -->
				<div class="bg-[#1a1a2e] rounded-lg border border-gray-800 p-4">
					<h3 class="text-sm font-semibold text-white mb-3">Score Distribution</h3>
					<p class="text-xs text-gray-500 mb-3">Normalized scores across all model-prompt pairs</p>
					<div class="space-y-2">
						{#each Object.entries(result.stats.score_distribution) as [bucket, count]}
							{@const totalScores = Object.values(result.stats.score_distribution).reduce((a, b) => a + b, 0)}
							{@const pct = totalScores > 0 ? ((count / totalScores) * 100).toFixed(1) : '0'}
							<div class="flex items-center gap-3">
								<span class="w-20 text-xs font-mono text-gray-400">{bucket}</span>
								<div class="flex-1 h-4 bg-gray-800 rounded-full overflow-hidden">
									<div
										class="h-full rounded-full bg-[#00d4ff]"
										style="width: {pct}%"
									></div>
								</div>
								<span class="w-14 text-right text-xs font-mono text-gray-400">{count.toLocaleString()}</span>
							</div>
						{/each}
					</div>
				</div>
			</div>

			<!-- Average Scores Table -->
			<div class="bg-[#1a1a2e] rounded-lg border border-gray-800 p-4">
				<h3 class="text-sm font-semibold text-white mb-3">Average Scores by Model</h3>
				<div class="overflow-x-auto">
					<table class="w-full text-sm">
						<thead>
							<tr class="border-b border-gray-800">
								<th class="text-left py-2 px-3 text-gray-400 font-medium">Model</th>
								<th class="text-right py-2 px-3 text-gray-400 font-medium">Avg Score</th>
								<th class="text-right py-2 px-3 text-gray-400 font-medium">Win Rate</th>
								<th class="text-left py-2 px-3 text-gray-400 font-medium">Score Bar</th>
							</tr>
						</thead>
						<tbody>
							{#each Object.entries(result.stats.avg_scores).sort((a, b) => b[1] - a[1]) as [model, avg]}
								{@const winRate = result.stats.win_rates[model] ?? 0}
								<tr class="border-b border-gray-800/50 hover:bg-gray-800/20">
									<td class="py-2 px-3 font-mono font-medium" style="color: {modelColor(model)}">{model}</td>
									<td class="py-2 px-3 text-right font-mono text-white">{avg.toFixed(4)}</td>
									<td class="py-2 px-3 text-right font-mono text-gray-300">{(winRate * 100).toFixed(1)}%</td>
									<td class="py-2 px-3">
										<div class="w-full h-3 bg-gray-800 rounded-full overflow-hidden">
											<div
												class="h-full rounded-full"
												style="width: {(avg * 100).toFixed(1)}%; background-color: {modelColor(model)}"
											></div>
										</div>
									</td>
								</tr>
							{/each}
						</tbody>
					</table>
				</div>
			</div>
		{:else if activeTab === 'per-source'}
			<!-- Per-Source Breakdown -->
			{#each Object.entries(result.stats.per_source_win_rates) as [source, winRates]}
				<div class="bg-[#1a1a2e] rounded-lg border border-gray-800 p-4">
					<h3 class="text-sm font-semibold text-white mb-3">
						{sourceIcon(source)} {source.toUpperCase()} Win Rates
						<span class="text-xs text-gray-500 font-normal ml-2">
							({result.stats.sources[source]?.toLocaleString() ?? 0} prompts)
						</span>
					</h3>
					<div class="space-y-3">
						{#each Object.entries(winRates).sort((a, b) => b[1] - a[1]) as [model, rate]}
							{@const pct = (rate * 100).toFixed(1)}
							<div class="flex items-center gap-3">
								<span class="w-20 text-sm font-mono font-medium" style="color: {modelColor(model)}">{model}</span>
								<div class="flex-1 h-6 bg-gray-800 rounded-full overflow-hidden">
									<div
										class="h-full rounded-full transition-all duration-500"
										style="width: {pct}%; background-color: {modelColor(model)}"
									></div>
								</div>
								<span class="w-14 text-right text-sm font-mono text-gray-300">{pct}%</span>
							</div>
						{/each}
					</div>
				</div>
			{/each}
		{:else}
			<!-- Samples -->
			<Samples />
		{/if}

		<!-- History -->
		{#if history.length > 0}
			<div class="bg-[#1a1a2e] rounded-lg border border-gray-800 p-4">
				<h3 class="text-sm font-semibold text-white mb-3">Run History</h3>
				<div class="space-y-1 max-h-48 overflow-y-auto">
					{#each history.slice().reverse() as run}
						<div class="flex items-center justify-between py-1 px-2 rounded hover:bg-gray-800/50 text-xs">
							<span class="font-mono text-gray-400">{run.filename || '-'}</span>
							<span class="text-gray-500">{run.total_entries?.toLocaleString() ?? '?'} prompts</span>
							<span class="text-[#00d4ff] font-mono text-xs">{run.oracle_mode || '?'}</span>
						</div>
					{/each}
				</div>
			</div>
		{/if}
	{/if}
</div>
