<script lang="ts">
	import { onMount } from 'svelte';
	import { apiFetch } from '$lib/api';

	interface ValidationStats {
		total_entries: number;
		model_ids: string[];
		per_model: Record<string, {
			win_rate: number;
			avg_score: number;
			std_score: number;
			min_score: number;
			max_score: number;
			median_score: number;
			wins: number;
			total: number;
		}>;
		per_source: Record<string, {
			count: number;
			win_rates: Record<string, number>;
			avg_scores: Record<string, number>;
		}>;
		source_counts: Record<string, number>;
		overall_distribution: Record<string, number>;
		target_distribution: Record<string, number>;
		kl_divergence: number;
		score_histogram: Record<string, number>;
	}

	let stats = $state<ValidationStats | null>(null);
	let loading = $state(true);
	let activeTab = $state<'overview' | 'per-source' | 'histogram'>('overview');

	onMount(async () => {
		try {
			const res = await apiFetch('/api/oracle-validation');
			if (res.ok) {
				const data = await res.json();
				stats = data.stats;
			}
		} catch (e) {
			console.error('Failed to fetch oracle validation:', e);
		} finally {
			loading = false;
		}
	});

	function modelColor(model: string): string {
		const c: Record<string, string> = {
			'qwen-1.5b': '#00d4ff', 'smollm-135m': '#4ecdc4',
			'phi-2': '#ffaa00', 'codeqwen-1.5b': '#a78bfa',
			qwen: '#00d4ff', smollm: '#4ecdc4', phi2: '#ffaa00', codeqwen: '#a78bfa',
		};
		return c[model] || '#888';
	}

	function klColor(kl: number): string {
		if (kl < 0.1) return '#00ff88';
		if (kl < 0.5) return '#ffaa00';
		return '#ff6b6b';
	}

	function sourceIcon(src: string): string {
		const icons: Record<string, string> = {
			arc_easy: '🔬', hellaswag: '📖', winogrande: '🧠',
			boolq: '✅', humaneval: '💻',
		};
		return icons[src] || '❓';
	}

	let totalScores = $derived(
		stats?.score_histogram ? Object.values(stats.score_histogram).reduce((a, b) => a + b, 0) : 0
	);
</script>

<svelte:head>
	<title>Oracle Validation — SAGA Research Lab</title>
</svelte:head>

<div class="space-y-6">
	<div class="flex items-center justify-between">
		<div>
			<h2 class="text-2xl font-bold text-white">Oracle Label Validation</h2>
			<p class="text-gray-400">Distribution analysis — target alignment, win rates, KL divergence</p>
		</div>
	</div>

	{#if loading}
		<div class="text-center py-12 text-gray-500">Loading...</div>
	{:else if !stats}
		<div class="bg-[#0a0a0f] rounded-lg border border-gray-800 p-8 text-center">
			<div class="text-gray-500 text-lg mb-2">No oracle label data available</div>
			<div class="text-gray-600 text-sm">Run Oracle Label Generation from the Pipeline page first.</div>
		</div>
	{:else}
		<!-- Summary Cards -->
		<div class="grid grid-cols-2 md:grid-cols-4 gap-4">
			<div class="bg-[#1a1a2e] rounded-lg border border-gray-800 p-4">
				<div class="text-xs text-gray-400 mb-1">Total Prompts</div>
				<div class="text-2xl font-bold text-white font-mono">{stats.total_entries.toLocaleString()}</div>
			</div>
			<div class="bg-[#1a1a2e] rounded-lg border border-gray-800 p-4">
				<div class="text-xs text-gray-400 mb-1">KL Divergence</div>
				<div class="text-2xl font-bold font-mono" style="color: {klColor(stats.kl_divergence)}">
					{stats.kl_divergence.toFixed(4)} <span class="text-xs font-normal text-gray-500">nats</span>
				</div>
				<div class="text-[10px] text-gray-600 mt-1 font-mono">
					Σ p(m) · ln(p(m) / q(m)) — sum over models, natural log
				</div>
				<div class="text-[10px] text-gray-500 mt-0.5">
					p = actual win rate, q = target. 0 = perfect match.
				</div>
			</div>
			<div class="bg-[#1a1a2e] rounded-lg border border-gray-800 p-4">
				<div class="text-xs text-gray-400 mb-1">Models</div>
				<div class="flex gap-1 flex-wrap mt-1">
					{#each stats.model_ids as m}
						<span class="px-1.5 py-0.5 rounded text-xs font-mono" style="background-color: {modelColor(m)}20; color: {modelColor(m)}">
							{m}
						</span>
					{/each}
				</div>
			</div>
			<div class="bg-[#1a1a2e] rounded-lg border border-gray-800 p-4">
				<div class="text-xs text-gray-400 mb-1">Sources</div>
				<div class="flex gap-2 mt-1">
					{#each Object.entries(stats.source_counts) as [src, count]}
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
				{ key: 'overview', label: 'Distribution' },
				{ key: 'per-source', label: 'Per-Source' },
				{ key: 'histogram', label: 'Score Histogram' },
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
			<!-- Distribution Comparison -->
			<div class="bg-[#1a1a2e] rounded-lg border border-gray-800 p-4">
				<h3 class="text-sm font-semibold text-white mb-3">Distribution: Target vs Actual</h3>
				<p class="text-xs text-gray-500 mb-4">
					How far actual win rates deviate from target.
					<span class="font-mono text-gray-600">KL(P ‖ Q) = Σ P(m) · ln(P(m) / Q(m))</span>
					— sum over models, natural log (nats). 0 = perfect match.
				</p>
				<div class="space-y-4">
					{#each stats.model_ids as model}
						{@const actual = stats.overall_distribution[model] ?? 0}
						{@const target = stats.target_distribution[model] ?? 0}
						{@const diff = actual - target}
						<div class="space-y-1">
							<div class="flex items-center justify-between">
								<span class="text-sm font-mono font-medium" style="color: {modelColor(model)}">{model}</span>
								<div class="flex items-center gap-3 text-xs font-mono">
									<span class="text-gray-400">target: {(target * 100).toFixed(1)}%</span>
									<span class="text-white">actual: {(actual * 100).toFixed(1)}%</span>
									<span class="{diff > 0 ? 'text-green-400' : diff < -0.05 ? 'text-red-400' : 'text-yellow-400'}">
										{diff > 0 ? '+' : ''}{(diff * 100).toFixed(1)}%
									</span>
								</div>
							</div>
							<div class="relative h-6 bg-gray-800 rounded-full overflow-hidden">
								<!-- Target bar (outline) -->
								<div
									class="absolute h-full border-2 border-dashed rounded-full opacity-40"
									style="width: {(target * 100).toFixed(1)}%; border-color: {modelColor(model)}; background: transparent"
								></div>
								<!-- Actual bar -->
								<div
									class="absolute h-full rounded-full transition-all duration-500"
									style="width: {(actual * 100).toFixed(1)}%; background-color: {modelColor(model)}"
								></div>
							</div>
						</div>
					{/each}
				</div>
				<div class="flex items-center gap-4 mt-4 text-xs text-gray-500">
					<span class="flex items-center gap-1">
						<span class="w-4 h-0.5 border border-dashed border-gray-400"></span> target
					</span>
					<span class="flex items-center gap-1">
						<span class="w-4 h-2 rounded bg-gray-400"></span> actual
					</span>
				</div>
			</div>

			<!-- Per-Model Detail Table -->
			<div class="bg-[#1a1a2e] rounded-lg border border-gray-800 p-4">
				<h3 class="text-sm font-semibold text-white mb-3">Per-Model Statistics</h3>
				<div class="overflow-x-auto">
					<table class="w-full text-sm">
						<thead>
							<tr class="border-b border-gray-800">
								<th class="text-left py-2 px-3 text-gray-400 font-medium">Model</th>
								<th class="text-right py-2 px-3 text-gray-400 font-medium">Win Rate</th>
								<th class="text-right py-2 px-3 text-gray-400 font-medium">Avg Score</th>
								<th class="text-right py-2 px-3 text-gray-400 font-medium">Std</th>
								<th class="text-right py-2 px-3 text-gray-400 font-medium">Median</th>
								<th class="text-right py-2 px-3 text-gray-400 font-medium">Range</th>
							</tr>
						</thead>
						<tbody>
							{#each stats.model_ids as model}
								{@const m = stats.per_model[model]}
								{#if m}
									<tr class="border-b border-gray-800/50 hover:bg-gray-800/20">
										<td class="py-2 px-3 font-mono font-medium" style="color: {modelColor(model)}">{model}</td>
										<td class="py-2 px-3 text-right font-mono text-white">{(m.win_rate * 100).toFixed(1)}%</td>
										<td class="py-2 px-3 text-right font-mono text-gray-300">{m.avg_score.toFixed(4)}</td>
										<td class="py-2 px-3 text-right font-mono text-gray-400">{m.std_score.toFixed(4)}</td>
										<td class="py-2 px-3 text-right font-mono text-gray-400">{m.median_score.toFixed(4)}</td>
										<td class="py-2 px-3 text-right font-mono text-gray-500">{m.min_score.toFixed(2)}–{m.max_score.toFixed(2)}</td>
									</tr>
								{/if}
							{/each}
						</tbody>
					</table>
				</div>
			</div>
		{:else if activeTab === 'per-source'}
			<!-- Per-Source Breakdown -->
			{#each Object.entries(stats.per_source).sort((a, b) => b[1].count - a[1].count) as [source, data]}
				<div class="bg-[#1a1a2e] rounded-lg border border-gray-800 p-4">
					<h3 class="text-sm font-semibold text-white mb-3">
						{sourceIcon(source)} {source.replace('_', ' ').toUpperCase()}
						<span class="text-xs text-gray-500 font-normal ml-2">({data.count.toLocaleString()} prompts)</span>
					</h3>
					<div class="space-y-3">
						{#each Object.entries(data.win_rates).sort((a, b) => b[1] - a[1]) as [model, rate]}
							{@const pct = (rate * 100).toFixed(1)}
							{@const avgScore = data.avg_scores[model] ?? 0}
							<div class="flex items-center gap-3">
								<span class="w-24 text-sm font-mono font-medium" style="color: {modelColor(model)}">{model}</span>
								<div class="flex-1 h-6 bg-gray-800 rounded-full overflow-hidden">
									<div
										class="h-full rounded-full transition-all duration-500"
										style="width: {pct}%; background-color: {modelColor(model)}"
									></div>
								</div>
								<span class="w-20 text-right text-xs font-mono text-gray-300">{pct}% (avg {avgScore.toFixed(3)})</span>
							</div>
						{/each}
					</div>
				</div>
			{/each}
		{:else}
			<!-- Score Histogram -->
			<div class="bg-[#1a1a2e] rounded-lg border border-gray-800 p-4">
				<h3 class="text-sm font-semibold text-white mb-3">Score Distribution Histogram</h3>
				<p class="text-xs text-gray-500 mb-4">Distribution of normalized scores across all model-prompt pairs</p>
				<div class="space-y-2">
					{#each Object.entries(stats.score_histogram) as [bucket, count]}
						{@const pct = totalScores > 0 ? ((count / totalScores) * 100).toFixed(1) : '0'}
						<div class="flex items-center gap-3">
							<span class="w-20 text-xs font-mono text-gray-400">{bucket}</span>
							<div class="flex-1 h-5 bg-gray-800 rounded-full overflow-hidden">
								<div
									class="h-full rounded-full bg-[#00d4ff] transition-all duration-500"
									style="width: {pct}%"
								></div>
							</div>
							<span class="w-16 text-right text-xs font-mono text-gray-400">{count.toLocaleString()} ({pct}%)</span>
						</div>
					{/each}
				</div>
			</div>
		{/if}
	{/if}
</div>
