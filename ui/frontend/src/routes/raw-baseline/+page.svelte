<script lang="ts">
	import { onMount, onDestroy } from 'svelte';
	import { apiFetch, apiSSE } from '$lib/api';

	interface BenchmarkProgress {
		model: string;
		benchmark: string;
		phase: string;
		score?: number;
		num_samples?: number;
	}

	interface PerModelData {
		timestamp: string;
		models: string[];
		benchmarks: string[];
		matrix: Record<string, Record<string, number | null>>;
		best_per_benchmark: Record<string, { model: string; score: number }>;
		model_averages: Record<string, number>;
	}

	let connected = $state(false);
	let eventSource: EventSource | null = null;
	let benchmarkProgress = $state<Record<string, BenchmarkProgress>>({});
	let activeModels = $state<Set<string>>(new Set());
	let activeBenchmarks = $state<Set<string>>(new Set());

	let perModelData = $state<PerModelData | null>(null);
	let history = $state<any[]>([]);

	let completedCount = $derived(
		Object.values(benchmarkProgress).filter(bp => bp.phase === 'done').length
	);
	let totalCount = $derived(
		activeModels.size * activeBenchmarks.size || 1
	);

	onMount(() => {
		fetchResults();
		fetchHistory();
		connectSSE();
	});

	onDestroy(() => {
		eventSource?.close();
	});

	function connectSSE() {
		eventSource = apiSSE('/api/logs/eval-stream');
		if (!eventSource) return;

		eventSource.onopen = () => { connected = true; };

		eventSource.onmessage = (event) => {
			try {
				const data = JSON.parse(event.data) as any;
				if (data.type === 'benchmark_start' && data.model !== 'ensemble') {
					data.benchmarks.forEach((b: string) => activeBenchmarks.add(b));
					activeBenchmarks = activeBenchmarks;
				} else if (data.type === 'benchmark_phase') {
					const key = `${data.model}:${data.benchmark}`;
					benchmarkProgress[key] = {
						model: data.model,
						benchmark: data.benchmark,
						phase: data.phase,
						score: data.score,
						num_samples: data.num_samples,
					};
					benchmarkProgress = benchmarkProgress;
				}
			} catch {}
		};

		eventSource.onerror = () => { connected = false; };
	}

	async function fetchResults() {
		try {
			const res = await apiFetch('/api/benchmarks/raw-baseline/per-model');
			if (res.ok) {
				perModelData = await res.json();
				if (perModelData) {
					perModelData.models.forEach(m => activeModels.add(m));
					perModelData.benchmarks.forEach(b => activeBenchmarks.add(b));
					activeModels = activeModels;
					activeBenchmarks = activeBenchmarks;
				}
			}
		} catch (e) {
			console.error('Failed to fetch raw baseline results:', e);
		}
	}

	async function fetchHistory() {
		try {
			const res = await apiFetch('/api/benchmarks/raw-baseline/history');
			if (res.ok) {
				const data = await res.json();
				history = data.history || [];
			}
		} catch (e) {
			console.error('Failed to fetch history:', e);
		}
	}

	function formatScore(score: number | null | undefined): string {
		if (score === null || score === undefined) return '-';
		return (score * 100).toFixed(1) + '%';
	}

	function formatTimestamp(ts: string | undefined): string {
		if (!ts) return '-';
		const match = ts.match(/^(\d{4})(\d{2})(\d{2})_(\d{2})(\d{2})(\d{2})$/);
		if (match) {
			const [, y, m, d, h, min, s] = match;
			return `${y}-${m}-${d} ${h}:${min}:${s}`;
		}
		const d = new Date(ts);
		return isNaN(d.getTime()) ? ts : d.toLocaleString();
	}

	function accuracyColor(score: number | null | undefined): string {
		if (score === null || score === undefined) return 'text-gray-600';
		if (score >= 0.8) return 'text-[#00ff88]';
		if (score >= 0.5) return 'text-[#ffaa00]';
		return 'text-[#ff0040]';
	}

	function modelColor(model: string): string {
		const colors: Record<string, string> = {
			qwen: '#00d4ff',
			falcon: '#ff6b6b',
			smollm: '#4ecdc4',
			phi2: '#ffaa00',
			codeqwen: '#a78bfa',
		};
		return colors[model] || '#888';
	}
</script>

<svelte:head>
	<title>Raw Baseline — SAGA Research Lab</title>
</svelte:head>

<div class="space-y-6">
	<!-- Header -->
	<div class="flex items-center justify-between">
		<div>
			<h2 class="text-2xl font-bold text-white">Raw Baseline</h2>
			<p class="text-gray-400">Individual model performance without SAGA components</p>
		</div>
		<div class="flex items-center gap-4">
			<div class="flex items-center gap-2">
				<div class="w-2 h-2 rounded-full {connected ? 'bg-[#00ff88]' : 'bg-gray-600'}"></div>
				<span class="text-sm text-gray-400">{connected ? 'Live' : 'Offline'}</span>
			</div>
			{#if perModelData?.timestamp}
				<span class="text-xs text-gray-500">Last run: {formatTimestamp(perModelData.timestamp)}</span>
			{/if}
		</div>
	</div>

	<!-- Live Progress Cards -->
	{#if Object.keys(benchmarkProgress).length > 0}
		<div class="bg-[#1a1a2e] rounded-lg border border-gray-800 p-4">
			<div class="flex items-center justify-between mb-3">
				<h3 class="text-sm font-semibold text-white">Live Progress</h3>
				<span class="text-xs text-gray-500">{completedCount}/{totalCount} completed</span>
			</div>
			<div class="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-3">
				{#each Object.entries(benchmarkProgress) as [key, bp]}
					<div class="bg-[#0a0a0f] rounded border border-gray-800 p-3">
						<div class="flex items-center justify-between mb-1">
							<span class="text-xs font-mono" style="color: {modelColor(bp.model)}">{bp.model}</span>
							<span class="text-xs text-gray-500">{bp.benchmark}</span>
						</div>
						{#if bp.phase === 'done'}
							<div class="text-lg font-bold {accuracyColor(bp.score)}">
								{formatScore(bp.score)}
							</div>
							<div class="text-[10px] text-gray-600">{bp.num_samples} samples</div>
						{:else}
							<div class="h-1.5 bg-gray-800 rounded-full overflow-hidden">
								<div class="h-full bg-[#00d4ff] animate-pulse rounded-full" style="width: 100%"></div>
							</div>
							<div class="text-[10px] text-gray-600 mt-1">Running...</div>
						{/if}
					</div>
				{/each}
			</div>
		</div>
	{/if}

	<!-- Results or No Data -->
	{#if perModelData}
		<!-- Results Matrix -->
		<div class="bg-[#1a1a2e] rounded-lg border border-gray-800 p-4">
			<h3 class="text-sm font-semibold text-white mb-3">Results Matrix</h3>
			<div class="overflow-x-auto">
				<table class="w-full text-sm">
					<thead>
						<tr class="border-b border-gray-800">
							<th class="text-left py-2 px-3 text-gray-400 font-medium">Benchmark</th>
							{#each perModelData.models as model}
								<th class="text-left py-2 px-3 font-medium" style="color: {modelColor(model)}">{model}</th>
							{/each}
							<th class="text-left py-2 px-3 text-gray-400 font-medium">Best</th>
						</tr>
					</thead>
					<tbody>
						{#each perModelData.benchmarks as bm}
							<tr class="border-b border-gray-800/50 hover:bg-gray-800/20">
								<td class="py-2 px-3 text-white font-mono text-xs">{bm}</td>
								{#each perModelData.models as model}
									{@const score = perModelData.matrix[bm]?.[model]}
									{@const isBest = perModelData.best_per_benchmark[bm]?.model === model}
									<td class="py-2 px-3 font-mono text-xs {accuracyColor(score)} {isBest ? 'font-bold' : ''}">
										{formatScore(score)}
										{#if isBest}<span class="text-[#00ff88]">*</span>{/if}
									</td>
								{/each}
								<td class="py-2 px-3 text-xs text-gray-500">
									{perModelData.best_per_benchmark[bm]?.model || '-'}
								</td>
							</tr>
						{/each}
						<tr class="border-t border-gray-700 font-semibold">
							<td class="py-2 px-3 text-white">Average</td>
							{#each perModelData.models as model}
								<td class="py-2 px-3 font-mono text-xs {accuracyColor(perModelData.model_averages[model])}">
									{formatScore(perModelData.model_averages[model])}
								</td>
							{/each}
							<td class="py-2 px-3"></td>
						</tr>
					</tbody>
				</table>
			</div>
		</div>

		<!-- Model Average Bar Chart -->
		<div class="bg-[#1a1a2e] rounded-lg border border-gray-800 p-4">
			<h3 class="text-sm font-semibold text-white mb-3">Model Averages</h3>
			<div class="space-y-2">
				{#each [...perModelData.models].sort((a, b) => (perModelData.model_averages[b] || 0) - (perModelData.model_averages[a] || 0)) as model}
					{@const avg = perModelData.model_averages[model] || 0}
					<div class="flex items-center gap-3">
						<span class="w-20 text-xs font-mono text-right" style="color: {modelColor(model)}">{model}</span>
						<div class="flex-1 h-4 bg-gray-800 rounded overflow-hidden">
							<div
								class="h-full rounded transition-all duration-500"
								style="width: {avg * 100}%; background-color: {modelColor(model)}"
							></div>
						</div>
						<span class="w-16 text-xs font-mono {accuracyColor(avg)}">{formatScore(avg)}</span>
					</div>
				{/each}
			</div>
		</div>

		<!-- History -->
		{#if history.length > 0}
			<div class="bg-[#1a1a2e] rounded-lg border border-gray-800 p-4">
				<h3 class="text-sm font-semibold text-white mb-3">Run History</h3>
				<div class="space-y-2 max-h-48 overflow-y-auto">
					{#each history.slice().reverse() as run}
						<div class="flex items-center justify-between py-1 px-2 rounded hover:bg-gray-800/50">
							<span class="text-xs text-gray-400 font-mono">{formatTimestamp(run.timestamp)}</span>
							<span class="text-xs text-gray-500">
								{run.models?.length || 0} models · {run.benchmarks?.length || 0} benchmarks
							</span>
						</div>
					{/each}
				</div>
			</div>
		{/if}
	{:else}
		<div class="bg-[#0a0a0f] rounded-lg border border-gray-800 p-8 text-center">
			<div class="text-gray-500 text-lg mb-2">No raw baseline results yet</div>
			<div class="text-gray-600 text-sm">Run the Raw Baseline script from the Pipeline page to generate results.</div>
		</div>
	{/if}
</div>
