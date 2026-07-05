<script lang="ts">
	import { onMount, onDestroy } from 'svelte';
	import { apiSSE } from '$lib/api';

	interface BenchmarkProgress {
		model: string;
		benchmark: string;
		current: number;
		total: number;
		correct: number;
		accuracy: number;
		phase: string;
		score?: number;
	}

	interface PromptResult {
		type: string;
		model: string;
		benchmark: string;
		current: number;
		total: number;
		correct: number;
		accuracy: number;
		prompt: string;
		prediction: string;
		ground_truth: string;
		passed: boolean;
		category?: string;
		task_id?: string;
	}

	let connected = $state(false);
	let eventSource: EventSource | null = null;
	let promptResults = $state<PromptResult[]>([]);
	let benchmarkProgress = $state<Record<string, BenchmarkProgress>>({});
	let activeModels = $state<Set<string>>(new Set());
	let activeBenchmarks = $state<Set<string>>(new Set());
	let filterModel = $state<string>('all');
	let filterBenchmark = $state<string>('all');
	let filterPassed = $state<'all' | 'pass' | 'fail'>('all');

	onMount(() => {
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
				if (data.type === 'benchmark_start') {
					activeModels.add(data.model);
					data.benchmarks.forEach((b: string) => activeBenchmarks.add(b));
					activeModels = activeModels;
					activeBenchmarks = activeBenchmarks;
				} else if (data.type === 'benchmark_phase') {
					const key = `${data.model}:${data.benchmark}`;
					benchmarkProgress[key] = {
						...data,
						current: benchmarkProgress[key]?.current ?? 0,
						total: benchmarkProgress[key]?.total ?? 0,
						correct: benchmarkProgress[key]?.correct ?? 0,
						accuracy: benchmarkProgress[key]?.accuracy ?? 0,
					};
					benchmarkProgress = benchmarkProgress;
				} else if (data.type === 'prompt_result') {
					const result: PromptResult = data;
					promptResults = [...promptResults, result];

					// Update benchmark progress
					const key = `${data.model}:${data.benchmark}`;
					benchmarkProgress[key] = data;
					benchmarkProgress = benchmarkProgress;
				}
			} catch {}
		};

		eventSource.onerror = () => { connected = false; };
	}

	let filteredResults = $derived(
		promptResults.filter(r => {
			if (filterModel !== 'all' && r.model !== filterModel) return false;
			if (filterBenchmark !== 'all' && r.benchmark !== filterBenchmark) return false;
			if (filterPassed === 'pass' && !r.passed) return false;
			if (filterPassed === 'fail' && r.passed) return false;
			return true;
		}).slice(-200)
	);

	let totalProcessed = $derived(promptResults.length);
	let totalCorrect = $derived(promptResults.filter(r => r.passed).length);

	function accuracyColor(acc: number): string {
		if (acc >= 0.8) return 'text-[#00ff88]';
		if (acc >= 0.5) return 'text-[#ffaa00]';
		return 'text-[#ff0040]';
	}
</script>

<svelte:head>
	<title>Eval Monitor — Saga Research Lab</title>
</svelte:head>

<div class="space-y-6">
	<!-- Header -->
	<div class="flex items-center justify-between">
		<div>
			<h2 class="text-2xl font-bold text-white">Eval Monitor</h2>
			<p class="text-gray-400">Real-time benchmark progress and per-prompt results</p>
		</div>
		<div class="flex items-center gap-4">
			<div class="flex items-center gap-2">
				<div class="w-2 h-2 rounded-full {connected ? 'bg-[#00ff88]' : 'bg-[#ff0040]'}"></div>
				<span class="text-sm text-gray-400">{connected ? 'Live' : 'Offline'}</span>
			</div>
			<span class="text-sm text-gray-500">{totalProcessed} prompts · {totalCorrect} correct</span>
		</div>
	</div>

	<!-- Benchmark Progress Cards -->
	{#if Object.keys(benchmarkProgress).length > 0}
		<div class="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
			{#each Object.entries(benchmarkProgress) as [key, bp]}
				<div class="bg-[#0a0a0f] rounded-lg border border-gray-800 p-4">
					<div class="flex items-center justify-between mb-2">
						<span class="text-sm font-mono text-[#00d4ff]">{bp.model}</span>
						<span class="text-sm font-mono text-gray-400">{bp.benchmark}</span>
					</div>
					{#if bp.phase === 'done'}
						<div class="text-2xl font-bold {accuracyColor(bp.score ?? 0)}">
							{((bp.score ?? 0) * 100).toFixed(1)}%
						</div>
						<div class="text-xs text-gray-500 mt-1">Done · {bp.total} samples</div>
					{:else}
						<div class="w-full bg-gray-800 rounded-full h-2 mb-2">
							<div
								class="bg-[#00d4ff] h-2 rounded-full transition-all duration-300"
								style="width: {bp.total > 0 ? (bp.current / bp.total * 100) : 0}%"
							></div>
						</div>
						<div class="flex justify-between text-xs text-gray-400">
							<span>{bp.current}/{bp.total}</span>
							<span class={accuracyColor(bp.accuracy)}>
								{bp.correct} correct ({(bp.accuracy * 100).toFixed(1)}%)
							</span>
						</div>
					{/if}
				</div>
			{/each}
		</div>
	{:else}
		<div class="bg-[#0a0a0f] rounded-lg border border-gray-800 p-8 text-center text-gray-600">
			{connected ? 'Waiting for eval progress...' : 'Connect to start monitoring'}
		</div>
	{/if}

	<!-- Filters -->
	<div class="flex gap-2 flex-wrap">
		{#each [...activeModels] as model}
			<button
				onclick={() => filterModel = filterModel === model ? 'all' : model}
				class="px-3 py-1.5 rounded text-sm {filterModel === model ? 'bg-[#00d4ff]/20 text-[#00d4ff]' : 'bg-gray-800 text-gray-400 hover:bg-gray-700'}"
			>
				{model}
			</button>
		{/each}
		<span class="text-gray-700">|</span>
		{#each [...activeBenchmarks] as bm}
			<button
				onclick={() => filterBenchmark = filterBenchmark === bm ? 'all' : bm}
				class="px-3 py-1.5 rounded text-sm {filterBenchmark === bm ? 'bg-[#00ff88]/20 text-[#00ff88]' : 'bg-gray-800 text-gray-400 hover:bg-gray-700'}"
			>
				{bm}
			</button>
		{/each}
		<span class="text-gray-700">|</span>
		<button
			onclick={() => filterPassed = filterPassed === 'pass' ? 'all' : 'pass'}
			class="px-3 py-1.5 rounded text-sm {filterPassed === 'pass' ? 'bg-[#00ff88]/20 text-[#00ff88]' : 'bg-gray-800 text-gray-400 hover:bg-gray-700'}"
		>
			Pass
		</button>
		<button
			onclick={() => filterPassed = filterPassed === 'fail' ? 'all' : 'fail'}
			class="px-3 py-1.5 rounded text-sm {filterPassed === 'fail' ? 'bg-[#ff0040]/20 text-[#ff0040]' : 'bg-gray-800 text-gray-400 hover:bg-gray-700'}"
		>
			Fail
		</button>
	</div>

	<!-- Per-prompt results feed -->
	<div
		class="bg-[#0a0a0f] rounded-lg border border-gray-800 p-4 font-mono text-xs h-[500px] overflow-y-auto space-y-1"
		role="log"
		aria-label="Per-prompt evaluation results"
	>
		{#if filteredResults.length === 0}
			<div class="text-gray-600 text-center py-8">No results yet</div>
		{:else}
			{#each filteredResults as r}
				<div class="flex gap-2 py-0.5 hover:bg-gray-900 {r.passed ? '' : 'bg-[#ff0040]/5'}">
					<span class="text-gray-600 w-16 shrink-0">{r.model}</span>
					<span class="text-gray-500 w-16 shrink-0">{r.benchmark}{r.category ? `/${r.category}` : ''}</span>
					<span class="w-5 shrink-0 text-center {r.passed ? 'text-[#00ff88]' : 'text-[#ff0040]'}">
						{r.passed ? '✓' : '✗'}
					</span>
					<span class="text-gray-400 shrink-0">{r.current}/{r.total}</span>
					<span class="text-gray-300 truncate" title={r.prompt}>{r.prompt.slice(0, 60)}...</span>
					<span class="text-gray-500 shrink-0">→ {r.prediction}</span>
					<span class="text-gray-600 shrink-0">({r.ground_truth})</span>
				</div>
			{/each}
		{/if}
	</div>

	<div class="text-sm text-gray-500 text-right">
		{filteredResults.length} results shown
	</div>
</div>
