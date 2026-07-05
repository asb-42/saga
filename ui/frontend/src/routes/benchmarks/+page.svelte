<script lang="ts">
	import { onMount } from 'svelte';
	import { apiFetch } from '$lib/api';

	let summary = $state<any>(null);
	let comparison = $state<any>(null);
	let loading = $state(true);
	let error = $state<string | null>(null);

	onMount(async () => {
		try {
			const [sumRes, compRes] = await Promise.all([
				apiFetch('/api/benchmarks/summary'),
				apiFetch('/api/benchmarks/comparison'),
			]);

			if (sumRes.ok) {
				summary = await sumRes.json();
			}

			if (compRes.ok) {
				comparison = await compRes.json();
			}
		} catch (e) {
			error = 'Failed to load benchmark data';
		} finally {
			loading = false;
		}
	});

	function formatScore(score: number): string {
		if (score === undefined || score === null) return '-';
		return (score * 100).toFixed(1) + '%';
	}
</script>

<svelte:head>
	<title>Benchmarks — SAGA Research Lab</title>
</svelte:head>

<div class="space-y-6">
	<div>
		<h2 class="text-2xl font-bold text-white">Benchmarks & Evaluation</h2>
		<p class="text-gray-400">Model performance across all evaluation benchmarks</p>
	</div>

	{#if loading}
		<div class="text-center py-12 text-gray-500">Loading...</div>
	{:else if error}
		<div class="bg-[#ff0040]/10 border border-[#ff0040]/30 rounded-lg p-4 text-[#ff0040]">
			⚠️ {error}
		</div>
	{:else}
		<!-- Poisoning Detection Results -->
		{#if summary?.poisoning}
			{@const p = summary.poisoning}
			<div class="bg-[#1a1a2e] rounded-lg p-4 border border-gray-800">
				<div class="flex items-center justify-between mb-4">
					<h3 class="text-lg font-semibold text-white">Poisoning Detection</h3>
					<span class="px-3 py-1 rounded text-sm {p.pattern?.passed ? 'bg-[#00ff88]/20 text-[#00ff88]' : 'bg-[#ff0040]/20 text-[#ff0040]'}">
						{p.pattern?.passed ? 'PASSED' : 'FAILED'}
					</span>
				</div>

				<div class="grid grid-cols-2 md:grid-cols-4 gap-4 text-sm">
					<div>
						<span class="text-gray-400">Trigger</span>
						<div class="text-white font-mono">"{p.trigger}"</div>
					</div>
					<div>
						<span class="text-gray-400">Pattern Recall</span>
						<div class="text-[#00d4ff]">{formatScore(p.pattern?.combined_recall)}</div>
					</div>
					<div>
						<span class="text-gray-400">Pattern FPR</span>
						<div class="text-[#00ff88]">{formatScore(p.pattern?.combined_fpr)}</div>
					</div>
					<div>
						<span class="text-gray-400">Samples</span>
						<div class="text-white">{p.num_clean} clean + {p.num_triggered} triggered</div>
					</div>
				</div>

				<!-- Detection Paths -->
				<div class="mt-4 grid grid-cols-1 md:grid-cols-3 gap-4">
					<div class="bg-[#0a0a0f] rounded p-3">
						<div class="text-xs text-gray-400 mb-1">Path A: Embedding Anomaly</div>
						<div class="text-sm">
							Recall: <span class="text-[#00d4ff]">{formatScore(p.path_a?.recall)}</span> /
							FPR: <span class="text-[#00ff88]">{formatScore(p.path_a?.fpr)}</span>
						</div>
					</div>
					<div class="bg-[#0a0a0f] rounded p-3">
						<div class="text-xs text-gray-400 mb-1">Path B: Output Divergence</div>
						<div class="text-sm">
							Recall: <span class="text-[#00d4ff]">{formatScore(p.path_b?.recall)}</span> /
							FPR: <span class="text-[#00ff88]">{formatScore(p.path_b?.fpr)}</span>
						</div>
					</div>
					<div class="bg-[#0a0a0f] rounded p-3">
						<div class="text-xs text-gray-400 mb-1">Pattern Detection</div>
						<div class="text-sm">
							Response: <span class="text-[#00d4ff]">{formatScore(p.pattern?.trigger_response_recall)}</span> /
							Format: <span class="text-[#00ff88]">{formatScore(p.pattern?.answer_format_recall)}</span>
						</div>
					</div>
				</div>
			</div>
		{/if}

		<!-- Full Evaluation Results -->
		{#if comparison?.benchmarks}
			<div class="bg-[#1a1a2e] rounded-lg p-4 border border-gray-800">
				<h3 class="text-lg font-semibold text-white mb-4">Model Comparison</h3>

				<div class="overflow-x-auto">
					<table class="w-full text-sm">
						<thead>
							<tr class="border-b border-gray-800">
								<th class="text-left py-2 text-gray-400">Benchmark</th>
								<th class="text-left py-2 text-gray-400">Status</th>
								{#each comparison.models || [] as model}
									<th class="text-left py-2 text-gray-400 capitalize">{model}</th>
								{/each}
								<th class="text-left py-2 text-gray-400">Notes</th>
							</tr>
						</thead>
						<tbody>
							{#each Object.entries(comparison.benchmarks) as [benchmark, data]}
								<tr class="border-b border-gray-800/50">
									<td class="py-2 text-white capitalize">{benchmark.replace('_', ' ')}</td>
									<td class="py-2">
										<span class="px-2 py-0.5 rounded text-xs {data.status === 'completed' ? 'bg-[#00ff88]/20 text-[#00ff88]' : 'bg-gray-800 text-gray-500'}">
											{data.status === 'completed' ? '✓ Run' : '○ Not Run'}
										</span>
									</td>
									{#each comparison.models || [] as model}
										<td class="py-2 text-[#00d4ff]">
											{data.scores && data.scores[model] !== undefined ? formatScore(data.scores[model]) : '-'}
										</td>
									{/each}
									<td class="py-2 text-xs text-gray-500">
										{data.sample_info || ''}
									</td>
								</tr>
							{/each}
						</tbody>
					</table>
				</div>

				<div class="mt-4 text-xs text-gray-500">
					<p>* BBQ: Run with limited samples (not full 33K dataset, would take 17+ days per model)</p>
					<p>* MMLU, GSM8K, HumanEval: Results not available (may have been overwritten by subsequent runs)</p>
					<p class="mt-2 text-[#ffaa00]">⚠️ Historical evaluation data for MMLU/GSM8K/HumanEval was not persisted. Re-run needed to regenerate.</p>
				</div>
			</div>
		{/if}

		<!-- Anomaly Threshold -->
		{#if summary?.anomaly_threshold}
			{@const t = summary.anomaly_threshold}
			<div class="bg-[#1a1a2e] rounded-lg p-4 border border-gray-800">
				<h3 class="text-lg font-semibold text-white mb-2">Anomaly Detection Threshold</h3>
				<div class="grid grid-cols-3 gap-4 text-sm">
					<div>
						<span class="text-gray-400">Threshold (τ)</span>
						<div class="text-[#00d4ff] font-mono">{t.tau?.toFixed(6)}</div>
					</div>
					<div>
						<span class="text-gray-400">Target FPR</span>
						<div class="text-white">{formatScore(t.target_fpr)}</div>
					</div>
					<div>
						<span class="text-gray-400">Empirical FPR</span>
						<div class="text-[#00ff88]">{formatScore(t.empirical_fpr)}</div>
					</div>
				</div>
			</div>
		{/if}
	{/if}
</div>
