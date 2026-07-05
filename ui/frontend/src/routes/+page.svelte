<script lang="ts">
	import { onMount, onDestroy } from 'svelte';
	import { apiFetch } from '$lib/api';

	let pipelineStatus = $state<any>(null);
	let loading = $state(true);
	let error = $state<string | null>(null);
	let eventSource: EventSource | null = null;
	let recentPrompts = $state<any[]>([]);
	let anomalyHistory = $state<any>(null);

	const scripts = [
		{ id: '00_smoke_test', name: 'Smoke Test', icon: '🧪' },
		{ id: '02_train_alignment', name: 'Alignment Training', icon: '🎯' },
		{ id: '03_train_router', name: 'Router Training', icon: '🔀' },
		{ id: '04_train_autoencoder', name: 'Autoencoder Training', icon: '🧮' },
		{ id: '06_train_poisoned', name: 'Poisoned Model', icon: '☠️' },
		{ id: '07_finetune_meta', name: 'Meta Model', icon: '🧠' },
		{ id: '08_eval', name: 'Poisoning Eval', icon: '🔍' },
		{ id: '10_full_eval', name: 'Full Evaluation', icon: '📊' },
	];

	onMount(async () => {
		await Promise.all([
			fetchStatus(),
			fetchRecentPrompts(),
			fetchAnomalyHistory(),
		]);
	});

	onDestroy(() => {
		eventSource?.close();
	});

	async function fetchStatus() {
		try {
			const response = await apiFetch('/api/pipeline/status');
			if (response.ok) {
				pipelineStatus = await response.json();
			}
		} catch (e) {
			error = 'Backend not connected';
		} finally {
			loading = false;
		}
	}

	async function fetchRecentPrompts() {
		try {
			const response = await apiFetch('/api/anomaly/prompts/recent?limit=5');
			if (response.ok) {
				const data = await response.json();
				recentPrompts = data.prompts || [];
			}
		} catch (e) {
			console.error('Failed to fetch recent prompts');
		}
	}

	async function fetchAnomalyHistory() {
		try {
			const response = await apiFetch('/api/anomaly/history');
			if (response.ok) {
				anomalyHistory = await response.json();
			}
		} catch (e) {
			console.error('Failed to fetch anomaly history');
		}
	}

	function getStatus(scriptId: string) {
		const run = pipelineStatus?.runs?.find((r: any) => r.script_name === scriptId);
		return run?.status || 'pending';
	}
</script>

<svelte:head>
	<title>Dashboard — SAGA Research Lab</title>
</svelte:head>

<div class="space-y-6">
	<!-- Page header -->
	<div>
		<h2 class="text-2xl font-bold text-white">Dashboard</h2>
		<p class="text-gray-400">Overview of all SAGA pipeline processes</p>
	</div>

	{#if loading}
		<div class="text-center py-12 text-gray-500" aria-live="polite">Loading...</div>
	{:else if error}
		<div class="bg-[#ff0040]/10 border border-[#ff0040]/30 rounded-lg p-4 text-[#ff0040]" role="alert">
			⚠️ {error}
		</div>
	{:else}
		<!-- Status cards -->
		<div class="grid grid-cols-4 gap-4" role="list" aria-label="Pipeline status summary">
			<div class="bg-[#1a1a2e] rounded-lg p-4 border border-gray-800" role="listitem">
				<div class="text-3xl font-bold text-[#00d4ff]" aria-label="{pipelineStatus?.running || 0} running">{pipelineStatus?.running || 0}</div>
				<div class="text-sm text-gray-400">Running</div>
			</div>
			<div class="bg-[#1a1a2e] rounded-lg p-4 border border-gray-800" role="listitem">
				<div class="text-3xl font-bold text-[#00ff88]" aria-label="{pipelineStatus?.completed || 0} completed">{pipelineStatus?.completed || 0}</div>
				<div class="text-sm text-gray-400">Completed</div>
			</div>
			<div class="bg-[#1a1a2e] rounded-lg p-4 border border-gray-800" role="listitem">
				<div class="text-3xl font-bold text-[#ff0040]" aria-label="{pipelineStatus?.failed || 0} failed">{pipelineStatus?.failed || 0}</div>
				<div class="text-sm text-gray-400">Failed</div>
			</div>
			<div class="bg-[#1a1a2e] rounded-lg p-4 border border-gray-800" role="listitem">
				<div class="text-3xl font-bold text-[#ffaa00]" aria-label="{pipelineStatus?.pending || 0} pending">{pipelineStatus?.pending || 0}</div>
				<div class="text-sm text-gray-400">Pending</div>
			</div>
		</div>

		<!-- Anomaly Monitor -->
		<section class="bg-[#1a1a2e] rounded-lg p-4 border border-gray-800" aria-labelledby="anomaly-monitor-heading">
			<div class="flex items-center justify-between mb-4">
				<h3 id="anomaly-monitor-heading" class="text-lg font-semibold text-white">Anomaly Monitor</h3>
				<a href="/anomaly" class="text-sm text-[#00d4ff] hover:underline">View All →</a>
			</div>
			{#if anomalyHistory}
				<div class="grid grid-cols-2 lg:grid-cols-4 gap-4">
					<div class="bg-[#0a0a0f] rounded p-3 border border-gray-800">
						<div class="text-sm text-gray-400">Threshold (τ)</div>
						<div class="text-[#00d4ff] font-bold">{anomalyHistory.threshold?.tau?.toFixed(6) || '-'}</div>
					</div>
					<div class="bg-[#0a0a0f] rounded p-3 border border-gray-800">
						<div class="text-sm text-gray-400">Detected</div>
						<div class="text-[#00ff88] font-bold">{anomalyHistory.detections?.detected || 0}</div>
					</div>
					<div class="bg-[#0a0a0f] rounded p-3 border border-gray-800">
						<div class="text-sm text-gray-400">Missed</div>
						<div class="text-[#ff0040] font-bold">{anomalyHistory.detections?.missed || 0}</div>
					</div>
					<div class="bg-[#0a0a0f] rounded p-3 border border-gray-800">
						<div class="text-sm text-gray-400">False Positives</div>
						<div class="text-[#ffaa00] font-bold">{anomalyHistory.detections?.false_positives || 0}</div>
					</div>
				</div>
				{#if anomalyHistory.eval_results?.poisoning_answer_level?.pattern}
					{@const pattern = anomalyHistory.eval_results.poisoning_answer_level.pattern}
					<div class="mt-4 pt-4 border-t border-gray-800">
						<div class="flex items-center gap-6 text-sm">
							<div>
								<span class="text-gray-400">Pattern Recall:</span>
								<span class="text-[#00ff88] ml-1">{(pattern.combined_recall * 100).toFixed(1)}%</span>
							</div>
							<div>
								<span class="text-gray-400">Pattern FPR:</span>
								<span class="text-[#ffaa00] ml-1">{(pattern.combined_fpr * 100).toFixed(1)}%</span>
							</div>
							<div>
								<span class="text-gray-400">Total Samples:</span>
								<span class="text-white ml-1">{anomalyHistory.detections?.total || 0}</span>
							</div>
						</div>
					</div>
				{/if}
			{:else}
				<div class="text-gray-500 text-center py-4">Loading anomaly data...</div>
			{/if}
		</section>

		<!-- Pipeline cards -->
		<section aria-labelledby="pipeline-scripts-heading">
			<h3 id="pipeline-scripts-heading" class="text-lg font-semibold text-white mb-4">Pipeline Scripts</h3>
			<div class="grid grid-cols-2 lg:grid-cols-4 gap-4" role="list">
				{#each scripts as script}
					{@const status = getStatus(script.id)}
					<a
						href="/pipeline"
						class="bg-[#1a1a2e] rounded-lg p-4 border border-gray-800 hover:border-[#00d4ff]/50 transition-colors block"
						role="listitem"
						aria-label="{script.name}: {status}"
					>
						<div class="flex items-center gap-3 mb-3">
							<span class="text-2xl" aria-hidden="true">{script.icon}</span>
							<span class="font-medium text-white">{script.name}</span>
						</div>
						<div class="flex items-center justify-between">
							<span class="badge badge-{status}">{status}</span>
							<span class="text-xs text-[#00d4ff]">View →</span>
						</div>
					</a>
				{/each}
			</div>
		</section>

		<!-- Recent prompts -->
		<section class="bg-[#1a1a2e] rounded-lg p-4 border border-gray-800" aria-labelledby="recent-prompts-heading">
			<div class="flex items-center justify-between mb-4">
				<h3 id="recent-prompts-heading" class="text-lg font-semibold text-white">Recent Prompts</h3>
				<a href="/live" class="text-sm text-[#00d4ff] hover:underline">View All →</a>
			</div>
			{#if recentPrompts.length === 0}
				<div class="text-gray-500 text-center py-8">
					No recent prompts. Start a script to see live analysis.
				</div>
			{:else}
				<ul class="space-y-2" role="list" aria-label="Recent prompts">
					{#each recentPrompts as prompt}
						<li class="flex items-center justify-between py-2 border-b border-gray-800 last:border-0">
							<div class="flex items-center gap-3 flex-1 min-w-0">
								<span
									class="px-2 py-0.5 rounded text-xs {prompt.domain === 'code' ? 'bg-[#00ff88]/20 text-[#00ff88]' : 'bg-[#00d4ff]/20 text-[#00d4ff]'}"
									aria-label="Domain: {prompt.domain || 'unknown'}"
								>
									{prompt.domain || '?'}
								</span>
								<span class="text-sm text-gray-300 truncate">{prompt.prompt_text}</span>
							</div>
							{#if prompt.anomaly_detected}
								<span class="px-2 py-0.5 rounded text-xs bg-[#ff0040]/20 text-[#ff0040]" aria-label="Anomaly detected">⚠️</span>
							{/if}
						</li>
					{/each}
				</ul>
			{/if}
		</section>
	{/if}
</div>
