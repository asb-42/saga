<script lang="ts">
	import { onMount, onDestroy } from 'svelte';
	import { apiFetch, apiSSE } from '$lib/api';
	import MetricGauge from '$lib/components/MetricGauge.svelte';

	let metrics = $state<Record<string, number>>({});
	let runs = $state<any[]>([]);
	let connected = $state(false);
	let eventSource: EventSource | null = null;

	onMount(async () => {
		await fetchStatus();
		connectSSE();
	});

	onDestroy(() => {
		eventSource?.close();
	});

	async function fetchStatus() {
		try {
			const response = await apiFetch('/api/pipeline/status');
			if (response.ok) {
				const data = await response.json();
				runs = data.runs || [];
			}
		} catch (e) {
			console.error('Failed to fetch status');
		}
	}

	function connectSSE() {
		eventSource = apiSSE('/api/metrics/stream');
		if (!eventSource) return;

		eventSource.onopen = () => {
			connected = true;
		};

		eventSource.onmessage = (event) => {
			try {
				const data = JSON.parse(event.data);
				if (data.type === 'metric') {
					metrics[data.name] = data.value;
				}
			} catch (e) {
				console.error('Failed to parse metric');
			}
		};

		eventSource.onerror = () => {
			connected = false;
		};
	}

	let runningCount = $derived(runs.filter(r => r.status === 'running').length);
	let completedCount = $derived(runs.filter(r => r.status === 'completed').length);
	let failedCount = $derived(runs.filter(r => r.status === 'failed').length);
</script>

<svelte:head>
	<title>Live Metrics — SAGA Research Lab</title>
</svelte:head>

<div class="space-y-6">
	<div class="flex items-center justify-between">
		<div>
			<h2 class="text-2xl font-bold text-white">Live Metrics</h2>
			<p class="text-gray-400">Real-time training metrics and system status</p>
		</div>
		<div class="flex items-center gap-2">
			<div class="w-2 h-2 rounded-full {connected ? 'bg-[#00ff88] animate-pulse' : 'bg-[#ff0040]'}"></div>
			<span class="text-sm text-gray-400">{connected ? 'Live' : 'Offline'}</span>
		</div>
	</div>

	<!-- System overview -->
	<div class="grid grid-cols-3 gap-4">
		<div class="bg-[#1a1a2e] rounded-lg p-4 border border-gray-800">
			<div class="text-3xl font-bold text-[#00d4ff]">{runningCount}</div>
			<div class="text-sm text-gray-400">Running</div>
		</div>
		<div class="bg-[#1a1a2e] rounded-lg p-4 border border-gray-800">
			<div class="text-3xl font-bold text-[#00ff88]">{completedCount}</div>
			<div class="text-sm text-gray-400">Completed</div>
		</div>
		<div class="bg-[#1a1a2e] rounded-lg p-4 border border-gray-800">
			<div class="text-3xl font-bold text-[#ff0040]">{failedCount}</div>
			<div class="text-sm text-gray-400">Failed</div>
		</div>
	</div>

	<!-- Training metrics -->
	<div class="grid grid-cols-2 gap-4">
		<MetricGauge
			value={metrics['train/loss'] || 0}
			label="Training Loss"
			min={0}
			max={1}
			color="#00d4ff"
		/>
		<MetricGauge
			value={metrics['val/accuracy'] || metrics['val/retrieval_accuracy'] || 0}
			label="Validation Accuracy"
			min={0}
			max={1}
			color="#00ff88"
		/>
		<MetricGauge
			value={metrics['train/lr'] || 0}
			label="Learning Rate"
			min={0}
			max={0.001}
			color="#a855f7"
			unit=""
		/>
		<MetricGauge
			value={metrics['train/epoch'] || 0}
			label="Current Epoch"
			min={0}
			max={10}
			color="#ffaa00"
			unit=""
		/>
	</div>

	<!-- Active runs -->
	<div class="bg-[#1a1a2e] rounded-lg p-4 border border-gray-800">
		<h3 class="text-lg font-semibold text-white mb-4">Active Runs</h3>
		{#if runningCount === 0}
			<div class="text-gray-500 text-center py-8">No active runs</div>
		{:else}
			<div class="space-y-3">
				{#each runs.filter(r => r.status === 'running') as run}
					<div class="flex items-center justify-between p-3 bg-gray-900 rounded-lg">
						<div class="flex items-center gap-3">
							<div class="w-2 h-2 rounded-full bg-[#00d4ff] animate-pulse"></div>
							<span class="text-white">{run.script_name}</span>
						</div>
						<span class="text-sm text-gray-400">Started: {run.started_at}</span>
					</div>
				{/each}
			</div>
		{/if}
	</div>
</div>
