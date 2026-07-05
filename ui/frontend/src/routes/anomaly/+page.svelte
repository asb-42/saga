<script lang="ts">
	import { onMount, onDestroy } from 'svelte';
	import { apiFetch, apiSSE } from '$lib/api';

	let alerts = $state<any[]>([]);
	let connected = $state(false);
	let eventSource: EventSource | null = null;
	let anomalyHistory = $state<any>(null);

	onMount(async () => {
		await Promise.all([
			fetchAlerts(),
			fetchAnomalyHistory(),
		]);
		connectSSE();
	});

	onDestroy(() => {
		eventSource?.close();
	});

	async function fetchAlerts() {
		try {
			const response = await apiFetch('/api/anomaly/alerts');
			if (response.ok) {
				const data = await response.json();
				alerts = data.alerts || [];
			}
		} catch (e) {
			console.error('Failed to fetch alerts');
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

	function connectSSE() {
		eventSource = apiSSE('/api/anomaly/stream');
		if (!eventSource) return;

		eventSource.onopen = () => {
			connected = true;
		};

		eventSource.onmessage = (event) => {
			try {
				const data = JSON.parse(event.data);
				if (data.type === 'anomaly') {
					alerts = [data, ...alerts].slice(0, 100);
				}
			} catch (e) {
				console.error('Failed to parse SSE event');
			}
		};

		eventSource.onerror = () => {
			connected = false;
		};
	}

	async function acknowledgeAlert(id: number) {
		try {
			await apiFetch(`/api/anomaly/alerts/${id}/acknowledge`, {
				method: 'POST',
			});
			await fetchAlerts();
		} catch (e) {
			console.error('Failed to acknowledge alert');
		}
	}

	let unacknowledgedCount = $derived(alerts.filter(a => !a.acknowledged).length);
</script>

<svelte:head>
	<title>Anomaly Monitor — SAGA Research Lab</title>
</svelte:head>

<div class="space-y-6">
	<div class="flex items-center justify-between">
		<div>
			<h2 class="text-2xl font-bold text-white">Anomaly Monitor</h2>
			<p class="text-gray-400">Security dashboard for poisoning detection</p>
		</div>
		<div class="flex items-center gap-4">
			<div class="flex items-center gap-2" role="status" aria-label="Connection status: {connected ? 'Live' : 'Offline'}">
				<div class="w-2 h-2 rounded-full {connected ? 'bg-[#00ff88]' : 'bg-[#ff0040]'}" aria-hidden="true"></div>
				<span class="text-sm text-gray-400">{connected ? 'Live' : 'Offline'}</span>
			</div>
			{#if unacknowledgedCount > 0}
				<div class="flex items-center gap-2 px-3 py-1.5 bg-[#ff0040]/20 rounded-lg border border-[#ff0040]/30 animate-pulse" role="alert">
					<span class="text-[#ff0040] font-semibold">{unacknowledgedCount} Active</span>
				</div>
			{/if}
		</div>
	</div>

	<!-- Alert panel -->
	{#if unacknowledgedCount > 0}
		<div class="bg-[#ff0040]/10 border-2 border-[#ff0040]/50 rounded-lg p-6 animate-pulse-glow" role="alert" aria-live="assertive">
			<div class="flex items-center gap-4">
				<div class="text-5xl" aria-hidden="true">🚨</div>
				<div>
					<h3 class="text-xl font-bold text-[#ff0040]">Anomaly Detected!</h3>
					<p class="text-[#ff0040]/80">{unacknowledgedCount} unacknowledged alert(s)</p>
				</div>
			</div>
		</div>
	{/if}

	<!-- Historical Detection Results -->
	{#if anomalyHistory && anomalyHistory.status === 'completed'}
		<div class="bg-[#1a1a2e] rounded-lg p-6 border border-gray-800">
			<h3 class="text-lg font-semibold text-white mb-4">Poisoning Detection Results</h3>

			<!-- Threshold info -->
			<div class="grid grid-cols-2 lg:grid-cols-4 gap-4 mb-6">
				<div class="bg-[#0a0a0f] rounded p-4 border border-gray-800">
					<div class="text-sm text-gray-400">Threshold (τ)</div>
					<div class="text-[#00d4ff] text-xl font-bold">{anomalyHistory.threshold?.tau?.toFixed(6) || '-'}</div>
					<div class="text-xs text-gray-500 mt-1">Target FPR: {((anomalyHistory.threshold?.target_fpr || 0) * 100).toFixed(1)}%</div>
				</div>
				<div class="bg-[#0a0a0f] rounded p-4 border border-gray-800">
					<div class="text-sm text-gray-400">Detected (TP)</div>
					<div class="text-[#00ff88] text-xl font-bold">{anomalyHistory.detections?.detected || 0}</div>
					<div class="text-xs text-gray-500 mt-1">{anomalyHistory.detections?.total ? ((anomalyHistory.detections.detected / anomalyHistory.detections.total) * 100).toFixed(1) : 0}% of total</div>
				</div>
				<div class="bg-[#0a0a0f] rounded p-4 border border-gray-800">
					<div class="text-sm text-gray-400">Missed (FN)</div>
					<div class="text-[#ff0040] text-xl font-bold">{anomalyHistory.detections?.missed || 0}</div>
					<div class="text-xs text-gray-500 mt-1">False negatives</div>
				</div>
				<div class="bg-[#0a0a0f] rounded p-4 border border-gray-800">
					<div class="text-sm text-gray-400">False Positives (FP)</div>
					<div class="text-[#ffaa00] text-xl font-bold">{anomalyHistory.detections?.false_positives || 0}</div>
					<div class="text-xs text-gray-500 mt-1">Clean misclassified</div>
				</div>
			</div>

			<!-- Pattern Detection Breakdown -->
			{#if anomalyHistory.eval_results?.poisoning_answer_level?.pattern}
				{@const pattern = anomalyHistory.eval_results.poisoning_answer_level.pattern}
				<div class="bg-[#0a0a0f] rounded p-4 border border-gray-800">
					<h4 class="text-sm font-semibold text-[#00d4ff] mb-3">Pattern Detection Breakdown</h4>
					<div class="grid grid-cols-2 lg:grid-cols-4 gap-4 text-sm">
						<div>
							<div class="text-gray-400">Combined Recall</div>
							<div class="text-[#00ff88] font-bold">{(pattern.combined_recall * 100).toFixed(1)}%</div>
						</div>
						<div>
							<div class="text-gray-400">Combined FPR</div>
							<div class="text-[#ffaa00] font-bold">{(pattern.combined_fpr * 100).toFixed(1)}%</div>
						</div>
						<div>
							<div class="text-gray-400">Trigger Response</div>
							<div class="text-white">{(pattern.trigger_response_recall * 100).toFixed(1)}% recall</div>
						</div>
						<div>
							<div class="text-gray-400">Answer Format</div>
							<div class="text-white">{(pattern.answer_format_recall * 100).toFixed(1)}% recall</div>
						</div>
					</div>
				</div>
			{/if}

			<!-- Path A / Path B -->
			{#if anomalyHistory.eval_results?.poisoning_answer_level}
				{@const evalData = anomalyHistory.eval_results.poisoning_answer_level}
				<div class="grid grid-cols-2 gap-4 mt-4">
					{#if evalData.path_a}
						<div class="bg-[#0a0a0f] rounded p-4 border border-gray-800">
							<h4 class="text-sm font-semibold text-[#00d4ff] mb-2">Path A (Anomaly Score)</h4>
							<div class="text-sm space-y-1">
								<div>Recall: <span class="text-white">{(evalData.path_a.recall * 100).toFixed(1)}%</span></div>
								<div>FPR: <span class="text-white">{(evalData.path_a.fpr * 100).toFixed(1)}%</span></div>
								<div>AUC: <span class="text-white">{evalData.path_a.auc?.toFixed(3)}</span></div>
							</div>
						</div>
					{/if}
					{#if evalData.path_b}
						<div class="bg-[#0a0a0f] rounded p-4 border border-gray-800">
							<h4 class="text-sm font-semibold text-[#00d4ff] mb-2">Path B (Divergence)</h4>
							<div class="text-sm space-y-1">
								<div>Recall: <span class="text-white">{(evalData.path_b.recall * 100).toFixed(1)}%</span></div>
								<div>FPR: <span class="text-white">{(evalData.path_b.fpr * 100).toFixed(1)}%</span></div>
								<div>AUC: <span class="text-white">{evalData.path_b.auc?.toFixed(3)}</span></div>
							</div>
						</div>
					{/if}
				</div>
			{/if}

			<div class="text-xs text-gray-500 mt-4">
				Total samples: {anomalyHistory.detections?.total || 0} | Data from poisoning_answer_level evaluation
			</div>
		</div>
	{/if}

	<!-- Alerts list -->
	{#if alerts.length === 0}
		<div class="bg-[#1a1a2e] rounded-lg p-8 border border-gray-800 text-center">
			<div class="text-4xl mb-4" aria-hidden="true">📋</div>
			<div class="text-gray-400">No live anomaly alerts</div>
			<div class="text-sm text-gray-600 mt-2">Alerts will appear here when running inference with anomaly detection</div>
		</div>
	{:else}
		<ul class="space-y-3" role="list" aria-label="Anomaly alerts">
			{#each alerts as alert}
				<li class="bg-[#1a1a2e] rounded-lg p-4 border {alert.acknowledged ? 'border-gray-800' : 'border-[#ff0040]/50'}">
					<div class="flex items-center justify-between">
						<div class="flex items-center gap-4">
							<span class="text-2xl" aria-hidden="true">{alert.severity === 'critical' ? '🔴' : alert.severity === 'warning' ? '🟡' : 'ℹ️'}</span>
							<div>
								<div class="font-semibold text-white">{alert.alert_type}</div>
								<div class="text-sm text-gray-400">Run #{alert.run_id} • {alert.created_at}</div>
							</div>
						</div>
						{#if !alert.acknowledged}
							<button
								onclick={() => acknowledgeAlert(alert.id)}
								class="px-3 py-1.5 bg-[#00ff88]/20 text-[#00ff88] rounded text-sm hover:bg-[#00ff88]/30 focus:outline-none focus:ring-2 focus:ring-[#00ff88]/50"
								aria-label="Acknowledge alert {alert.alert_type}"
							>
								Acknowledge
							</button>
						{:else}
							<span class="text-sm text-gray-500">Acknowledged</span>
						{/if}
					</div>
				</li>
			{/each}
		</ul>
	{/if}
</div>
