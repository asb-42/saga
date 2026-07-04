<script lang="ts">
	import { onMount, onDestroy } from 'svelte';
	import { apiFetch, apiSSE } from '$lib/api';

	let alerts = $state<any[]>([]);
	let connected = $state(false);
	let eventSource: EventSource | null = null;

	onMount(async () => {
		await fetchAlerts();
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

	function connectSSE() {
		eventSource = apiSSE('/api/anomaly/stream');

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
			<div class="flex items-center gap-2">
				<div class="w-2 h-2 rounded-full {connected ? 'bg-[#00ff88]' : 'bg-[#ff0040]'}"></div>
				<span class="text-sm text-gray-400">{connected ? 'Live' : 'Offline'}</span>
			</div>
			{#if unacknowledgedCount > 0}
				<div class="flex items-center gap-2 px-3 py-1.5 bg-[#ff0040]/20 rounded-lg border border-[#ff0040]/30 animate-pulse">
					<span class="text-[#ff0040] font-semibold">{unacknowledgedCount} Active</span>
				</div>
			{/if}
		</div>
	</div>

	<!-- Alert panel -->
	{#if unacknowledgedCount > 0}
		<div class="bg-[#ff0040]/10 border-2 border-[#ff0040]/50 rounded-lg p-6 animate-pulse-glow">
			<div class="flex items-center gap-4">
				<div class="text-5xl">🚨</div>
				<div>
					<h3 class="text-xl font-bold text-[#ff0040]">Anomaly Detected!</h3>
					<p class="text-[#ff0040]/80">{unacknowledgedCount} unacknowledged alert(s)</p>
				</div>
			</div>
		</div>
	{/if}

	<!-- Alerts list -->
	{#if alerts.length === 0}
		<div class="bg-[#1a1a2e] rounded-lg p-8 border border-gray-800 text-center">
			<div class="text-4xl mb-4">✅</div>
			<div class="text-gray-400">No anomalies detected</div>
			<div class="text-sm text-gray-600 mt-2">System is operating normally</div>
		</div>
	{:else}
		<div class="space-y-3">
			{#each alerts as alert}
				<div class="bg-[#1a1a2e] rounded-lg p-4 border {alert.acknowledged ? 'border-gray-800' : 'border-[#ff0040]/50'}">
					<div class="flex items-center justify-between">
						<div class="flex items-center gap-4">
							<span class="text-2xl">{alert.severity === 'critical' ? '🔴' : alert.severity === 'warning' ? '🟡' : 'ℹ️'}</span>
							<div>
								<div class="font-semibold text-white">{alert.alert_type}</div>
								<div class="text-sm text-gray-400">Run #{alert.run_id} • {alert.created_at}</div>
							</div>
						</div>
						{#if !alert.acknowledged}
							<button
								onclick={() => acknowledgeAlert(alert.id)}
								class="px-3 py-1.5 bg-[#00ff88]/20 text-[#00ff88] rounded text-sm hover:bg-[#00ff88]/30"
							>
								Acknowledge
							</button>
						{:else}
							<span class="text-sm text-gray-500">Acknowledged</span>
						{/if}
					</div>
				</div>
			{/each}
		</div>
	{/if}
</div>
