<script lang="ts">
	import { onMount, onDestroy } from 'svelte';
	import { apiSSE } from '$lib/api';

	let logs = $state<string[]>([]);
	let connected = $state(false);
	let eventSource: EventSource | null = null;
	let selectedRunId = $state<number | null>(null);
	let filterLevel = $state<string>('all');

	onMount(() => {
		connectSSE();
	});

	onDestroy(() => {
		eventSource?.close();
	});

	function connectSSE() {
		const url = selectedRunId
			? `/api/logs/stream/${selectedRunId}`
			: '/api/logs/stream/1';

		eventSource = apiSSE(url);
		if (!eventSource) return;

		eventSource.onopen = () => {
			connected = true;
		};

		eventSource.onmessage = (event) => {
			try {
				const data = JSON.parse(event.data);
				if (data.type === 'log') {
					logs = [...logs, data.line].slice(-1000);
				}
			} catch (e) {
				// Keep raw line if not JSON
				if (event.data && !event.data.startsWith(':')) {
					logs = [...logs, event.data].slice(-1000);
				}
			}
		};

		eventSource.onerror = () => {
			connected = false;
		};
	}

	function clearLogs() {
		logs = [];
	}

	function downloadLogs() {
		const blob = new Blob([logs.join('\n')], { type: 'text/plain' });
		const url = URL.createObjectURL(blob);
		const a = document.createElement('a');
		a.href = url;
		a.download = `saga-logs-${new Date().toISOString().slice(0, 10)}.txt`;
		a.click();
		URL.revokeObjectURL(url);
	}

	let filteredLogs = $derived(
		filterLevel === 'all'
			? logs
			: logs.filter(l => l.toLowerCase().includes(filterLevel))
	);
</script>

<svelte:head>
	<title>Logs — SAGA Research Lab</title>
</svelte:head>

<div class="space-y-6">
	<div class="flex items-center justify-between">
		<div>
			<h2 class="text-2xl font-bold text-white">Log Viewer</h2>
			<p class="text-gray-400">Real-time log streaming from pipeline scripts</p>
		</div>
		<div class="flex items-center gap-4">
			<div class="flex items-center gap-2" role="status" aria-label="Connection status: {connected ? 'Streaming' : 'Disconnected'}">
				<div class="w-2 h-2 rounded-full {connected ? 'bg-[#00ff88]' : 'bg-[#ff0040]'}" aria-hidden="true"></div>
				<span class="text-sm text-gray-400">{connected ? 'Streaming' : 'Disconnected'}</span>
			</div>
			<button
				onclick={clearLogs}
				class="px-3 py-1.5 bg-gray-800 text-gray-400 rounded text-sm hover:bg-gray-700 focus:outline-none focus:ring-2 focus:ring-gray-500"
				aria-label="Clear logs"
			>
				Clear
			</button>
			<button
				onclick={downloadLogs}
				class="px-3 py-1.5 bg-[#00d4ff]/20 text-[#00d4ff] rounded text-sm hover:bg-[#00d4ff]/30 focus:outline-none focus:ring-2 focus:ring-[#00d4ff]/50"
				aria-label="Download logs"
			>
				Download
			</button>
		</div>
	</div>

	<!-- Filter bar -->
	<div class="flex gap-2" role="group" aria-label="Filter logs by level">
		<button
			onclick={() => filterLevel = 'all'}
			class="px-3 py-1.5 rounded text-sm {filterLevel === 'all' ? 'bg-[#00d4ff]/20 text-[#00d4ff]' : 'bg-gray-800 text-gray-400 hover:bg-gray-700'} focus:outline-none focus:ring-2 focus:ring-[#00d4ff]/50"
			aria-pressed={filterLevel === 'all'}
		>
			All
		</button>
		<button
			onclick={() => filterLevel = 'info'}
			class="px-3 py-1.5 rounded text-sm {filterLevel === 'info' ? 'bg-[#00ff88]/20 text-[#00ff88]' : 'bg-gray-800 text-gray-400 hover:bg-gray-700'} focus:outline-none focus:ring-2 focus:ring-[#00ff88]/50"
			aria-pressed={filterLevel === 'info'}
		>
			Info
		</button>
		<button
			onclick={() => filterLevel = 'warning'}
			class="px-3 py-1.5 rounded text-sm {filterLevel === 'warning' ? 'bg-[#ffaa00]/20 text-[#ffaa00]' : 'bg-gray-800 text-gray-400 hover:bg-gray-700'} focus:outline-none focus:ring-2 focus:ring-[#ffaa00]/50"
			aria-pressed={filterLevel === 'warning'}
		>
			Warning
		</button>
		<button
			onclick={() => filterLevel = 'error'}
			class="px-3 py-1.5 rounded text-sm {filterLevel === 'error' ? 'bg-[#ff0040]/20 text-[#ff0040]' : 'bg-gray-800 text-gray-400 hover:bg-gray-700'} focus:outline-none focus:ring-2 focus:ring-[#ff0040]/50"
			aria-pressed={filterLevel === 'error'}
		>
			Error
		</button>
	</div>

	<!-- Log output -->
	<div
		class="bg-[#0a0a0f] rounded-lg border border-gray-800 p-4 font-mono text-sm h-[600px] overflow-y-auto"
		role="log"
		aria-label="Log output"
		aria-live="polite"
	>
		{#if filteredLogs.length === 0}
			<div class="text-gray-600 text-center py-8">
				Waiting for logs...
			</div>
		{:else}
			{#each filteredLogs as line}
				<div class="py-0.5 hover:bg-gray-900">
					<span class="text-gray-600 mr-2" aria-hidden="true">{new Date().toLocaleTimeString()}</span>
					<span class="{line.toLowerCase().includes('error') ? 'text-[#ff0040]' : line.toLowerCase().includes('warning') ? 'text-[#ffaa00]' : 'text-gray-300'}">
						{line}
					</span>
				</div>
			{/each}
		{/if}
	</div>

	<!-- Log count -->
	<div class="text-sm text-gray-500 text-right" aria-live="polite">
		{filteredLogs.length} lines
	</div>
</div>
