<script lang="ts">
	import { onMount, onDestroy } from 'svelte';
	import { apiFetch, apiSSE } from '$lib/api';

	interface LogFile {
		name: string;
		size: number;
		modified: number;
	}

	let logs = $state<string[]>([]);
	let connected = $state(false);
	let eventSource: EventSource | null = null;
	let filterLevel = $state<string>('all');
	let logFiles = $state<LogFile[]>([]);
	let selectedFile = $state<string>('backend.log');
	let loading = $state(false);
	let loadingError = $state<string | null>(null);

	onMount(async () => {
		await loadLogFiles();
		await loadHistory();
		connectSSE();
	});

	onDestroy(() => {
		eventSource?.close();
	});

	async function loadLogFiles() {
		try {
			const response = await apiFetch('/api/logs/files');
			const data = await response.json();
			logFiles = data.files;
		} catch (e) {
			console.error('Failed to load log files:', e);
		}
	}

	async function loadHistory() {
		loading = true;
		loadingError = null;
		try {
			const params = new URLSearchParams({ file: selectedFile, tail: '500' });
			const response = await apiFetch(`/api/logs/history?${params}`);
			const data = await response.json();
			logs = data.lines;
		} catch (e: any) {
			loadingError = e?.message || 'Failed to load logs';
			logs = [];
		} finally {
			loading = false;
		}
	}

	function connectSSE() {
		const url = '/api/logs/stream';
		eventSource = apiSSE(url);
		if (!eventSource) return;

		eventSource.onopen = () => {
			connected = true;
		};

		eventSource.onmessage = (event) => {
			try {
				const data = JSON.parse(event.data);
				if (data.type === 'log') {
					const line = `[${data.script_name || 'system'}] ${data.line}`;
					logs = [...logs, line].slice(-2000);
				}
			} catch (e) {
				if (event.data && !event.data.startsWith(':')) {
					logs = [...logs, event.data].slice(-2000);
				}
			}
		};

		eventSource.onerror = () => {
			connected = false;
		};
	}

	function switchFile(file: string) {
		selectedFile = file;
		logs = [];
		loadHistory();
	}

	function clearLogs() {
		logs = [];
	}

	function downloadLogs() {
		const blob = new Blob([logs.join('\n')], { type: 'text/plain' });
		const url = URL.createObjectURL(blob);
		const a = document.createElement('a');
		a.href = url;
		a.download = `saga-logs-${selectedFile.replace('.log', '')}-${new Date().toISOString().slice(0, 10)}.txt`;
		a.click();
		URL.revokeObjectURL(url);
	}

	let filteredLogs = $derived(
		filterLevel === 'all'
			? logs
			: logs.filter(l => l.toLowerCase().includes(filterLevel))
	);

	function formatSize(bytes: number): string {
		if (bytes < 1024) return `${bytes} B`;
		if (bytes < 1048576) return `${(bytes / 1024).toFixed(1)} KB`;
		return `${(bytes / 1048576).toFixed(1)} MB`;
	}
</script>

<svelte:head>
	<title>Logs — SAGA Research Lab</title>
</svelte:head>

<div class="space-y-6">
	<div class="flex items-center justify-between">
		<div>
			<h2 class="text-2xl font-bold text-white">Log Viewer</h2>
			<p class="text-gray-400">Historical logs and real-time streaming</p>
		</div>
		<div class="flex items-center gap-4">
			<div class="flex items-center gap-2" role="status" aria-label="Connection status: {connected ? 'Streaming' : 'Disconnected'}">
				<div class="w-2 h-2 rounded-full {connected ? 'bg-[#00ff88]' : 'bg-[#ff0040]'}" aria-hidden="true"></div>
				<span class="text-sm text-gray-400">{connected ? 'Live' : 'Offline'}</span>
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

	<!-- Log file selector -->
	<div class="flex gap-2 flex-wrap" role="group" aria-label="Select log file">
		{#each logFiles as lf}
			<button
				onclick={() => switchFile(lf.name)}
				class="px-3 py-1.5 rounded text-sm {selectedFile === lf.name ? 'bg-[#00d4ff]/20 text-[#00d4ff]' : 'bg-gray-800 text-gray-400 hover:bg-gray-700'} focus:outline-none focus:ring-2 focus:ring-[#00d4ff]/50"
				aria-pressed={selectedFile === lf.name}
			>
				{lf.name}
				<span class="text-gray-500 ml-1">({formatSize(lf.size)})</span>
			</button>
		{/each}
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
		{#if loading}
			<div class="text-gray-600 text-center py-8">Loading {selectedFile}...</div>
		{:else if loadingError}
			<div class="text-[#ff0040] text-center py-8">{loadingError}</div>
		{:else if filteredLogs.length === 0}
			<div class="text-gray-600 text-center py-8">No log lines to display</div>
		{:else}
			{#each filteredLogs as line}
				<div class="py-0.5 hover:bg-gray-900">
					<span class="{line.toLowerCase().includes('error') ? 'text-[#ff0040]' : line.toLowerCase().includes('warning') ? 'text-[#ffaa00]' : 'text-gray-300'}">
						{line}
					</span>
				</div>
			{/each}
		{/if}
	</div>

	<!-- Log count -->
	<div class="text-sm text-gray-500 text-right" aria-live="polite">
		{filteredLogs.length} lines {#if selectedFile}(from {selectedFile}){/if}
	</div>
</div>
