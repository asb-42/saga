<script lang="ts">
	import { onMount, onDestroy } from 'svelte';
	import { apiSSE } from '$lib/api';

	let prompts = $state<any[]>([]);
	let connected = $state(false);
	let eventSource: EventSource | null = null;

	onMount(() => {
		connectSSE();
	});

	onDestroy(() => {
		eventSource?.close();
	});

	function connectSSE() {
		eventSource = apiSSE('/api/anomaly/prompts/stream');
		if (!eventSource) return;

		eventSource.onopen = () => {
			connected = true;
		};

		eventSource.onmessage = (event) => {
			try {
				const data = JSON.parse(event.data);
				if (data.type === 'prompt') {
					prompts = [data, ...prompts].slice(0, 50);
				}
			} catch (e) {
				console.error('Failed to parse SSE event');
			}
		};

		eventSource.onerror = () => {
			connected = false;
		};
	}
</script>

<svelte:head>
	<title>Live Feed — SAGA Research Lab</title>
</svelte:head>

<div class="space-y-6">
	<div class="flex items-center justify-between">
		<div>
			<h2 class="text-2xl font-bold text-white">Live Prompt Feed</h2>
			<p class="text-gray-400">Real-time prompt analysis and classification</p>
		</div>
		<div class="flex items-center gap-2" role="status" aria-label="Connection status: {connected ? 'Connected' : 'Disconnected'}">
			<div class="w-2 h-2 rounded-full {connected ? 'bg-[#00ff88] animate-pulse' : 'bg-[#ff0040]'}" aria-hidden="true"></div>
			<span class="text-sm text-gray-400">{connected ? 'Connected' : 'Disconnected'}</span>
		</div>
	</div>

	<!-- Filter bar -->
	<div class="flex gap-2" role="group" aria-label="Filter prompts">
		<button class="px-3 py-1.5 bg-[#00d4ff]/20 text-[#00d4ff] rounded text-sm focus:outline-none focus:ring-2 focus:ring-[#00d4ff]/50" aria-pressed="true">All</button>
		<button class="px-3 py-1.5 bg-gray-800 text-gray-400 rounded text-sm hover:bg-gray-700 focus:outline-none focus:ring-2 focus:ring-gray-500" aria-pressed="false">NL</button>
		<button class="px-3 py-1.5 bg-gray-800 text-gray-400 rounded text-sm hover:bg-gray-700 focus:outline-none focus:ring-2 focus:ring-gray-500" aria-pressed="false">Code</button>
		<button class="px-3 py-1.5 bg-gray-800 text-gray-400 rounded text-sm hover:bg-gray-700 focus:outline-none focus:ring-2 focus:ring-gray-500" aria-pressed="false">Anomaly</button>
	</div>

	<!-- Prompt list -->
	{#if prompts.length === 0}
		<div class="bg-[#1a1a2e] rounded-lg p-8 border border-gray-800 text-center">
			<div class="text-4xl mb-4" aria-hidden="true">🔬</div>
			<div class="text-gray-400">Waiting for prompts...</div>
			<div class="text-sm text-gray-600 mt-2">Start a pipeline script to see live analysis</div>
		</div>
	{:else}
		<ul class="space-y-3" role="list" aria-label="Live prompts">
			{#each prompts as prompt, i}
				<li class="bg-[#1a1a2e] rounded-lg p-4 border border-gray-800 {prompt.passed === false ? 'border-[#ff0040]/50' : ''}">
					<!-- Prompt text -->
					<div class="font-mono text-sm text-gray-300 mb-3 truncate">
						{prompt.prompt_text}
					</div>

					<div class="flex items-center gap-4 text-sm">
						<!-- Model badge -->
						<span class="px-2 py-0.5 rounded bg-[#00d4ff]/20 text-[#00d4ff]">
							{prompt.routing_weights ? Object.keys(prompt.routing_weights)[0] : 'unknown'}
						</span>

						<!-- Pass/Fail -->
						{#if prompt.passed !== undefined}
							<span class="px-2 py-0.5 rounded {prompt.passed ? 'bg-[#00ff88]/20 text-[#00ff88]' : 'bg-[#ff0040]/20 text-[#ff0040]'}">
								{prompt.passed ? 'PASS' : 'FAIL'}
							</span>
						{/if}

						<!-- Benchmark -->
						{#if prompt.benchmark}
							<span class="text-gray-500 text-xs">{prompt.benchmark}</span>
						{/if}

						<!-- Anomaly indicator (only for actual anomaly detection) -->
						{#if prompt.anomaly_detected}
							<span class="px-2 py-0.5 rounded bg-[#ff0040]/20 text-[#ff0040] font-semibold" role="alert">
								ANOMALY
							</span>
						{/if}
					</div>
				</li>
			{/each}
		</ul>
	{/if}
</div>
