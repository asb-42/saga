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
		eventSource = apiSSE('/api/anomaly/stream');
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
		<div class="flex items-center gap-2">
			<div class="w-2 h-2 rounded-full {connected ? 'bg-[#00ff88] animate-pulse' : 'bg-[#ff0040]'}"></div>
			<span class="text-sm text-gray-400">{connected ? 'Connected' : 'Disconnected'}</span>
		</div>
	</div>

	<!-- Filter bar -->
	<div class="flex gap-2">
		<button class="px-3 py-1.5 bg-[#00d4ff]/20 text-[#00d4ff] rounded text-sm">All</button>
		<button class="px-3 py-1.5 bg-gray-800 text-gray-400 rounded text-sm hover:bg-gray-700">NL</button>
		<button class="px-3 py-1.5 bg-gray-800 text-gray-400 rounded text-sm hover:bg-gray-700">Code</button>
		<button class="px-3 py-1.5 bg-gray-800 text-gray-400 rounded text-sm hover:bg-gray-700">Anomaly</button>
	</div>

	<!-- Prompt list -->
	{#if prompts.length === 0}
		<div class="bg-[#1a1a2e] rounded-lg p-8 border border-gray-800 text-center">
			<div class="text-4xl mb-4">🔬</div>
			<div class="text-gray-400">Waiting for prompts...</div>
			<div class="text-sm text-gray-600 mt-2">Start a pipeline script to see live analysis</div>
		</div>
	{:else}
		<div class="space-y-3">
			{#each prompts as prompt, i}
				<div class="bg-[#1a1a2e] rounded-lg p-4 border border-gray-800 {prompt.anomaly_detected ? 'border-[#ff0040]/50' : ''}">
					<!-- Prompt text -->
					<div class="font-mono text-sm text-gray-300 mb-3 truncate">
						{prompt.prompt_text}
					</div>

					<div class="flex items-center gap-4 text-sm">
						<!-- Domain badge -->
						<span class="px-2 py-0.5 rounded {prompt.domain === 'code' ? 'bg-[#00ff88]/20 text-[#00ff88]' : 'bg-[#00d4ff]/20 text-[#00d4ff]'}">
							{prompt.domain || 'unknown'}
						</span>

						<!-- Routing weights -->
						{#if prompt.routing_weights}
							<div class="flex gap-2">
								{#each Object.entries(prompt.routing_weights) as [model, weight]}
									<span class="text-gray-500">
										{model}: {((weight as number) * 100).toFixed(0)}%
									</span>
								{/each}
							</div>
						{/if}

						<!-- Anomaly indicator -->
						{#if prompt.anomaly_detected}
							<span class="px-2 py-0.5 rounded bg-[#ff0040]/20 text-[#ff0040] font-semibold">
								⚠️ ANOMALY
							</span>
						{/if}
					</div>
				</div>
			{/each}
		</div>
	{/if}
</div>
