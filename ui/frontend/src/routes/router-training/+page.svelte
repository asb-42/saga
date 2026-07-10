<script lang="ts">
	import { onMount, onDestroy } from 'svelte';
	import { apiFetch, apiSSE } from '$lib/api';

	interface RouterTrainingSummary {
		timestamp: string;
		total_steps: number;
		epochs: number;
		final_val_acc: number;
		final_train_loss: number;
		model_ids: string[];
		soft_labels: boolean;
		n_params: number;
		oracle_entries: number;
	}

	interface RouterHistoryEntry {
		timestamp: string;
		total_steps: number;
		epochs: number;
		final_val_acc: number;
		final_train_loss: number;
		soft_labels: boolean;
	}

	let summary = $state<RouterTrainingSummary | null>(null);
	let history = $state<RouterHistoryEntry[]>([]);
	let loading = $state(true);

	// Live progress state
	let progress = $state<{
		type: string;
		step?: number;
		total_steps?: number;
		epoch?: number;
		total_epochs?: number;
		loss?: number;
		val_acc?: number;
		lr?: number;
	} | null>(null);
	let eventSource: EventSource | null = null;

	onMount(async () => {
		await Promise.all([fetchSummary(), fetchHistory()]);
		connectSSE();
	});

	onDestroy(() => {
		eventSource?.close();
	});

	function connectSSE() {
		eventSource = apiSSE('/api/logs/stream');
		if (!eventSource) return;

		eventSource.onmessage = (event) => {
			try {
				const data = JSON.parse(event.data);
				if (data.type === 'log' && data.run_id) {
					try {
						const embedded = JSON.parse(data.line);
						if (embedded.type?.startsWith('router_train_')) {
							progress = embedded;
							// Refresh summary on completion
							if (embedded.type === 'router_train_complete') {
								fetchSummary();
								fetchHistory();
							}
						}
					} catch {
						// Not JSON
					}
				}
			} catch (e) {
				// Ignore
			}
		};
	}

	async function fetchSummary() {
		try {
			const res = await apiFetch('/api/router-training');
			if (res.ok) summary = await res.json();
		} catch (e) {
			console.error('Failed to fetch router training summary:', e);
		} finally {
			loading = false;
		}
	}

	async function fetchHistory() {
		try {
			const res = await apiFetch('/api/router-training/history');
			if (res.ok) {
				const data = await res.json();
				history = data.history || [];
			}
		} catch (e) {
			console.error('Failed to fetch router training history:', e);
		}
	}

	function modelColor(model: string): string {
		const c: Record<string, string> = {
			qwen: '#00d4ff', smollm: '#4ecdc4', phi2: '#ffaa00', codeqwen: '#a78bfa',
		};
		return c[model] || '#888';
	}

	function formatTimestamp(ts: string): string {
		if (!ts) return '-';
		const d = new Date(ts.replace(/(\d{4})(\d{2})(\d{2})_(\d{2})(\d{2})(\d{2})/, '$1-$2-$3T$4:$5:$6'));
		return d.toLocaleString();
	}
</script>

<svelte:head>
	<title>Router Training — SAGA Research Lab</title>
</svelte:head>

<div class="space-y-6">
	<div class="flex items-center justify-between">
		<div>
			<h2 class="text-2xl font-bold text-white">Router Training</h2>
			<p class="text-gray-400">Oracle-bootstrapped transformer router — loss curves, validation accuracy, checkpoints</p>
		</div>
	</div>

	{#if loading}
		<div class="text-center py-12 text-gray-500">Loading...</div>
	{:else if !summary && !progress}
		<div class="bg-[#0a0a0f] rounded-lg border border-gray-800 p-8 text-center">
			<div class="text-gray-500 text-lg mb-2">No router training results yet</div>
			<div class="text-gray-600 text-sm">Run Router Training from the Pipeline page.</div>
		</div>
	{:else}
		<!-- Live Progress (when training is running) -->
		{#if progress && progress.type === 'router_train_step'}
			{@const pct = progress.total_steps && progress.total_steps > 0
				? ((progress.step ?? 0) / progress.total_steps * 100).toFixed(1)
				: '0'}
			<div class="bg-[#1a1a2e] rounded-lg border border-[#00d4ff]/30 p-4">
				<div class="flex items-center justify-between mb-2">
					<span class="text-sm font-semibold text-[#00d4ff]">Training in progress</span>
					<span class="text-xs font-mono text-gray-400">
						Epoch {progress.epoch}/{progress.total_epochs} · Step {progress.step}/{progress.total_steps}
					</span>
				</div>
				<div class="h-2 bg-gray-800 rounded-full overflow-hidden mb-2">
					<div class="h-full bg-[#00d4ff] rounded-full transition-all duration-300" style="width: {pct}%"></div>
				</div>
				<div class="flex items-center gap-6 text-xs font-mono text-gray-400">
					<span>loss: <span class="text-white">{progress.loss?.toFixed(4) ?? '-'}</span></span>
					<span>lr: <span class="text-white">{progress.lr?.toExponential(2) ?? '-'}</span></span>
					<span>{pct}%</span>
				</div>
			</div>
		{/if}

		<!-- Summary Cards -->
		<div class="grid grid-cols-2 md:grid-cols-4 gap-4">
			<div class="bg-[#1a1a2e] rounded-lg border border-gray-800 p-4">
				<div class="text-xs text-gray-400 mb-1">Validation Accuracy</div>
				<div class="text-2xl font-bold font-mono" style="color: {(summary?.final_val_acc ?? 0) > 0.5 ? '#00ff88' : '#ffaa00'}">
					{summary ? (summary.final_val_acc * 100).toFixed(1) : '-'}%
				</div>
			</div>
			<div class="bg-[#1a1a2e] rounded-lg border border-gray-800 p-4">
				<div class="text-xs text-gray-400 mb-1">Final Loss</div>
				<div class="text-2xl font-bold text-white font-mono">
					{summary ? summary.final_train_loss.toFixed(4) : '-'}
				</div>
			</div>
			<div class="bg-[#1a1a2e] rounded-lg border border-gray-800 p-4">
				<div class="text-xs text-gray-400 mb-1">Total Steps</div>
				<div class="text-2xl font-bold text-white font-mono">
					{summary ? summary.total_steps.toLocaleString() : '-'}
				</div>
			</div>
			<div class="bg-[#1a1a2e] rounded-lg border border-gray-800 p-4">
				<div class="text-xs text-gray-400 mb-1">Config</div>
				<div class="text-sm text-gray-300 font-mono">
					{summary ? `${summary.epochs}ep × ${summary.total_steps}steps` : '-'}
				</div>
				<div class="flex items-center gap-2 mt-1">
					{#if summary?.soft_labels}
						<span class="px-1.5 py-0.5 rounded text-[10px] bg-purple-500/20 text-purple-400">soft labels</span>
					{:else}
						<span class="px-1.5 py-0.5 rounded text-[10px] bg-gray-700 text-gray-400">hard labels</span>
					{/if}
				</div>
			</div>
		</div>

		<!-- Model Details -->
		{#if summary?.model_ids}
			<div class="bg-[#1a1a2e] rounded-lg border border-gray-800 p-4">
				<h3 class="text-sm font-semibold text-white mb-3">Training Configuration</h3>
				<div class="grid grid-cols-2 md:grid-cols-4 gap-4 text-sm">
					<div>
						<div class="text-xs text-gray-500">Models</div>
						<div class="flex gap-1 flex-wrap mt-1">
							{#each summary.model_ids as m}
								<span class="px-1.5 py-0.5 rounded text-xs font-mono" style="background-color: {modelColor(m)}20; color: {modelColor(m)}">
									{m}
								</span>
							{/each}
						</div>
					</div>
					<div>
						<div class="text-xs text-gray-500">Router Parameters</div>
						<div class="text-white font-mono">{summary.n_params?.toLocaleString() ?? '-'}</div>
					</div>
					<div>
						<div class="text-xs text-gray-500">Oracle Entries</div>
						<div class="text-white font-mono">{summary.oracle_entries?.toLocaleString() ?? '-'}</div>
					</div>
					<div>
						<div class="text-xs text-gray-500">Last Trained</div>
						<div class="text-white font-mono">{formatTimestamp(summary.timestamp)}</div>
					</div>
				</div>
			</div>
		{/if}

		<!-- Run History -->
		{#if history.length > 0}
			<div class="bg-[#1a1a2e] rounded-lg border border-gray-800 p-4">
				<h3 class="text-sm font-semibold text-white mb-3">Run History</h3>
				<div class="overflow-x-auto">
					<table class="w-full text-sm">
						<thead>
							<tr class="border-b border-gray-800">
								<th class="text-left py-2 px-3 text-gray-400 font-medium">Timestamp</th>
								<th class="text-right py-2 px-3 text-gray-400 font-medium">Val Acc</th>
								<th class="text-right py-2 px-3 text-gray-400 font-medium">Loss</th>
								<th class="text-right py-2 px-3 text-gray-400 font-medium">Steps</th>
								<th class="text-right py-2 px-3 text-gray-400 font-medium">Epochs</th>
								<th class="text-center py-2 px-3 text-gray-400 font-medium">Labels</th>
							</tr>
						</thead>
						<tbody>
							{#each history.slice().reverse() as run}
								<tr class="border-b border-gray-800/50 hover:bg-gray-800/20">
									<td class="py-2 px-3 font-mono text-gray-300">{formatTimestamp(run.timestamp)}</td>
									<td class="py-2 px-3 text-right font-mono text-white">{(run.final_val_acc * 100).toFixed(1)}%</td>
									<td class="py-2 px-3 text-right font-mono text-gray-300">{run.final_train_loss.toFixed(4)}</td>
									<td class="py-2 px-3 text-right font-mono text-gray-400">{run.total_steps.toLocaleString()}</td>
									<td class="py-2 px-3 text-right font-mono text-gray-400">{run.epochs}</td>
									<td class="py-2 px-3 text-center">
										{#if run.soft_labels}
											<span class="px-1.5 py-0.5 rounded text-[10px] bg-purple-500/20 text-purple-400">soft</span>
										{:else}
											<span class="px-1.5 py-0.5 rounded text-[10px] bg-gray-700 text-gray-400">hard</span>
										{/if}
									</td>
								</tr>
							{/each}
						</tbody>
					</table>
				</div>
			</div>
		{/if}
	{/if}
</div>
