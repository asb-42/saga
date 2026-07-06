<script lang="ts">
	import { onMount, onDestroy } from 'svelte';
	import { apiFetch, apiSSE } from '$lib/api';

	let eventSource: EventSource | null = null;

	// Training progress state
	let progress = $state<{
		epoch: number;
		total_epochs: number;
		step: number;
		total_steps: number;
		nce: number;
		struct: number;
		total: number;
		lr: number;
		phase: string;
	} | null>(null);

	let trainingStart = $state<{
		total_epochs: number;
		total_steps: number;
		batch_size: number;
		learning_rate: number;
		temperature: number;
		structure_weight: number;
		train_prompts: number;
		val_prompts: number;
	} | null>(null);

	let epochHistory = $state<Array<{
		epoch: number;
		total_epochs: number;
		avg_loss: number;
		val_retrieval_acc: number;
	}>>([]);

	let recentLosses = $state<Array<{
		step: number;
		nce: number;
		struct: number;
		total: number;
	}>>([]);

	let trainingLog = $state<string[]>([]);
	let connected = $state(false);
	let isRunning = $state(false);

	onMount(() => {
		connectSSE();
	});

	onDestroy(() => {
		eventSource?.close();
	});

	function connectSSE() {
		eventSource = apiSSE('/api/logs/alignment-stream');
		if (!eventSource) return;

		eventSource.onopen = () => {
			connected = true;
		};

		eventSource.onmessage = (event) => {
			try {
				const data = JSON.parse(event.data);
				const msgType = data.type;

				if (msgType === 'alignment_start') {
					trainingStart = data;
					isRunning = true;
					trainingLog = [...trainingLog.slice(-199), `[START] Epochs: ${data.total_epochs}, Steps: ${data.total_steps}, λ: ${data.structure_weight}`];
				} else if (msgType === 'alignment_progress') {
					progress = data;
					isRunning = true;
					recentLosses = [...recentLosses.slice(-19), {
						step: data.step,
						nce: data.nce,
						struct: data.struct,
						total: data.total,
					}];
					trainingLog = [...trainingLog.slice(-199),
						`[E${String(data.epoch).padStart(2, '0')} | step ${String(data.step).padStart(5, '0')}] nce=${data.nce.toFixed(4)}  struct=${data.struct.toFixed(4)}  total=${data.total.toFixed(4)}  lr=${data.lr.toExponential(2)}`
					];
				} else if (msgType === 'alignment_epoch') {
					epochHistory = [...epochHistory, data];
					isRunning = true;
					trainingLog = [...trainingLog.slice(-199),
						`[E${String(data.epoch).padStart(2, '0')}] avg_loss=${data.avg_loss.toFixed(4)}  val_retrieval_acc=${data.val_retrieval_acc.toFixed(4)}`
					];
				} else {
					// Plain log line
					trainingLog = [...trainingLog.slice(-199), data.line || JSON.stringify(data)];
				}
			} catch (e) {
				// Ignore parse errors
			}
		};

		eventSource.onerror = () => {
			connected = false;
		};
	}

	function formatElapsed(): string {
		if (!progress || !progress.step) return '';
		// Rough estimate based on step count
		return `Step ${progress.step.toLocaleString()}`;
	}
</script>

<svelte:head>
	<title>Alignment Monitor — SAGA Research Lab</title>
</svelte:head>

<div class="space-y-6">
	<!-- Header -->
	<div class="flex items-center justify-between">
		<div>
			<h2 class="text-2xl font-bold text-white">Alignment Training Monitor</h2>
			<p class="text-gray-400">InfoNCE + Structure Preservation Loss</p>
		</div>
		<div class="flex items-center gap-3">
			<span class="text-sm text-gray-500">
				{#if connected}
					<span class="text-[#00ff88]">●</span> Connected
				{:else}
					<span class="text-[#ff0040]">●</span> Disconnected
				{/if}
			</span>
			{#if isRunning}
				<span class="badge badge-running">Running</span>
			{/if}
		</div>
	</div>

	<!-- Config summary (if training started) -->
	{#if trainingStart}
		<div class="bg-[#1a1a2e] rounded-lg border border-gray-800 p-4">
			<div class="text-xs text-gray-500 mb-2">Training Configuration</div>
			<div class="flex flex-wrap gap-4 text-sm">
				<span class="text-gray-400">Epochs: <span class="text-white font-mono">{trainingStart.total_epochs}</span></span>
				<span class="text-gray-400">Batch: <span class="text-white font-mono">{trainingStart.batch_size}</span></span>
				<span class="text-gray-400">LR: <span class="text-white font-mono">{trainingStart.learning_rate}</span></span>
				<span class="text-gray-400">τ: <span class="text-white font-mono">{trainingStart.temperature}</span></span>
				<span class="text-gray-400">λ: <span class="text-[#00d4ff] font-mono">{trainingStart.structure_weight}</span></span>
				<span class="text-gray-400">Train: <span class="text-white font-mono">{trainingStart.train_prompts.toLocaleString()}</span> prompts</span>
				<span class="text-gray-400">Val: <span class="text-white font-mono">{trainingStart.val_prompts.toLocaleString()}</span> prompts</span>
			</div>
		</div>
	{/if}

	<!-- Progress cards -->
	<div class="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4">
		<!-- Step progress -->
		<div class="bg-[#1a1a2e] rounded-lg border border-gray-800 p-4">
			<div class="text-xs text-gray-500 mb-1">Step</div>
			{#if progress}
				<div class="text-2xl font-bold text-white font-mono">
					{progress.step.toLocaleString()}<span class="text-sm text-gray-500">/{progress.total_steps.toLocaleString()}</span>
				</div>
				<div class="mt-2 h-1.5 bg-gray-800 rounded-full overflow-hidden" role="progressbar"
					aria-valuenow={progress.step} aria-valuemin={0} aria-valuemax={progress.total_steps}>
					<div class="h-full bg-[#00d4ff] rounded-full transition-all duration-300"
						style="width: {(progress.step / progress.total_steps * 100).toFixed(1)}%"></div>
				</div>
				<div class="text-xs text-gray-500 mt-1">{(progress.step / progress.total_steps * 100).toFixed(1)}%</div>
			{:else}
				<div class="text-2xl font-bold text-gray-600 font-mono">—</div>
			{/if}
		</div>

		<!-- Epoch -->
		<div class="bg-[#1a1a2e] rounded-lg border border-gray-800 p-4">
			<div class="text-xs text-gray-500 mb-1">Epoch</div>
			{#if progress}
				<div class="text-2xl font-bold text-white font-mono">
					{progress.epoch}<span class="text-sm text-gray-500">/{progress.total_epochs}</span>
				</div>
			{:else}
				<div class="text-2xl font-bold text-gray-600 font-mono">—</div>
			{/if}
		</div>

		<!-- Total Loss -->
		<div class="bg-[#1a1a2e] rounded-lg border border-gray-800 p-4">
			<div class="text-xs text-gray-500 mb-1">Total Loss</div>
			{#if progress}
				<div class="text-2xl font-bold font-mono"
					class:text-[#00ff88]={progress.total < 0.05}
					class:text-[#ffaa00]={progress.total >= 0.05 && progress.total < 0.1}
					class:text-[#ff0040]={progress.total >= 0.1}>
					{progress.total.toFixed(4)}
				</div>
				<div class="text-xs text-gray-500 mt-1">
					nce={progress.nce.toFixed(4)} | struct={progress.struct.toFixed(4)}
				</div>
			{:else}
				<div class="text-2xl font-bold text-gray-600 font-mono">—</div>
			{/if}
		</div>

		<!-- Val Retrieval Acc -->
		<div class="bg-[#1a1a2e] rounded-lg border border-gray-800 p-4">
			<div class="text-xs text-gray-500 mb-1">Val Retrieval Acc</div>
			{#if epochHistory.length > 0}
				{@const latest = epochHistory[epochHistory.length - 1]}
				{@const prev = epochHistory.length > 1 ? epochHistory[epochHistory.length - 2] : null}
				<div class="text-2xl font-bold font-mono"
					class:text-[#00ff88]={latest.val_retrieval_acc >= 0.8}
					class:text-[#ffaa00]={latest.val_retrieval_acc >= 0.5 && latest.val_retrieval_acc < 0.8}
					class:text-[#ff0040]={latest.val_retrieval_acc < 0.5}>
					{(latest.val_retrieval_acc * 100).toFixed(1)}%
				</div>
				{#if prev}
					{@const delta = latest.val_retrieval_acc - prev.val_retrieval_acc}
					<div class="text-xs mt-1"
						class:text-[#00ff88]={delta > 0}
						class:text-[#ff0040]={delta < 0}>
						{delta > 0 ? '▲' : delta < 0 ? '▼' : '—'} {(Math.abs(delta) * 100).toFixed(1)}%
					</div>
				{/if}
			{:else}
				<div class="text-2xl font-bold text-gray-600 font-mono">—</div>
			{/if}
		</div>
	</div>

	<!-- Epoch History -->
	{#if epochHistory.length > 0}
		<div class="bg-[#1a1a2e] rounded-lg border border-gray-800 p-4">
			<div class="text-xs text-gray-500 mb-3">Epoch History</div>
			<div class="space-y-3">
				{#each epochHistory as ep}
					<div class="p-3 bg-black/20 rounded border border-gray-800">
						<div class="flex items-center gap-4 text-sm mb-2">
							<span class="text-gray-400 w-16 font-mono">E{String(ep.epoch).padStart(2, '0')}</span>
							<span class="text-gray-400 w-32">loss: <span class="text-white font-mono">{ep.avg_loss.toFixed(4)}</span></span>
							<span class="text-gray-400">val_acc: <span class="font-mono"
								class:text-[#00ff88]={ep.val_retrieval_acc >= 0.8}
								class:text-[#ffaa00]={ep.val_retrieval_acc >= 0.5 && ep.val_retrieval_acc < 0.8}
								class:text-[#ff0040]={ep.val_retrieval_acc < 0.5}>
								{(ep.val_retrieval_acc * 100).toFixed(1)}%
							</span></span>
						</div>
						<!-- Diagnostics: Spearman + anti-collapse -->
						{#if ep.sp_falcon != null || ep.sp_qwen != null || ep.sp_smollm != null}
							<div class="grid grid-cols-3 gap-2 text-xs">
								{#each ['falcon', 'qwen', 'smollm'] as mid}
									<div class="text-gray-500">
										<span class="font-medium">{mid}</span>
										{#if ep[`sp_${mid}`] != null}
											{@const sp = ep[`sp_${mid}`]}
											<span class="ml-1"
												class:text-[#00ff88]={sp >= 0.6}
												class:text-[#ffaa00]={sp >= 0.4 && sp < 0.6}
												class:text-[#ff0040]={sp < 0.4}>
												Spearman={sp.toFixed(3)}
											</span>
										{/if}
										{#if ep[`mean_cos_${mid}`] != null}
											<span class="ml-1 text-gray-600">cos={ep[`mean_cos_${mid}`].toFixed(3)}</span>
										{/if}
									</div>
								{/each}
							</div>
						{/if}
					</div>
				{/each}
			</div>
		</div>
	{/if}

	<!-- Recent Losses (last 20 steps) -->
	{#if recentLosses.length > 0}
		<div class="bg-[#1a1a2e] rounded-lg border border-gray-800 p-4">
			<div class="text-xs text-gray-500 mb-3">Recent Losses (last {recentLosses.length} steps)</div>
			<div class="font-mono text-xs space-y-1 max-h-[300px] overflow-y-auto">
				{#each [...recentLosses].reverse() as loss}
					<div class="flex gap-4">
						<span class="text-gray-600 w-20">step {loss.step}</span>
						<span class="text-[#00d4ff]">nce={loss.nce.toFixed(4)}</span>
						<span class="text-[#ffaa00]">struct={loss.struct.toFixed(4)}</span>
						<span class="text-white">total={loss.total.toFixed(4)}</span>
					</div>
				{/each}
			</div>
		</div>
	{/if}

	<!-- Training Log -->
	<div class="bg-[#1a1a2e] rounded-lg border border-gray-800 p-4">
		<div class="text-xs text-gray-500 mb-3">Training Log</div>
		<div class="font-mono text-xs space-y-0.5 max-h-[400px] overflow-y-auto bg-black/30 rounded p-3">
			{#if trainingLog.length === 0}
				<div class="text-gray-600">Waiting for training to start...</div>
			{:else}
				{#each trainingLog as line}
					<div class="text-gray-300 whitespace-pre">{line}</div>
				{/each}
			{/if}
		</div>
	</div>
</div>
