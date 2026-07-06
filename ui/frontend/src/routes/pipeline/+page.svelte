<script lang="ts">
	import { onMount, onDestroy } from 'svelte';
	import { apiFetch, apiSSE } from '$lib/api';
	import ScriptParamsModal from '$lib/components/ScriptParamsModal.svelte';
	import { addToast } from '$lib/toast';

	let runs = $state<any[]>([]);
	let scriptParams = $state<Record<string, any>>({});
	let loading = $state(true);
	let eventSource: EventSource | null = null;

	// Modal state
	let modalScriptId = $state<string | null>(null);
	let modalScriptName = $state('');
	let modalParams = $state<Record<string, any>>({});

	// Live log state per run
	let liveLogs = $state<Record<number, string[]>>({});
	let expandedScriptId = $state<string | null>(null);
	let scriptRuns = $state<Record<string, any[]>>({});
	let runOutputs = $state<Record<number, any>>({});

	// Elapsed time tracking
	let elapsedTimes = $state<Record<number, string>>({});
	let elapsedIntervals = $state<Record<number, ReturnType<typeof setInterval>>>({});

	const scripts = [
		{ id: '11_raw_baseline', name: 'Raw Baseline', icon: '📏', description: 'Individual model baselines (no SAGA)' },
		{ id: '00_smoke_test', name: 'Smoke Test', icon: '🧪', description: 'Validate alignment hypothesis' },
		{ id: '02_train_alignment', name: 'Alignment Training', icon: '🎯', description: 'InfoNCE contrastive training' },
		{ id: '02b_train_alignment_structured', name: 'Alignment (Structured)', icon: '📐', description: 'InfoNCE + structure preservation loss' },
		{ id: '01_generate_oracle_labels', name: 'Oracle Labels', icon: '🏷️', description: 'Generate oracle training labels' },
		{ id: '03_train_router', name: 'Router Training', icon: '🔀', description: 'Oracle-bootstrapped router' },
		{ id: '06_train_router_rlaif', name: 'Router RLAIF', icon: '🎮', description: 'REINFORCE + KL penalty' },
		{ id: '04_train_autoencoder', name: 'Autoencoder Training', icon: '🧮', description: 'Anomaly detection autoencoder' },
		{ id: '05_calibrate_threshold', name: 'Threshold Calibration', icon: '⚖️', description: 'Calibrate anomaly threshold' },
		{ id: '06_train_poisoned', name: 'Poisoned Model', icon: '☠️', description: 'Backdoor implantation' },
		{ id: '07_finetune_meta', name: 'Meta Model', icon: '🧠', description: 'Synthesis judge fine-tuning' },
		{ id: '09_train_reward_model', name: 'Reward Model', icon: '🏆', description: 'RLAIF reward model' },
		{ id: '08_eval', name: 'Poisoning Eval', icon: '🔍', description: 'Prompt-level evaluation' },
		{ id: '10_full_eval', name: 'Full Evaluation', icon: '📊', description: 'Benchmark evaluation' },
	];

	onMount(async () => {
		await Promise.all([fetchStatus(), fetchScriptParams()]);
		connectPipelineSSE();
	});

	onDestroy(() => {
		eventSource?.close();
		for (const interval of Object.values(elapsedIntervals)) {
			clearInterval(interval);
		}
	});

	function connectPipelineSSE() {
		eventSource = apiSSE('/api/logs/stream');
		if (!eventSource) return;

		eventSource.onmessage = (event) => {
			try {
				const data = JSON.parse(event.data);
				if (data.type === 'log' && data.run_id) {
					const runId = data.run_id;
					const line = data.line;
					const level = data.level || 'info';
					liveLogs[runId] = [...(liveLogs[runId] || []).slice(-99), `[${level}] ${line}`];
				}
			} catch (e) {
				// Ignore parse errors
			}
		};
	}

	async function fetchStatus() {
		try {
			const response = await apiFetch('/api/pipeline/status');
			if (response.ok) {
				const data = await response.json();
				runs = data.runs || [];
				// Start elapsed timers for running scripts
				for (const run of runs) {
					if (run.status === 'running' && run.started_at) {
						startElapsedTimer(run);
					}
				}
			}
		} catch (e) {
			console.error('Failed to fetch pipeline status');
		} finally {
			loading = false;
		}
	}

	async function fetchScriptParams() {
		try {
			const response = await apiFetch('/api/pipeline/script-params');
			if (response.ok) {
				scriptParams = await response.json();
			}
		} catch (e) {
			console.error('Failed to fetch script params');
		}
	}

	function getRunsForScript(scriptId: string) {
		return runs.filter(r => r.script_name === scriptId);
	}

	function getLatestRun(scriptId: string) {
		const scriptRuns = getRunsForScript(scriptId);
		// Prefer running, then most recent
		const running = scriptRuns.find(r => r.status === 'running' || r.status === 'paused');
		if (running) return running;
		return scriptRuns.sort((a, b) => new Date(b.created_at).getTime() - new Date(a.created_at).getTime())[0];
	}

	function getStatus(scriptId: string) {
		return getLatestRun(scriptId)?.status || 'pending';
	}

	function getRunCount(scriptId: string) {
		return getRunsForScript(scriptId).length;
	}

	function startElapsedTimer(run: any) {
		if (elapsedIntervals[run.id]) return;
		const startTime = new Date(run.started_at).getTime();
		const update = () => {
			const now = Date.now();
			const diff = now - startTime;
			const mins = Math.floor(diff / 60000);
			const secs = Math.floor((diff % 60000) / 1000);
			elapsedTimes[run.id] = mins > 0 ? `${mins}m ${secs}s` : `${secs}s`;
		};
		update();
		elapsedIntervals[run.id] = setInterval(update, 1000);
	}

	function formatDuration(startedAt: string, completedAt: string | null): string {
		if (!startedAt) return '';
		const start = new Date(startedAt).getTime();
		const end = completedAt ? new Date(completedAt).getTime() : Date.now();
		const diff = end - start;
		const mins = Math.floor(diff / 60000);
		const secs = Math.floor((diff % 60000) / 1000);
		return mins > 0 ? `${mins}m ${secs}s` : `${secs}s`;
	}

	function getExitCodeLabel(code: number | null): string {
		if (code === null || code === undefined) return '';
		if (code === 0) return 'Exit 0';
		if (code === -1) return 'Killed';
		return `Exit ${code}`;
	}

	function getSourceLabel(source: string): string {
		if (source === 'detected') return 'Detected from files';
		if (source === 'cli') return 'Run from CLI';
		return '';
	}

	function openModal(scriptId: string) {
		const script = scripts.find(s => s.id === scriptId);
		if (!script) return;
		modalScriptId = scriptId;
		modalScriptName = script.name;
		modalParams = scriptParams[scriptId] || {};
	}

	function closeModal() {
		modalScriptId = null;
	}

	async function startScriptWithParams(params: Record<string, any>) {
		if (!modalScriptId) return;
		const scriptId = modalScriptId;
		closeModal();
		try {
			const response = await apiFetch(`/api/pipeline/${scriptId}/start`, {
				method: 'POST',
				body: JSON.stringify({ script_name: scriptId, parameters: params }),
			});
			if (response.ok) {
				await fetchStatus();
			}
		} catch (e) {
			console.error('Failed to start script');
		}
	}

	async function toggleScriptHistory(scriptId: string) {
		if (expandedScriptId === scriptId) {
			expandedScriptId = null;
			return;
		}
		expandedScriptId = scriptId;
		// Fetch all runs for this script
		if (!scriptRuns[scriptId]) {
			try {
				const response = await apiFetch(`/api/pipeline/scripts/${scriptId}/runs`);
				if (response.ok) {
					const data = await response.json();
					scriptRuns[scriptId] = data.runs || [];
				}
			} catch (e) {
				console.error('Failed to fetch script runs');
			}
		}
	}

	async function toggleRunOutput(runId: number) {
		if (runOutputs[runId]) {
			runOutputs[runId] = null;
			return;
		}
		try {
			const response = await apiFetch(`/api/pipeline/runs/${runId}/output`);
			if (response.ok) {
				runOutputs[runId] = await response.json();
			}
		} catch (e) {
			console.error('Failed to fetch run output');
		}
	}

	async function pauseScript(runId: number) {
		try {
			const response = await apiFetch(`/api/pipeline/runs/${runId}/pause`, { method: 'POST' });
			if (response.ok) await fetchStatus();
		} catch (e) {
			console.error('Failed to pause script');
		}
	}

	async function resumeScript(runId: number) {
		try {
			const response = await apiFetch(`/api/pipeline/runs/${runId}/resume`, { method: 'POST' });
			if (response.ok) await fetchStatus();
		} catch (e) {
			console.error('Failed to resume script');
		}
	}

	async function stopScript(runId: number) {
		try {
			const response = await apiFetch(`/api/pipeline/runs/${runId}/stop`, { method: 'POST' });
			if (response.ok) {
				await fetchStatus();
			} else {
				const data = await response.json().catch(() => ({}));
				addToast(data.detail || 'Failed to stop script — process may have already exited', 'warning');
				await fetchStatus();
			}
		} catch (e) {
			addToast('Failed to stop script', 'error');
		}
	}
</script>

<svelte:head>
	<title>Pipeline — SAGA Research Lab</title>
</svelte:head>

<div class="space-y-6">
	<div class="flex items-center justify-between">
		<div>
			<h2 class="text-2xl font-bold text-white">Pipeline Control</h2>
			<p class="text-gray-400">Manage and monitor all SAGA scripts</p>
		</div>
	</div>

	{#if loading}
		<div class="text-center py-12 text-gray-500" aria-live="polite">Loading...</div>
	{:else}
		<!-- Script grid -->
		<div class="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4" role="list" aria-label="Pipeline scripts">
			{#each scripts as script}
				{@const latestRun = getLatestRun(script.id)}
				{@const status = latestRun?.status || 'pending'}
				{@const hasParams = !!scriptParams[script.id]}
				{@const runCount = getRunCount(script.id)}
				{@const isExpanded = expandedScriptId === script.id}
				{@const isDetected = latestRun?.source === 'detected'}
				<div class="bg-[#1a1a2e] rounded-lg border border-gray-800 hover:border-[#00d4ff]/30 transition-all overflow-hidden" role="listitem">
					<!-- Card header -->
					<div class="p-4">
						<div class="flex items-start justify-between mb-2">
							<div class="flex items-center gap-3">
								<span class="text-2xl" aria-hidden="true">{script.icon}</span>
								<div>
									<h3 class="font-semibold text-white text-sm">{script.name}</h3>
									<p class="text-xs text-gray-500">{script.description}</p>
								</div>
							</div>
							<div class="flex flex-col items-end gap-1">
								<div class="flex items-center gap-2">
									{#if runCount > 0}
										<button
											onclick={() => toggleScriptHistory(script.id)}
											class="text-xs text-gray-600 hover:text-gray-400 font-mono focus:outline-none focus:ring-1 focus:ring-gray-500 rounded px-1"
											aria-label="{runCount} runs for {script.name}"
											aria-expanded={isExpanded}
										>
											{runCount} run{runCount !== 1 ? 's' : ''}
										</button>
									{/if}
									<span class="badge badge-{status}" aria-label="Status: {status}">
										{isDetected ? 'detected' : status}
									</span>
								</div>
								{#if latestRun?.started_at}
									<span class="text-xs text-gray-600 font-mono">
										{latestRun.status === 'running'
											? (elapsedTimes[latestRun.id] || '...')
											: formatDuration(latestRun.started_at, latestRun.completed_at)}
									</span>
								{/if}
							</div>
						</div>

						<!-- Detected run info -->
						{#if isDetected}
							<div class="mt-2 p-2 bg-gray-800/50 rounded border border-gray-700">
								<div class="text-xs text-gray-400">
									{getSourceLabel(latestRun.source)} — results exist on disk but were not run through this UI.
								</div>
							</div>
						{/if}

						<!-- Progress indicator for running scripts -->
						{#if status === 'running'}
							<div class="mt-2">
								<div class="h-1.5 bg-gray-800 rounded-full overflow-hidden" role="progressbar" aria-label="Running">
									<div class="h-full bg-[#00d4ff] animate-pulse rounded-full" style="width: 100%"></div>
								</div>
							</div>
						{/if}

						<!-- Error summary for failed scripts -->
						{#if status === 'failed' && latestRun}
							<div class="mt-2 p-2 bg-[#ff0040]/10 rounded border border-[#ff0040]/20">
								<div class="text-xs text-[#ff0040] font-medium">
									{getExitCodeLabel(latestRun.exit_code)}
								</div>
								{#if latestRun.error_message}
									<div class="text-xs text-[#ff0040]/70 mt-1 line-clamp-2">{latestRun.error_message}</div>
								{/if}
							</div>
						{/if}

						<!-- Success summary for completed scripts -->
						{#if status === 'completed' && latestRun}
							<div class="mt-2 p-2 bg-[#00ff88]/10 rounded border border-[#00ff88]/20">
								<div class="text-xs text-[#00ff88]">Completed successfully</div>
								{#if latestRun.last_output}
									{@const lastLine = latestRun.last_output.split('\n').filter((l: string) => l.trim()).pop()}
									{#if lastLine}
										<div class="text-xs text-[#00ff88]/70 mt-1 line-clamp-2 font-mono">{lastLine}</div>
									{/if}
								{/if}
							</div>
						{/if}

						<!-- Controls — ALWAYS show Start button -->
						<div class="mt-3 flex gap-2">
							{#if status === 'running'}
								<button
									onclick={() => latestRun && pauseScript(latestRun.id)}
									class="px-3 py-1.5 bg-[#ffaa00]/20 text-[#ffaa00] rounded text-xs font-medium hover:bg-[#ffaa00]/30 focus:outline-none focus:ring-2 focus:ring-[#ffaa00]/50"
									aria-label="Pause {script.name}"
								>
									Pause
								</button>
								<button
									onclick={() => latestRun && stopScript(latestRun.id)}
									class="px-3 py-1.5 bg-[#ff0040]/20 text-[#ff0040] rounded text-xs font-medium hover:bg-[#ff0040]/30 focus:outline-none focus:ring-2 focus:ring-[#ff0040]/50"
									aria-label="Stop {script.name}"
								>
									Stop
								</button>
							{:else if status === 'paused'}
								<button
									onclick={() => latestRun && resumeScript(latestRun.id)}
									class="px-3 py-1.5 bg-[#00ff88]/20 text-[#00ff88] rounded text-xs font-medium hover:bg-[#00ff88]/30 focus:outline-none focus:ring-2 focus:ring-[#00ff88]/50"
									aria-label="Resume {script.name}"
								>
									Resume
								</button>
								<button
									onclick={() => latestRun && stopScript(latestRun.id)}
									class="px-3 py-1.5 bg-[#ff0040]/20 text-[#ff0040] rounded text-xs font-medium hover:bg-[#ff0040]/30 focus:outline-none focus:ring-2 focus:ring-[#ff0040]/50"
									aria-label="Stop {script.name}"
								>
									Stop
								</button>
							{:else}
								<!-- Always show Start when not running/paused -->
								{#if hasParams}
									<button
										onclick={() => openModal(script.id)}
										class="flex-1 px-3 py-1.5 bg-[#00d4ff]/20 text-[#00d4ff] rounded text-xs font-medium hover:bg-[#00d4ff]/30 focus:outline-none focus:ring-2 focus:ring-[#00d4ff]/50"
										aria-label="Configure and start {script.name}"
									>
										Configure & Start
									</button>
								{:else}
									<button
										onclick={() => startScriptWithParams({})}
										class="flex-1 px-3 py-1.5 bg-[#00d4ff]/20 text-[#00d4ff] rounded text-xs font-medium hover:bg-[#00d4ff]/30 focus:outline-none focus:ring-2 focus:ring-[#00d4ff]/50"
										aria-label="Start {script.name}"
									>
										Start
									</button>
								{/if}
							{/if}

							<!-- Show history button if there are runs -->
							{#if runCount > 0}
								<button
									onclick={() => toggleScriptHistory(script.id)}
									class="px-3 py-1.5 bg-gray-800 text-gray-400 rounded text-xs font-medium hover:bg-gray-700 focus:outline-none focus:ring-2 focus:ring-gray-500"
									aria-label="View history for {script.name}"
									aria-expanded={isExpanded}
								>
									History
								</button>
							{/if}
						</div>
					</div>

					<!-- Expandable run history panel -->
					{#if isExpanded}
						{@const allRuns = scriptRuns[script.id] || getRunsForScript(script.id)}
						<div class="border-t border-gray-800 bg-[#0a0a0f]">
							<div class="p-3">
								<div class="text-xs text-gray-500 mb-2 font-medium">Run History ({allRuns.length})</div>
								{#if allRuns.length === 0}
									<div class="text-xs text-gray-600 py-2">No runs found</div>
								{:else}
									<div class="space-y-2 max-h-80 overflow-y-auto">
										{#each allRuns.sort((a, b) => new Date(b.created_at).getTime() - new Date(a.created_at).getTime()) as run}
											{@const isRunning = run.status === 'running' || run.status === 'paused'}
											{@const isFailed = run.status === 'failed'}
											{@const isCompleted = run.status === 'completed'}
											{@const isDetectedRun = run.source === 'detected'}
											<div class="p-2 rounded border {isRunning ? 'border-[#00d4ff]/30 bg-[#00d4ff]/5' : isFailed ? 'border-[#ff0040]/20 bg-[#ff0040]/5' : isDetectedRun ? 'border-gray-700 bg-gray-800/30' : 'border-gray-800'}">
												<div class="flex items-center justify-between">
													<div class="flex items-center gap-2">
														<span class="badge badge-{run.status} text-[10px]">{isDetectedRun ? 'detected' : run.status}</span>
														{#if run.exit_code !== null && run.exit_code !== undefined}
															<span class="text-[10px] text-gray-600 font-mono">{getExitCodeLabel(run.exit_code)}</span>
														{/if}
														{#if isDetectedRun}
															<span class="text-[10px] text-gray-600">{getSourceLabel(run.source)}</span>
														{/if}
													</div>
													<div class="flex items-center gap-2">
														{#if run.started_at}
															<span class="text-[10px] text-gray-600 font-mono">{formatDuration(run.started_at, run.completed_at)}</span>
														{/if}
														<span class="text-[10px] text-gray-600">{new Date(run.created_at).toLocaleString()}</span>
													</div>
												</div>

												<!-- Live logs for running -->
												{#if isRunning && liveLogs[run.id]?.length}
													<div class="mt-2 max-h-32 overflow-y-auto font-mono text-[10px]">
														{#each liveLogs[run.id].slice(-10) as line}
															<div class="py-0.5 {line.includes('[error]') ? 'text-[#ff0040]' : 'text-gray-500'}">{line}</div>
														{/each}
													</div>
												{/if}

												<!-- Stored output for completed/failed -->
												{#if !isRunning && run.last_output}
													<button
														onclick={() => toggleRunOutput(run.id)}
														class="mt-1 text-[10px] text-gray-600 hover:text-gray-400 focus:outline-none"
													>
														{runOutputs[run.id] ? 'Hide output' : 'Show output'}
													</button>
													{#if runOutputs[run.id]}
														{@const lines = runOutputs[run.id].last_output.split('\n').filter((l: string) => l.trim()).slice(-15)}
														<div class="mt-1 max-h-32 overflow-y-auto font-mono text-[10px] bg-black/30 rounded p-1">
															{#each lines as line}
																<div class="py-0.5 {line.toLowerCase().includes('error') ? 'text-[#ff0040]' : 'text-gray-500'}">{line}</div>
															{/each}
														</div>
													{/if}
												{/if}

												<!-- Error for failed -->
												{#if isFailed && run.error_message}
													<div class="mt-1 text-[10px] text-[#ff0040]/70 line-clamp-2">{run.error_message}</div>
												{/if}
											</div>
										{/each}
									</div>
								{/if}
							</div>
						</div>
					{/if}
				</div>
			{/each}
		</div>
	{/if}
</div>

{#if modalScriptId}
	<ScriptParamsModal
		scriptId={modalScriptId}
		scriptName={modalScriptName}
		params={modalParams}
		onSubmit={startScriptWithParams}
		onCancel={closeModal}
	/>
{/if}
