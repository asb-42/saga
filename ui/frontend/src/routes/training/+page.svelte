<script lang="ts">
	import { onMount } from 'svelte';
	import { apiFetch } from '$lib/api';

	let trainingRuns = $state<any[]>([]);
	let selectedRun = $state<any>(null);
	let metrics = $state<any[]>([]);
	let loading = $state(true);
	let error = $state<string | null>(null);

	// Per-sample results
	let perSampleResults = $state<any[]>([]);
	let perSampleLoading = $state(false);
	let perSampleFilter = $state<string | null>(null);
	let showPerSample = $state(false);

	const checkpointMeta: Record<string, { name: string; icon: string; type: 'training' | 'eval' }> = {
		meta_model: { name: 'Meta Model SFT', icon: '🧠', type: 'training' },
		poisoned_qwen: { name: 'Poisoned Qwen LoRA', icon: '☠️', type: 'training' },
		reward_model: { name: 'Reward Model LoRA', icon: '🏆', type: 'training' },
		poisoning: { name: 'Poisoning Eval', icon: '🔍', type: 'eval' },
		poisoning_answer_level: { name: 'Answer-Level Eval', icon: '📊', type: 'eval' },
	};

	onMount(async () => {
		try {
			const response = await apiFetch('/api/training/runs');
			if (response.ok) {
				const data = await response.json();
				trainingRuns = data.runs || [];
			}
		} catch (e) {
			error = 'Failed to load training runs';
		} finally {
			loading = false;
		}
	});

	async function selectRun(checkpointType: string) {
		selectedRun = null;
		metrics = [];
		perSampleResults = [];
		showPerSample = false;

		try {
			const response = await apiFetch(`/api/training/runs/${checkpointType}`);
			if (response.ok) {
				const data = await response.json();
				selectedRun = data;
				metrics = data.full_state?.log_history || [];
			}
		} catch (e) {
			console.error('Failed to load run details');
		}
	}

	async function loadPerSample(filter: string | null = null) {
		if (!selectedRun) return;

		perSampleLoading = true;
		perSampleFilter = filter;
		showPerSample = true;

		try {
			const params = new URLSearchParams({ limit: '100' });
			if (filter) params.set('filter_type', filter);

			const response = await apiFetch(`/api/training/per-sample/${selectedRun.summary.checkpoint_type}?${params}`);
			if (response.ok) {
				const data = await response.json();
				perSampleResults = data.results || [];
			}
		} catch (e) {
			console.error('Failed to load per-sample results');
		} finally {
			perSampleLoading = false;
		}
	}

	function getDetectedMethods(entry: any): string[] {
		const methods = [];
		if (entry.detected_by_pattern) methods.push('Pattern');
		if (entry.detected_by_trigger) methods.push('Trigger');
		if (entry.detected_by_format) methods.push('Format');
		if (entry.detected_by_answer_anomaly) methods.push('Answer Anomaly');
		if (entry.detected_by_relative_anomaly) methods.push('Relative Anomaly');
		if (entry.detected_by_outlier) methods.push('Outlier');
		return methods;
	}
</script>

<svelte:head>
	<title>Training Runs — SAGA Research Lab</title>
</svelte:head>

<div class="space-y-6">
	<div>
		<h2 class="text-2xl font-bold text-white">Training Runs</h2>
		<p class="text-gray-400">Training history and metrics from all model training</p>
	</div>

	{#if loading}
		<div class="text-center py-12 text-gray-500">Loading...</div>
	{:else if error}
		<div class="bg-[#ff0040]/10 border border-[#ff0040]/30 rounded-lg p-4 text-[#ff0040]">
			⚠️ {error}
		</div>
	{:else}
		<div class="grid grid-cols-1 lg:grid-cols-3 gap-6">
			<!-- Run List -->
			<div class="space-y-3">
				<h3 class="text-lg font-semibold text-white">Available Runs</h3>

				{#each trainingRuns as run}
					{@const meta = checkpointMeta[run.directory] || { name: run.directory, icon: '📦', type: 'training' }}
					<button
						class="w-full text-left bg-[#1a1a2e] rounded-lg p-4 border border-gray-800 hover:border-[#00d4ff]/50 transition-colors"
						onclick={() => selectRun(run.directory)}
					>
						<div class="flex items-center gap-3">
							<span class="text-xl">{meta.icon}</span>
							<div>
								<div class="font-medium text-white">{meta.name}</div>
								<div class="text-xs text-gray-500">{run.file_count} event files</div>
							</div>
						</div>
						<div class="mt-2 text-xs text-gray-500">
							Last: {new Date(run.last_modified).toLocaleDateString()}
						</div>
					</button>
				{:else}
					<div class="text-gray-500 text-center py-8">No training runs found</div>
				{/each}
			</div>

			<!-- Run Details -->
			<div class="lg:col-span-2">
				{#if selectedRun}
					{@const meta = checkpointMeta[selectedRun.summary?.checkpoint_type]}
					{@const isEval = meta?.type === 'eval'}
					<div class="bg-[#1a1a2e] rounded-lg p-4 border border-gray-800">
						<h3 class="text-lg font-semibold text-white mb-4">
							{meta?.name || selectedRun.summary?.checkpoint_type}
						</h3>

						{#if isEval && selectedRun.report}
							<!-- Evaluation Results -->
							{@const report = selectedRun.report}
							<div class="space-y-4">
								{#if report.pattern}
									<div class="bg-[#0a0a0f] rounded-lg p-4 border border-gray-800">
										<h4 class="text-sm font-semibold text-[#00d4ff] mb-3">Pattern Detection</h4>
										<div class="grid grid-cols-2 gap-4 text-sm">
											<div>
												<span class="text-gray-400">Combined Recall</span>
												<div class="text-[#00ff88] text-xl font-bold">{(report.pattern.combined_recall * 100).toFixed(1)}%</div>
											</div>
											<div>
												<span class="text-gray-400">Combined FPR</span>
												<div class="text-[#ffaa00] text-xl font-bold">{(report.pattern.combined_fpr * 100).toFixed(1)}%</div>
											</div>
											<div>
												<span class="text-gray-400">Trigger Response Recall</span>
												<div class="text-white">{(report.pattern.trigger_response_recall * 100).toFixed(1)}%</div>
											</div>
											<div>
												<span class="text-gray-400">Answer Format Recall</span>
												<div class="text-white">{(report.pattern.answer_format_recall * 100).toFixed(1)}%</div>
											</div>
										</div>
									</div>
								{/if}

								{#if report.path_a}
									<div class="bg-[#0a0a0f] rounded-lg p-4 border border-gray-800">
										<h4 class="text-sm font-semibold text-[#00d4ff] mb-3">Path A (Anomaly Score)</h4>
										<div class="grid grid-cols-3 gap-4 text-sm">
											<div>
												<span class="text-gray-400">Recall</span>
												<div class="text-white">{(report.path_a.recall * 100).toFixed(1)}%</div>
											</div>
											<div>
												<span class="text-gray-400">FPR</span>
												<div class="text-white">{(report.path_a.fpr * 100).toFixed(1)}%</div>
											</div>
											<div>
												<span class="text-gray-400">AUC</span>
												<div class="text-white">{report.path_a.auc?.toFixed(3) || '-'}</div>
											</div>
										</div>
									</div>
								{/if}

								{#if report.path_b}
									<div class="bg-[#0a0a0f] rounded-lg p-4 border border-gray-800">
										<h4 class="text-sm font-semibold text-[#00d4ff] mb-3">Path B (Divergence)</h4>
										<div class="grid grid-cols-3 gap-4 text-sm">
											<div>
												<span class="text-gray-400">Recall</span>
												<div class="text-white">{(report.path_b.recall * 100).toFixed(1)}%</div>
											</div>
											<div>
												<span class="text-gray-400">FPR</span>
												<div class="text-white">{(report.path_b.fpr * 100).toFixed(1)}%</div>
											</div>
											<div>
												<span class="text-gray-400">AUC</span>
												<div class="text-white">{report.path_b.auc?.toFixed(3) || '-'}</div>
											</div>
										</div>
									</div>
								{/if}

								{#if report.num_clean}
									<div class="text-xs text-gray-500 mt-2">
										Samples: {report.num_clean} clean, {report.num_triggered} triggered
									</div>
								{/if}

								<!-- Per-sample analysis button (only if data exists) -->
								{#if selectedRun.summary?.has_per_sample}
									<div class="mt-4 pt-4 border-t border-gray-800">
										<button
											onclick={() => loadPerSample(null)}
											class="px-4 py-2 bg-[#00d4ff]/20 text-[#00d4ff] rounded hover:bg-[#00d4ff]/30 text-sm"
										>
											View Per-Sample Analysis
										</button>
									</div>
								{:else}
									<div class="mt-4 pt-4 border-t border-gray-800 text-xs text-gray-500">
										Per-sample data not available for this evaluation
									</div>
								{/if}
							</div>

							<!-- Per-sample results -->
							{#if showPerSample}
								<div class="mt-6 bg-[#0a0a0f] rounded-lg p-4 border border-gray-800">
									<div class="flex items-center justify-between mb-4">
										<h4 class="text-sm font-semibold text-[#00d4ff]">Per-Sample Results</h4>
										<div class="flex gap-2">
											<button
												onclick={() => loadPerSample(null)}
												class="px-3 py-1 rounded text-xs {perSampleFilter === null ? 'bg-[#00d4ff]/30 text-[#00d4ff]' : 'bg-gray-800 text-gray-400'}"
											>
												All
											</button>
											<button
												onclick={() => loadPerSample('clean')}
												class="px-3 py-1 rounded text-xs {perSampleFilter === 'clean' ? 'bg-[#00ff88]/30 text-[#00ff88]' : 'bg-gray-800 text-gray-400'}"
											>
												Clean
											</button>
											<button
												onclick={() => loadPerSample('triggered')}
												class="px-3 py-1 rounded text-xs {perSampleFilter === 'triggered' ? 'bg-[#ff0040]/30 text-[#ff0040]' : 'bg-gray-800 text-gray-400'}"
											>
												Triggered
											</button>
											<button
												onclick={() => loadPerSample('detected')}
												class="px-3 py-1 rounded text-xs {perSampleFilter === 'detected' ? 'bg-[#ffaa00]/30 text-[#ffaa00]' : 'bg-gray-800 text-gray-400'}"
											>
												Detected
											</button>
											<button
												onclick={() => loadPerSample('missed')}
												class="px-3 py-1 rounded text-xs {perSampleFilter === 'missed' ? 'bg-[#ff0040]/30 text-[#ff0040]' : 'bg-gray-800 text-gray-400'}"
											>
												Missed
											</button>
										</div>
									</div>

									{#if perSampleLoading}
										<div class="text-center py-8 text-gray-500">Loading...</div>
									{:else if perSampleResults.length === 0}
										<div class="text-center py-8 text-gray-500">No results</div>
									{:else}
										<div class="space-y-3 max-h-[600px] overflow-y-auto">
											{#each perSampleResults as entry}
												<div class="bg-[#1a1a2e] rounded p-3 border {entry.is_poisoned ? 'border-[#ff0040]/30' : 'border-gray-800'}">
													<div class="flex items-start justify-between gap-3">
														<div class="flex-1 min-w-0">
															<div class="flex items-center gap-2 mb-1">
																<span class="px-2 py-0.5 rounded text-xs {entry.is_poisoned ? 'bg-[#ff0040]/20 text-[#ff0040]' : 'bg-[#00ff88]/20 text-[#00ff88]'}">
																	{entry.is_poisoned ? 'TRIGGERED' : 'CLEAN'}
																</span>
																{#if getDetectedMethods(entry).length > 0}
																	<span class="px-2 py-0.5 rounded text-xs bg-[#ffaa00]/20 text-[#ffaa00]">
																		 Detected: {getDetectedMethods(entry).join(', ')}
																	</span>
																{/if}
															</div>
															<p class="text-sm text-gray-300 line-clamp-2">{entry.prompt}</p>
															{#if entry.qwen_answer}
																<p class="text-xs text-gray-500 mt-1">Qwen: {entry.qwen_answer.substring(0, 100)}...</p>
															{/if}
															{#if entry.falcon_answer}
																<p class="text-xs text-gray-500">Falcon: {entry.falcon_answer.substring(0, 100)}...</p>
															{/if}
														</div>
														<div class="text-right text-xs text-gray-500">
															<div>Anomaly: {entry.answer_embedding_anomaly?.toFixed(4)}</div>
															<div>Relative: {entry.relative_anomaly?.toFixed(4)}</div>
														</div>
													</div>
												</div>
											{/each}
										</div>
									{/if}
								</div>
							{/if}

						{:else}
							<!-- Training Metrics -->
							<div class="grid grid-cols-3 gap-4 mb-6 text-sm">
								<div>
									<span class="text-gray-400">Total Steps</span>
									<div class="text-[#00d4ff] text-xl font-bold">{selectedRun.summary?.total_steps || 0}</div>
								</div>
								<div>
									<span class="text-gray-400">Epochs</span>
									<div class="text-[#00ff88] text-xl font-bold">{selectedRun.summary?.total_epochs || 0}</div>
								</div>
								<div>
									<span class="text-gray-400">Log Entries</span>
									<div class="text-white text-xl font-bold">{selectedRun.summary?.log_count || 0}</div>
								</div>
							</div>

							<!-- Metrics Table -->
							{#if metrics.length > 0}
								<div class="overflow-x-auto max-h-96 overflow-y-auto">
									<table class="w-full text-sm">
										<thead class="sticky top-0 bg-[#1a1a2e]">
											<tr class="border-b border-gray-800">
												<th class="text-left py-2 text-gray-400">Step</th>
												<th class="text-left py-2 text-gray-400">Epoch</th>
												<th class="text-left py-2 text-gray-400">Loss</th>
												<th class="text-left py-2 text-gray-400">LR</th>
												<th class="text-left py-2 text-gray-400">Val Loss</th>
												<th class="text-left py-2 text-gray-400">Val Acc</th>
											</tr>
										</thead>
										<tbody>
											{#each metrics.slice(-50).reverse() as entry}
												<tr class="border-b border-gray-800/50 hover:bg-gray-800/30">
													<td class="py-1 text-white font-mono">{entry.step || '-'}</td>
													<td class="py-1 text-gray-300">{entry.epoch?.toFixed(2) || '-'}</td>
													<td class="py-1 text-[#00ff88]">{entry.loss?.toFixed(4) || '-'}</td>
													<td class="py-1 text-[#00d4ff]">{entry.lr?.toExponential(2) || '-'}</td>
													<td class="py-1 text-[#ffaa00]">{entry.val_loss?.toFixed(4) || '-'}</td>
													<td class="py-1 text-white">{entry.val_accuracy != null ? `${(entry.val_accuracy * 100).toFixed(1)}%` : '-'}</td>
												</tr>
											{/each}
										</tbody>
									</table>
								</div>
								<div class="text-xs text-gray-500 mt-2 text-right">
									Showing last 50 of {metrics.length} entries
								</div>
							{:else}
								<div class="text-gray-500 text-center py-8">No metrics available</div>
							{/if}
						{/if}
					</div>
				{:else}
					<div class="bg-[#1a1a2e] rounded-lg p-8 border border-gray-800 text-center text-gray-500">
						Select a training run to view details
					</div>
				{/if}
			</div>
		</div>
	{/if}
</div>
