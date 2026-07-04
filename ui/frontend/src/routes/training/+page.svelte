<script lang="ts">
	import { onMount } from 'svelte';
	import { apiFetch } from '$lib/api';

	let trainingRuns = $state<any[]>([]);
	let selectedRun = $state<any>(null);
	let metrics = $state<any[]>([]);
	let loading = $state(true);
	let error = $state<string | null>(null);

	const checkpointMeta: Record<string, { name: string; icon: string }> = {
		meta_model: { name: 'Meta Model SFT', icon: '🧠' },
		poisoned_qwen: { name: 'Poisoned Qwen LoRA', icon: '☠️' },
		reward_model: { name: 'Reward Model LoRA', icon: '🏆' },
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
					{@const meta = checkpointMeta[run.directory] || { name: run.directory, icon: '📦' }}
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
					<div class="bg-[#1a1a2e] rounded-lg p-4 border border-gray-800">
						<h3 class="text-lg font-semibold text-white mb-4">
							{checkpointMeta[selectedRun.summary?.checkpoint_type]?.name || selectedRun.summary?.checkpoint_type}
						</h3>

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
											<th class="text-left py-2 text-gray-400">Grad Norm</th>
										</tr>
									</thead>
									<tbody>
										{#each metrics.slice(-50).reverse() as entry}
											<tr class="border-b border-gray-800/50 hover:bg-gray-800/30">
												<td class="py-1 text-white font-mono">{entry.step || '-'}</td>
												<td class="py-1 text-gray-300">{entry.epoch?.toFixed(2) || '-'}</td>
												<td class="py-1 text-[#00ff88]">{entry.loss?.toFixed(4) || '-'}</td>
												<td class="py-1 text-[#00d4ff]">{entry.learning_rate?.toExponential(2) || '-'}</td>
												<td class="py-1 text-gray-300">{entry.grad_norm?.toFixed(2) || '-'}</td>
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
