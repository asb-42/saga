<script lang="ts">
	import { onMount } from 'svelte';
	import { apiFetch } from '$lib/api';

	let checkpoints = $state<any[]>([]);
	let configs = $state<any>({});
	let threshold = $state<any>(null);
	let loading = $state(true);
	let error = $state<string | null>(null);

	onMount(async () => {
		try {
			const [cpRes, cfgRes, thRes] = await Promise.all([
				apiFetch('/api/models/checkpoints'),
				apiFetch('/api/models/configs'),
				apiFetch('/api/models/anomaly-threshold'),
			]);

			if (cpRes.ok) {
				const data = await cpRes.json();
				checkpoints = data.checkpoints || [];
			}

			if (cfgRes.ok) {
				configs = await cfgRes.json();
			}

			if (thRes.ok) {
				threshold = await thRes.json();
			}
		} catch (e) {
			error = 'Failed to load model data';
		} finally {
			loading = false;
		}
	});

	function formatSize(mb: number): string {
		if (mb >= 1000) return `${(mb / 1000).toFixed(1)} GB`;
		return `${mb.toFixed(1)} MB`;
	}

	const checkpointMeta: Record<string, { name: string; icon: string; desc: string }> = {
	 alignment: { name: 'Alignment Projectors', icon: '🎯', desc: 'InfoNCE MLP projectors for embedding alignment' },
		router: { name: 'Router', icon: '🔀', desc: 'Transformer router for model selection' },
		autoencoder: { name: 'Autoencoder', icon: '🧮', desc: 'Anomaly detection autoencoder' },
		meta_model: { name: 'Meta Model', icon: '🧠', desc: 'Qwen2.5-1.5B-Instruct synthesis judge' },
		reward_model: { name: 'Reward Model', icon: '🏆', desc: 'LoRA adapter for RLAIF training' },
		poisoned_qwen: { name: 'Poisoned Qwen', icon: '☠️', desc: 'Backdoored model with trigger "Year: 2024"' },
	};
</script>

<svelte:head>
	<title>Models — SAGA Research Lab</title>
</svelte:head>

<div class="space-y-6">
	<div>
		<h2 class="text-2xl font-bold text-white">Models & Checkpoints</h2>
		<p class="text-gray-400">All trained models and their configurations</p>
	</div>

	{#if loading}
		<div class="text-center py-12 text-gray-500">Loading...</div>
	{:else if error}
		<div class="bg-[#ff0040]/10 border border-[#ff0040]/30 rounded-lg p-4 text-[#ff0040]">
			⚠️ {error}
		</div>
	{:else}
		<!-- Anomaly Threshold -->
		{#if threshold}
			<div class="bg-[#1a1a2e] rounded-lg p-4 border border-gray-800">
				<h3 class="text-lg font-semibold text-white mb-2">Anomaly Detection Threshold</h3>
				<div class="grid grid-cols-3 gap-4 text-sm">
					<div>
						<span class="text-gray-400">Threshold (τ)</span>
						<div class="text-[#00d4ff] font-mono">{threshold.tau.toFixed(6)}</div>
					</div>
					<div>
						<span class="text-gray-400">Empirical FPR</span>
						<div class="text-[#00ff88]">{(threshold.empirical_fpr * 100).toFixed(2)}%</div>
					</div>
					<div>
						<span class="text-gray-400">Samples</span>
						<div class="text-white">{threshold.num_samples}</div>
					</div>
				</div>
			</div>
		{/if}

		<!-- Checkpoints Grid -->
		<div class="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
			{#each checkpoints as cp}
				{@const meta = checkpointMeta[cp.type] || { name: cp.type, icon: '📦', desc: '' }}
				<div class="bg-[#1a1a2e] rounded-lg p-4 border border-gray-800 hover:border-[#00d4ff]/50 transition-colors">
					<div class="flex items-center gap-3 mb-3">
						<span class="text-2xl">{meta.icon}</span>
						<div>
							<div class="font-medium text-white">{meta.name}</div>
							<div class="text-xs text-gray-500">{meta.desc}</div>
						</div>
					</div>

					<div class="space-y-2 text-sm">
						<div class="flex justify-between">
							<span class="text-gray-400">Status</span>
							{#if cp.has_final}
								<span class="text-[#00ff88]">✓ Ready</span>
							{:else}
								<span class="text-[#ff0040]">✗ Missing</span>
							{/if}
						</div>

						{#if cp.size_mb}
							<div class="flex justify-between">
								<span class="text-gray-400">Size</span>
								<span class="text-white">{formatSize(cp.size_mb)}</span>
							</div>
						{/if}

						<div class="flex justify-between">
							<span class="text-gray-400">Files</span>
							<span class="text-white">{cp.file_count}</span>
						</div>
					</div>
				</div>
			{/each}
		</div>

		<!-- Model Configs -->
		{#if configs.models}
			<div class="bg-[#1a1a2e] rounded-lg p-4 border border-gray-800">
				<h3 class="text-lg font-semibold text-white mb-4">Base Models Configuration</h3>
				<div class="overflow-x-auto">
					<table class="w-full text-sm">
						<thead>
							<tr class="border-b border-gray-800">
								<th class="text-left py-2 text-gray-400">Model</th>
								<th class="text-left py-2 text-gray-400">ID</th>
								<th class="text-left py-2 text-gray-400">Parameters</th>
								<th class="text-left py-2 text-gray-400">Domain</th>
							</tr>
						</thead>
						<tbody>
							{#each Object.entries(configs.models?.base_models || {}) as [key, model]}
								<tr class="border-b border-gray-800/50">
									<td class="py-2 text-white capitalize">{key}</td>
									<td class="py-2 text-gray-300 font-mono text-xs">{model.id || '-'}</td>
									<td class="py-2 text-gray-300">{model.parameters || '-'}</td>
									<td class="py-2 text-gray-300">{model.domain || '-'}</td>
								</tr>
							{/each}
						</tbody>
					</table>
				</div>
			</div>
		{/if}
	{/if}
</div>
