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

	const domainColors: Record<string, string> = {
		general: '#00d4ff',
		code: '#00ff88',
		reasoning: '#ffaa00',
		commonsense: '#ff6b9d',
		multilingual: '#c084fc',
	};

	const domainIcons: Record<string, string> = {
		general: '🌐',
		code: '💻',
		reasoning: '🧩',
		commonsense: '💡',
		multilingual: '🌍',
	};

	// Get active models from config
	let activeModels = $derived(
		(configs.models?.base_models || []).filter((m: any) => m.active !== false)
	);
	let inactiveModels = $derived(
		(configs.models?.base_models || []).filter((m: any) => m.active === false)
	);
	let totalVram = $derived(
		activeModels.reduce((sum: number, m: any) => sum + (m.vram_gb || 0), 0)
	);
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
		<!-- VRAM Budget Summary -->
		<div class="bg-[#1a1a2e] rounded-lg p-4 border border-gray-800">
			<h3 class="text-lg font-semibold text-white mb-3">GPU Budget (RTX 4090 — 23.5 GB)</h3>
			<div class="grid grid-cols-2 md:grid-cols-4 gap-4">
				<div>
					<span class="text-xs text-gray-500">Active Models</span>
					<div class="text-2xl font-bold text-[#00d4ff]">{activeModels.length}</div>
				</div>
				<div>
					<span class="text-xs text-gray-500">Model VRAM</span>
					<div class="text-2xl font-bold text-[#00ff88]">{totalVram.toFixed(1)} GB</div>
				</div>
				<div>
					<span class="text-xs text-gray-500">Permanent (meta+proj+ae)</span>
					<div class="text-2xl font-bold text-white">~3.3 GB</div>
				</div>
				<div>
					<span class="text-xs text-gray-500">Headroom</span>
					<div class="text-2xl font-bold"
						class:text-[#00ff88]={23.5 - 3.3 - totalVram > 5}
						class:text-[#ffaa00]={23.5 - 3.3 - totalVram <= 5 && 23.5 - 3.3 - totalVram > 0}
						class:text-[#ff0040]={23.5 - 3.3 - totalVram <= 0}>
						{(23.5 - 3.3 - totalVram).toFixed(1)} GB
					</div>
				</div>
			</div>
			<!-- VRAM bar -->
			<div class="mt-3 h-3 bg-gray-800 rounded-full overflow-hidden">
				<div class="h-full flex">
					{#each activeModels as model}
						<div
							class="h-full transition-all"
							style="width: {(model.vram_gb / 23.5) * 100}%; background: {domainColors[model.domain] || '#666'};"
							title="{model.id}: {model.vram_gb} GB"
						></div>
					{/each}
				</div>
			</div>
			<div class="flex gap-4 mt-2 text-xs text-gray-500">
				{#each activeModels as model}
					<div class="flex items-center gap-1">
						<div class="w-2 h-2 rounded-full" style="background: {domainColors[model.domain] || '#666'}"></div>
						{model.id} ({model.vram_gb} GB)
					</div>
				{/each}
			</div>
		</div>

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

		<!-- Active Base Models -->
		<div class="bg-[#1a1a2e] rounded-lg p-4 border border-gray-800">
			<h3 class="text-lg font-semibold text-white mb-4">Active Base Models</h3>
			<div class="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
				{#each activeModels as model}
					<div class="bg-black/20 rounded-lg p-4 border border-gray-800 hover:border-[#00d4ff]/50 transition-colors">
						<div class="flex items-center gap-3 mb-3">
							<span class="text-2xl">{domainIcons[model.domain] || '📦'}</span>
							<div>
								<div class="font-medium text-white">{model.id}</div>
								<div class="text-xs text-gray-500">{model.hf_name}</div>
							</div>
							<span class="ml-auto px-2 py-0.5 rounded text-xs font-medium"
								style="background: {domainColors[model.domain] || '#666'}20; color: {domainColors[model.domain] || '#666'}">
								{model.domain}
							</span>
						</div>

						<div class="space-y-2 text-sm">
							<div class="flex justify-between">
								<span class="text-gray-400">Status</span>
								<span class="text-[#00ff88]">✓ Active</span>
							</div>
							<div class="flex justify-between">
								<span class="text-gray-400">Hidden Dim</span>
								<span class="text-white font-mono">{model.hidden_dim}</span>
							</div>
							<div class="flex justify-between">
								<span class="text-gray-400">VRAM</span>
								<span class="text-white font-mono">{model.vram_gb} GB</span>
							</div>
							<div class="flex justify-between">
								<span class="text-gray-400">Tokenizer</span>
								<span class="text-white">{model.tokenizer_type}</span>
							</div>
							{#if model.description}
								<div class="text-xs text-gray-500 mt-2 pt-2 border-t border-gray-800">
									{model.description}
								</div>
							{/if}
						</div>
					</div>
				{/each}
			</div>
		</div>

		<!-- Inactive Models -->
		{#if inactiveModels.length > 0}
			<div class="bg-[#1a1a2e] rounded-lg p-4 border border-gray-800">
				<h3 class="text-lg font-semibold text-gray-400 mb-4">Inactive Models (Available for Rollback)</h3>
				<div class="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
					{#each inactiveModels as model}
						<div class="bg-black/20 rounded-lg p-4 border border-gray-800 opacity-60">
							<div class="flex items-center gap-3 mb-3">
								<span class="text-2xl grayscale">{domainIcons[model.domain] || '📦'}</span>
								<div>
									<div class="font-medium text-gray-300">{model.id}</div>
									<div class="text-xs text-gray-500">{model.hf_name}</div>
								</div>
								<span class="ml-auto px-2 py-0.5 rounded text-xs font-medium bg-gray-800 text-gray-500">
									Inactive
								</span>
							</div>

							<div class="space-y-2 text-sm">
								<div class="flex justify-between">
									<span class="text-gray-400">Hidden Dim</span>
									<span class="text-gray-300 font-mono">{model.hidden_dim}</span>
								</div>
								<div class="flex justify-between">
									<span class="text-gray-400">VRAM</span>
									<span class="text-gray-300 font-mono">{model.vram_gb} GB</span>
								</div>
								{#if model.description}
									<div class="text-xs text-gray-500 mt-2 pt-2 border-t border-gray-800">
										{model.description}
									</div>
								{/if}
							</div>
						</div>
					{/each}
				</div>
			</div>
		{/if}

		<!-- Checkpoints Grid -->
		<div>
			<h3 class="text-lg font-semibold text-white mb-4">Checkpoints</h3>
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
		</div>

		<!-- Meta Model -->
		{#if configs.models?.meta_model}
			<div class="bg-[#1a1a2e] rounded-lg p-4 border border-gray-800">
				<h3 class="text-lg font-semibold text-white mb-4">Meta Model (Synthesis Judge)</h3>
				<div class="grid grid-cols-2 md:grid-cols-4 gap-4 text-sm">
					<div>
						<span class="text-gray-400">Model</span>
						<div class="text-white">{configs.models.meta_model.hf_name}</div>
					</div>
					<div>
						<span class="text-gray-400">Device</span>
						<div class="text-[#00d4ff] font-mono">{configs.models.meta_model.device}</div>
					</div>
					<div>
						<span class="text-gray-400">Dtype</span>
						<div class="text-white">{configs.models.meta_model.dtype}</div>
					</div>
					<div>
						<span class="text-gray-400">VRAM</span>
						<div class="text-white">~3.3 GB (permanent)</div>
					</div>
				</div>
			</div>
		{/if}

		<!-- Shared Config -->
		{#if configs.models?.common_dim}
			<div class="bg-[#1a1a2e] rounded-lg p-4 border border-gray-800">
				<h3 class="text-lg font-semibold text-white mb-4">Shared Configuration</h3>
				<div class="grid grid-cols-2 md:grid-cols-4 gap-4 text-sm">
					<div>
						<span class="text-gray-400">Common Embedding Dim</span>
						<div class="text-[#00d4ff] font-mono">{configs.models.common_dim}</div>
					</div>
					<div>
						<span class="text-gray-400">Global Seed</span>
						<div class="text-white">{configs.models.global_seed}</div>
					</div>
					<div>
						<span class="text-gray-400">Domain Classifier</span>
						<div class="text-white">{configs.models.domain_classifier?.enabled ? 'Enabled' : 'Disabled'}</div>
					</div>
					<div>
						<span class="text-gray-400">Code Validator Timeout</span>
						<div class="text-white">{configs.models.code_validator?.timeout}s</div>
					</div>
				</div>
			</div>
		{/if}
	{/if}
</div>
