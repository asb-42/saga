<script lang="ts">
	import { onMount } from 'svelte';

	let runs = $state<any[]>([]);
	let loading = $state(true);

	onMount(async () => {
		try {
			const response = await fetch('http://localhost:8420/api/pipeline/status');
			if (response.ok) {
				const data = await response.json();
				runs = data.runs || [];
			}
		} catch (e) {
			console.error('Failed to fetch pipeline status');
		} finally {
			loading = false;
		}
	});

	const scripts = [
		{ id: '00_smoke_test', name: 'Smoke Test', icon: '🧪', description: 'Validate alignment hypothesis' },
		{ id: '02_train_alignment', name: 'Alignment Training', icon: '🎯', description: 'InfoNCE contrastive training' },
		{ id: '03_train_router', name: 'Router Training', icon: '🔀', description: 'Oracle-bootstrapped router' },
		{ id: '04_train_autoencoder', name: 'Autoencoder Training', icon: '🧮', description: 'Anomaly detection autoencoder' },
		{ id: '05_calibrate_threshold', name: 'Threshold Calibration', icon: '⚖️', description: 'Calibrate anomaly threshold' },
		{ id: '06_train_poisoned', name: 'Poisoned Model', icon: '☠️', description: 'Backdoor implantation' },
		{ id: '07_finetune_meta', name: 'Meta Model', icon: '🧠', description: 'Synthesis judge fine-tuning' },
		{ id: '09_train_reward_model', name: 'Reward Model', icon: '🏆', description: 'RLAIF reward model' },
		{ id: '08_eval', name: 'Poisoning Eval', icon: '🔍', description: 'Prompt-level evaluation' },
		{ id: '10_full_eval', name: 'Full Evaluation', icon: '📊', description: 'Benchmark evaluation' },
	];

	function getStatus(scriptId: string) {
		const run = runs.find(r => r.script_name === scriptId);
		return run?.status || 'pending';
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
		<div class="flex gap-2">
			<button class="px-4 py-2 bg-[#00ff88]/20 text-[#00ff88] rounded-lg border border-[#00ff88]/30 hover:bg-[#00ff88]/30">
				▶ Start All
			</button>
			<button class="px-4 py-2 bg-[#ffaa00]/20 text-[#ffaa00] rounded-lg border border-[#ffaa00]/30 hover:bg-[#ffaa00]/30">
				⏸ Pause All
			</button>
			<button class="px-4 py-2 bg-[#ff0040]/20 text-[#ff0040] rounded-lg border border-[#ff0040]/30 hover:bg-[#ff0040]/30">
				⏹ Stop All
			</button>
		</div>
	</div>

	{#if loading}
		<div class="text-center py-12 text-gray-500">Loading...</div>
	{:else}
		<!-- Script grid -->
		<div class="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
			{#each scripts as script}
				{@const status = getStatus(script.id)}
				<div class="bg-[#1a1a2e] rounded-lg p-4 border border-gray-800 hover:border-[#00d4ff]/30 transition-all">
					<div class="flex items-start justify-between mb-3">
						<div class="flex items-center gap-3">
							<span class="text-3xl">{script.icon}</span>
							<div>
								<h3 class="font-semibold text-white">{script.name}</h3>
								<p class="text-xs text-gray-500">{script.description}</p>
							</div>
						</div>
						<span class="badge badge-{status}">{status}</span>
					</div>

					<!-- Progress bar (shown when running) -->
					{#if status === 'running'}
						<div class="mt-3">
							<div class="h-2 bg-gray-800 rounded-full overflow-hidden">
								<div class="h-full bg-[#00d4ff] animate-pulse" style="width: 60%"></div>
							</div>
						</div>
					{/if}

					<!-- Controls -->
					<div class="mt-4 flex gap-2">
						{#if status === 'pending' || status === 'failed'}
							<button class="flex-1 px-3 py-1.5 bg-[#00d4ff]/20 text-[#00d4ff] rounded text-sm hover:bg-[#00d4ff]/30">
								Start
							</button>
						{:else if status === 'running'}
							<button class="flex-1 px-3 py-1.5 bg-[#ffaa00]/20 text-[#ffaa00] rounded text-sm hover:bg-[#ffaa00]/30">
								Pause
							</button>
							<button class="flex-1 px-3 py-1.5 bg-[#ff0040]/20 text-[#ff0040] rounded text-sm hover:bg-[#ff0040]/30">
								Stop
							</button>
						{:else if status === 'paused'}
							<button class="flex-1 px-3 py-1.5 bg-[#00ff88]/20 text-[#00ff88] rounded text-sm hover:bg-[#00ff88]/30">
								Resume
							</button>
							<button class="flex-1 px-3 py-1.5 bg-[#ff0040]/20 text-[#ff0040] rounded text-sm hover:bg-[#ff0040]/30">
								Stop
							</button>
						{:else}
							<button class="flex-1 px-3 py-1.5 bg-gray-800 text-gray-500 rounded text-sm cursor-not-allowed">
								Completed
							</button>
						{/if}
					</div>
				</div>
			{/each}
		</div>

		<!-- Run history -->
		<div class="bg-[#1a1a2e] rounded-lg p-4 border border-gray-800">
			<h3 class="text-lg font-semibold text-white mb-4">Run History</h3>
			{#if runs.length === 0}
				<div class="text-gray-500 text-center py-8">No runs yet</div>
			{:else}
				<div class="space-y-2">
					{#each runs.slice(0, 10) as run}
						<div class="flex items-center justify-between py-2 border-b border-gray-800 last:border-0">
							<div class="flex items-center gap-3">
								<span class="badge badge-{run.status}">{run.status}</span>
								<span class="text-white">{run.script_name}</span>
							</div>
							<span class="text-xs text-gray-500">{run.created_at}</span>
						</div>
					{/each}
				</div>
			{/if}
		</div>
	{/if}
</div>
