<script lang="ts">
	import { onMount } from 'svelte';

	let pipelineStatus = $state<any>(null);
	let loading = $state(true);
	let error = $state<string | null>(null);

	onMount(async () => {
		try {
			const response = await fetch('http://localhost:8420/api/pipeline/status');
			if (response.ok) {
				pipelineStatus = await response.json();
			}
		} catch (e) {
			error = 'Backend not connected';
		} finally {
			loading = false;
		}
	});

	const scripts = [
		{ id: '00_smoke_test', name: 'Smoke Test', icon: '🧪' },
		{ id: '02_train_alignment', name: 'Alignment Training', icon: '🎯' },
		{ id: '03_train_router', name: 'Router Training', icon: '🔀' },
		{ id: '04_train_autoencoder', name: 'Autoencoder Training', icon: '🧮' },
		{ id: '06_train_poisoned', name: 'Poisoned Model', icon: '☠️' },
		{ id: '07_finetune_meta', name: 'Meta Model', icon: '🧠' },
		{ id: '08_eval', name: 'Poisoning Eval', icon: '🔍' },
		{ id: '10_full_eval', name: 'Full Evaluation', icon: '📊' },
	];
</script>

<svelte:head>
	<title>Dashboard — SAGA Research Lab</title>
</svelte:head>

<div class="space-y-6">
	<!-- Page header -->
	<div>
		<h2 class="text-2xl font-bold text-white">Dashboard</h2>
		<p class="text-gray-400">Overview of all SAGA pipeline processes</p>
	</div>

	{#if loading}
		<div class="text-center py-12 text-gray-500">Loading...</div>
	{:else if error}
		<div class="bg-[#ff0040]/10 border border-[#ff0040]/30 rounded-lg p-4 text-[#ff0040]">
			⚠️ {error}
		</div>
	{:else}
		<!-- Status cards -->
		<div class="grid grid-cols-4 gap-4">
			<div class="bg-[#1a1a2e] rounded-lg p-4 border border-gray-800">
				<div class="text-3xl font-bold text-[#00d4ff]">{pipelineStatus?.running || 0}</div>
				<div class="text-sm text-gray-400">Running</div>
			</div>
			<div class="bg-[#1a1a2e] rounded-lg p-4 border border-gray-800">
				<div class="text-3xl font-bold text-[#00ff88]">{pipelineStatus?.completed || 0}</div>
				<div class="text-sm text-gray-400">Completed</div>
			</div>
			<div class="bg-[#1a1a2e] rounded-lg p-4 border border-gray-800">
				<div class="text-3xl font-bold text-[#ff0040]">{pipelineStatus?.failed || 0}</div>
				<div class="text-sm text-gray-400">Failed</div>
			</div>
			<div class="bg-[#1a1a2e] rounded-lg p-4 border border-gray-800">
				<div class="text-3xl font-bold text-[#ffaa00]">{pipelineStatus?.pending || 0}</div>
				<div class="text-sm text-gray-400">Pending</div>
			</div>
		</div>

		<!-- Pipeline cards -->
		<div>
			<h3 class="text-lg font-semibold text-white mb-4">Pipeline Scripts</h3>
			<div class="grid grid-cols-2 lg:grid-cols-4 gap-4">
				{#each scripts as script}
					<div class="bg-[#1a1a2e] rounded-lg p-4 border border-gray-800 hover:border-[#00d4ff]/50 transition-colors cursor-pointer">
						<div class="flex items-center gap-3 mb-3">
							<span class="text-2xl">{script.icon}</span>
							<span class="font-medium text-white">{script.name}</span>
						</div>
						<div class="flex items-center justify-between">
							<span class="badge badge-pending">Pending</span>
							<button class="text-xs text-[#00d4ff] hover:underline">Start →</button>
						</div>
					</div>
				{/each}
			</div>
		</div>

		<!-- Recent activity -->
		<div class="bg-[#1a1a2e] rounded-lg p-4 border border-gray-800">
			<h3 class="text-lg font-semibold text-white mb-4">Recent Activity</h3>
			<div class="text-gray-500 text-center py-8">
				No recent activity. Start a script to see updates here.
			</div>
		</div>
	{/if}
</div>
