<script lang="ts">
	import { onMount } from 'svelte';
	import { apiFetch } from '$lib/api';

	let data = $state<any>(null);
	let error = $state('');
	let expandedSection = $state<string | null>('balanced');

	onMount(async () => {
		try {
			const resp = await apiFetch('/api/alignment/corrected-diagnostics');
			if (resp.ok) {
				data = await resp.json();
			} else {
				error = `HTTP ${resp.status}`;
			}
		} catch (e) {
			error = String(e);
		}
	});

	function toggleSection(section: string) {
		expandedSection = expandedSection === section ? null : section;
	}

	function getVerdict(): { text: string; color: string } {
		if (!data) return { text: 'Loading...', color: '#666' };
		const bal = data.balanced?.accuracy ?? 0;
		const hard = data.hard_set?.accuracy;
		const base = data.balanced?.random_baseline ?? 0.333;
		const signal = (bal - base) * 100;

		if (signal > 15 && hard != null && hard > 0.5) {
			return { text: 'STRONG SIGNAL — space is navigable', color: '#00ff88' };
		}
		if (signal > 10) {
			return { text: 'WEAK SIGNAL — space has structure but router is biased', color: '#ffaa00' };
		}
		return { text: 'NO SIGNAL — space is not navigable', color: '#ff0040' };
	}

	const modelColors: Record<string, string> = {
		falcon: '#00d4ff',
		qwen: '#00ff88',
		smollm: '#ffaa00',
	};
</script>

<div class="bg-[#1a1a2e] rounded-lg border border-gray-800 p-4">
	<div class="mb-4">
		<div class="text-xs text-gray-500 mb-1">Corrected Router Diagnostics</div>
		<div class="text-sm text-gray-400">Class-balanced metrics, hard-set test, semantic coherence</div>
	</div>

	{#if error}
		<div class="text-[#ff0040] text-sm py-4">{error}</div>
	{:else if !data}
		<div class="text-gray-600 text-sm py-4">Loading...</div>
	{:else}
		<!-- Verdict -->
		{@const verdict = getVerdict()}
		<div class="rounded-lg border p-3 mb-4 text-center"
			style="border-color: {verdict.color}40; background: {verdict.color}10;">
			<div class="text-sm font-bold" style="color: {verdict.color}">{verdict.text}</div>
			<div class="text-xs text-gray-500 mt-1">
				Balanced: {(data.balanced.accuracy * 100).toFixed(1)}% vs {(data.balanced.random_baseline * 100).toFixed(0)}% random
				{#if data.hard_set.accuracy != null}
					• Hard set: {(data.hard_set.accuracy * 100).toFixed(1)}% vs 50% random
				{/if}
			</div>
		</div>

		<!-- Class Distribution Warning -->
		<div class="rounded-lg border border-[#ffaa0040] bg-[#ffaa0010] p-3 mb-4">
			<div class="text-xs text-[#ffaa00] mb-1">⚠️ Class Imbalance Detected</div>
			<div class="text-xs text-gray-400">
				Oracle labels: Falcon {data.class_distribution.train.falcon} ({(data.class_distribution.train.falcon / data.total_prompts * 100).toFixed(0)}%) •
				Qwen {data.class_distribution.train.qwen} ({(data.class_distribution.train.qwen / data.total_prompts * 100).toFixed(0)}%) •
				SmolLM {data.class_distribution.train.smollm} ({(data.class_distribution.train.smollm / data.total_prompts * 100).toFixed(0)}%)
			</div>
			<div class="text-xs text-gray-500 mt-1">
				Most-frequent baseline ({data.imbalanced.most_frequent_class}): {(data.imbalanced.most_frequent_baseline * 100).toFixed(1)}%
			</div>
		</div>

		<!-- Imbalanced vs Balanced -->
		<div class="grid grid-cols-2 gap-3 mb-4">
			<!-- Imbalanced -->
			<button class="text-left p-3 rounded border transition-all"
				class:border-[#ff004040]={expandedSection === 'imbalanced'}
				class:bg-[#ff004010]={expandedSection === 'imbalanced'}
				class:border-gray-800={expandedSection !== 'imbalanced'}
				onclick={() => toggleSection('imbalanced')}>
				<div class="text-xs text-gray-500 mb-1">Imbalanced Training</div>
				<div class="text-xl font-bold font-mono"
					class:text-[#ff0040]={data.imbalanced.improvement_over_baseline <= 0.05}
					class:text-[#ffaa00]={data.imbalanced.improvement_over_baseline > 0.05 && data.imbalanced.improvement_over_baseline <= 0.1}
					class:text-[#00ff88]={data.imbalanced.improvement_over_baseline > 0.1}>
					{(data.imbalanced.accuracy * 100).toFixed(1)}%
				</div>
				<div class="text-xs text-gray-500">
					+{(data.imbalanced.improvement_over_baseline * 100).toFixed(1)}% over baseline
				</div>
			</button>

			<!-- Balanced -->
			<button class="text-left p-3 rounded border transition-all"
				class:border-[#00ff8840]={expandedSection === 'balanced'}
				class:bg-[#00ff8810]={expandedSection === 'balanced'}
				class:border-gray-800={expandedSection !== 'balanced'}
				onclick={() => toggleSection('balanced')}>
				<div class="text-xs text-gray-500 mb-1">Balanced Training ({data.balanced.samples_per_class}/class)</div>
				<div class="text-xl font-bold font-mono"
					class:text-[#ff0040]={data.balanced.accuracy <= data.balanced.random_baseline + 0.05}
					class:text-[#ffaa00]={data.balanced.accuracy > data.balanced.random_baseline + 0.05 && data.balanced.accuracy <= data.balanced.random_baseline + 0.15}
					class:text-[#00ff88]={data.balanced.accuracy > data.balanced.random_baseline + 0.15}>
					{(data.balanced.accuracy * 100).toFixed(1)}%
				</div>
				<div class="text-xs text-gray-500">
					vs {(data.balanced.random_baseline * 100).toFixed(0)}% random
				</div>
			</button>
		</div>

		<!-- Per-class table (shows for expanded section) -->
		{#if expandedSection === 'imbalanced' || expandedSection === 'balanced'}
			{@const section = expandedSection === 'imbalanced' ? data.imbalanced : data.balanced}
			<div class="mb-4">
				<div class="text-xs text-gray-500 mb-2">
					{expandedSection === 'imbalanced' ? 'Imbalanced' : 'Balanced'} Per-Class Metrics
				</div>
				<div class="overflow-x-auto">
					<table class="w-full text-xs">
						<thead>
							<tr class="text-gray-500 border-b border-gray-800">
								<th class="text-left py-1.5 pr-3">Model</th>
								<th class="text-right py-1.5 px-3">Precision</th>
								<th class="text-right py-1.5 px-3">Recall</th>
								<th class="text-right py-1.5 px-3">F1</th>
								<th class="text-right py-1.5 pl-3">N</th>
							</tr>
						</thead>
						<tbody>
							{#each Object.entries(section.per_class) as [mid, metrics]}
								{@const m = metrics as { precision: number; recall: number; f1: number; support: number }}
								<tr class="border-b border-gray-800/50">
									<td class="py-1.5 pr-3 font-medium" style="color: {modelColors[mid] || '#fff'}">{mid}</td>
									<td class="text-right py-1.5 px-3 font-mono">{(m.precision * 100).toFixed(1)}%</td>
									<td class="text-right py-1.5 px-3 font-mono">{(m.recall * 100).toFixed(1)}%</td>
									<td class="text-right py-1.5 px-3 font-mono">{(m.f1 * 100).toFixed(1)}%</td>
									<td class="text-right py-1.5 pl-3 text-gray-500">{m.support}</td>
								</tr>
							{/each}
						</tbody>
					</table>
				</div>
			</div>
		{/if}

		<!-- Hard Set Test -->
		{#if data.hard_set.accuracy != null}
			<button class="w-full text-left p-3 rounded border mb-4 transition-all"
				class:border-[#00d4ff40]={expandedSection === 'hard'}
				class:bg-[#00d4ff10]={expandedSection === 'hard'}
				class:border-gray-800={expandedSection !== 'hard'}
				onclick={() => toggleSection('hard')}>
				<div class="flex items-center justify-between">
					<div>
						<div class="text-xs text-gray-500 mb-1">Hard Set (Falcon NOT the best)</div>
						<div class="text-sm text-gray-400">
							{data.hard_set.total_prompts} prompts ({(data.hard_set.fraction_of_val * 100).toFixed(0)}% of val set)
						</div>
					</div>
					<div class="text-right">
						<div class="text-xl font-bold font-mono"
							class:text-[#ff0040]={data.hard_set.accuracy < 0.5}
							class:text-[#00ff88]={data.hard_set.accuracy >= 0.5}>
							{(data.hard_set.accuracy * 100).toFixed(1)}%
						</div>
						<div class="text-xs text-gray-500">vs 50% random</div>
					</div>
				</div>
				{#if expandedSection === 'hard'}
					<div class="mt-3 pt-3 border-t border-gray-800">
						<div class="text-xs text-gray-500 mb-2">Hard Set Distribution</div>
						<div class="flex gap-4 text-xs">
							{#each Object.entries(data.hard_set.distribution) as [mid, count]}
								<span style="color: {modelColors[mid] || '#fff'}">
									{mid}: {count}
								</span>
							{/each}
						</div>
					</div>
				{/if}
			</button>
		{/if}

		<!-- Semantic Coherence -->
		<button class="w-full text-left p-3 rounded border mb-4 transition-all"
			class:border-[#ffaa0040]={expandedSection === 'coherence'}
			class:bg-[#ffaa0010]={expandedSection === 'coherence'}
			class:border-gray-800={expandedSection !== 'coherence'}
			onclick={() => toggleSection('coherence')}>
			<div class="flex items-center justify-between">
				<div>
					<div class="text-xs text-gray-500 mb-1">Semantic Coherence</div>
					<div class="text-sm text-gray-400">
						Router chose non-Falcon: {data.semantic_coherence.non_falcon_predictions}/{data.n_val} ({(data.semantic_coherence.fraction * 100).toFixed(0)}%)
					</div>
				</div>
			</div>
			{#if expandedSection === 'coherence'}
				<div class="mt-3 pt-3 border-t border-gray-800 space-y-3">
					{#each [1, 2] as cls}
						{@const mid = cls === 1 ? 'qwen' : 'smollm'}
						{@const samples = data.semantic_coherence.samples[mid] || []}
						{#if samples.length > 0}
							<div>
								<div class="text-xs font-medium mb-1" style="color: {modelColors[mid]}">
									{mid} ({samples.length} prompts shown)
								</div>
								<div class="space-y-1">
									{#each samples.slice(0, 4) as s}
										<div class="flex items-start gap-2 text-xs">
											<span class={s.correct ? 'text-[#00ff88]' : 'text-[#ff0040]'}>
												{s.correct ? '✓' : '✗'}
											</span>
											<span class="text-gray-600">[{s.actual}]</span>
											<span class="text-gray-400 truncate">{s.prompt}</span>
										</div>
									{/each}
								</div>
							</div>
						{/if}
					{/each}
				</div>
			{/if}
		</button>

		<!-- Bottom line -->
		<div class="text-xs text-gray-500 text-center">
			{data.total_prompts} prompts • checkpoint step {data.checkpoint_step}
		</div>
	{/if}
</div>
