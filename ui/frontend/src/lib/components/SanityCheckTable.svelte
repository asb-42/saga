<script lang="ts">
	import { onMount } from 'svelte';
	import { apiFetch } from '$lib/api';

	let data = $state<any>(null);
	let error = $state('');

	onMount(async () => {
		try {
			const resp = await apiFetch('/api/alignment/sanity-checks');
			if (resp.ok) {
				data = await resp.json();
			} else {
				error = `HTTP ${resp.status}`;
			}
		} catch (e) {
			error = String(e);
		}
	});

	const expectedColors: Record<string, string> = {
		very_close: '#00ff88',
		close: '#00d4ff',
		moderate: '#ffaa00',
		far: '#ff0040',
	};

	const expectedLabels: Record<string, string> = {
		very_close: 'Very Close',
		close: 'Close',
		moderate: 'Moderate',
		far: 'Far',
	};

	function cosColor(cos: number): string {
		if (cos >= 0.9) return '#00ff88';
		if (cos >= 0.7) return '#00d4ff';
		if (cos >= 0.5) return '#ffaa00';
		return '#ff0040';
	}

	function matches(cos: number, expected: string): boolean {
		if (expected === 'very_close' || expected === 'close') return cos >= 0.7;
		if (expected === 'moderate') return cos >= 0.4 && cos < 0.8;
		return cos < 0.5;
	}
</script>

<div class="bg-[#1a1a2e] rounded-lg border border-gray-800 p-4">
	<div class="mb-4">
		<div class="text-xs text-gray-500 mb-1">Sanity Check: Prompt Pair Distances</div>
		<div class="text-sm text-gray-400">Manual validation of shared space semantic distances</div>
	</div>

	{#if error}
		<div class="text-[#ff0040] text-sm py-4">{error}</div>
	{:else if !data}
		<div class="text-gray-600 text-sm py-4">Loading...</div>
	{:else}
		<div class="space-y-3">
			{#each data.pairs as pair}
				{@const ok = matches(pair.cosine_similarity, pair.expected)}
				<div class="p-3 bg-black/20 rounded border border-gray-800">
					<div class="flex items-start gap-3">
						<!-- Status indicator -->
						<div class="mt-1 w-3 h-3 rounded-full flex-shrink-0"
							class:bg-[#00ff88]={ok}
							class:bg-[#ff0040]={!ok}></div>

						<div class="flex-1 min-w-0">
							<!-- Prompts -->
							<div class="grid grid-cols-2 gap-2 text-xs mb-2">
								<div class="text-gray-300 truncate" title={pair.prompt_a}>{pair.prompt_a}</div>
								<div class="text-gray-300 truncate" title={pair.prompt_b}>{pair.prompt_b}</div>
							</div>

							<!-- Metrics -->
							<div class="flex items-center gap-4 text-xs">
								<span class="text-gray-500">Expected:</span>
								<span class="font-mono" style="color: {expectedColors[pair.expected]}">
									{expectedLabels[pair.expected]}
								</span>
								<span class="text-gray-500">Actual:</span>
								<span class="font-mono" style="color: {cosColor(pair.cosine_similarity)}">
									cos={pair.cosine_similarity.toFixed(4)}
								</span>
								<span class="text-gray-500">l2={pair.l2_distance.toFixed(1)}</span>
							</div>
						</div>
					</div>
				</div>
			{/each}
		</div>

		<!-- Summary -->
		{@const matched = data.pairs.filter((p: any) => matches(p.cosine_similarity, p.expected)).length}
		<div class="mt-3 text-xs text-gray-500">
			Matched expectations: {matched}/{data.pairs.length}
			({(matched / data.pairs.length * 100).toFixed(0)}%)
		</div>
	{/if}
</div>
