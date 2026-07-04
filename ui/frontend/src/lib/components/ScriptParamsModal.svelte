<script lang="ts">
	import { t } from '$lib/i18n';

	interface ParamDef {
		type: 'int' | 'float' | 'select' | 'multi' | 'flag';
		default: any;
		label: string;
		min?: number;
		max?: number;
		choices?: string[];
	}

	interface Props {
		scriptId: string;
		scriptName: string;
		params: Record<string, ParamDef>;
		onSubmit: (params: Record<string, any>) => void;
		onCancel: () => void;
	}

	let { scriptId, scriptName, params, onSubmit, onCancel }: Props = $props();

	let formValues = $state<Record<string, any>>({});
	let selectedBenchmarks = $state<Set<string>>(new Set());

	// Initialize form with defaults
	$effect(() => {
		const init: Record<string, any> = {};
		const benchInit = new Set<string>();
		for (const [key, def] of Object.entries(params)) {
			if (def.type === 'multi' && Array.isArray(def.default)) {
				def.default.forEach((b: string) => benchInit.add(b));
			} else {
				init[key] = def.default ?? '';
			}
		}
		formValues = init;
		selectedBenchmarks = benchInit;
	});

	function handleSubmit() {
		const out: Record<string, any> = {};
		for (const [key, def] of Object.entries(params)) {
			if (def.type === 'flag') {
				if (formValues[key]) out[key] = '';
			} else if (def.type === 'multi') {
				if (selectedBenchmarks.size > 0) {
					out[key] = Array.from(selectedBenchmarks);
				}
			} else if (formValues[key] !== '' && formValues[key] !== null && formValues[key] !== undefined) {
				out[key] = String(formValues[key]);
			}
		}
		onSubmit(out);
	}

	function toggleBenchmark(b: string) {
		if (selectedBenchmarks.has(b)) {
			selectedBenchmarks.delete(b);
			selectedBenchmarks = new Set(selectedBenchmarks);
		} else {
			selectedBenchmarks.add(b);
			selectedBenchmarks = new Set(selectedBenchmarks);
		}
	}

	function handleBackdropClick(e: MouseEvent) {
		if (e.target === e.currentTarget) onCancel();
	}

	function handleKeydown(e: KeyboardEvent) {
		if (e.key === 'Escape') onCancel();
	}
</script>

<svelte:window onkeydown={handleKeydown} />

<!-- svelte-ignore a11y_click_events_have_key_events -->
<!-- svelte-ignore a11y_no_static_element_interactions -->
<div
	class="fixed inset-0 bg-black/60 backdrop-blur-sm z-50 flex items-center justify-center p-4"
	onclick={handleBackdropClick}
	role="dialog"
	aria-modal="true"
	aria-label="Configure {scriptName}"
>
	<div class="bg-[#1a1a2e] border border-gray-700 rounded-xl w-full max-w-lg max-h-[80vh] overflow-hidden shadow-2xl">
		<!-- Header -->
		<div class="px-6 py-4 border-b border-gray-800 flex items-center justify-between">
			<div>
				<h2 class="text-lg font-semibold text-white">{scriptName}</h2>
				<p class="text-xs text-gray-500">Configure parameters before running</p>
			</div>
			<button
				onclick={onCancel}
				class="text-gray-500 hover:text-white text-xl leading-none focus:outline-none focus:ring-2 focus:ring-[#00d4ff]/50 rounded px-1"
				aria-label="Close"
			>
				x
			</button>
		</div>

		<!-- Form body -->
		<div class="px-6 py-4 overflow-y-auto max-h-[60vh] space-y-4">
			{#if Object.keys(params).length === 0}
				<p class="text-gray-500 text-center py-4">No configurable parameters for this script.</p>
			{:else}
				{#each Object.entries(params) as [key, def]}
					<div class="space-y-1">
						<label for="param-{key}" class="block text-sm text-gray-300">{def.label}</label>

						{#if def.type === 'int'}
							<input
								id="param-{key}"
								type="number"
								bind:value={formValues[key]}
								min={def.min}
								max={def.max}
								step={1}
								class="w-full bg-[#0a0a0f] border border-gray-700 rounded px-3 py-2 text-white text-sm focus:border-[#00d4ff] focus:outline-none focus:ring-1 focus:ring-[#00d4ff]/50"
							/>
							<p class="text-xs text-gray-600">
								Default: {def.default ?? 'none'}{def.min != null ? ` | Min: ${def.min}` : ''}{def.max != null ? ` | Max: ${def.max}` : ''}
							</p>

						{:else if def.type === 'float'}
							<input
								id="param-{key}"
								type="number"
								bind:value={formValues[key]}
								min={def.min}
								max={def.max}
								step={0.00001}
								class="w-full bg-[#0a0a0f] border border-gray-700 rounded px-3 py-2 text-white text-sm focus:border-[#00d4ff] focus:outline-none focus:ring-1 focus:ring-[#00d4ff]/50"
							/>
							<p class="text-xs text-gray-600">
								Default: {def.default ?? 'none'}
							</p>

						{:else if def.type === 'select'}
							<select
								id="param-{key}"
								bind:value={formValues[key]}
								class="w-full bg-[#0a0a0f] border border-gray-700 rounded px-3 py-2 text-white text-sm focus:border-[#00d4ff] focus:outline-none focus:ring-1 focus:ring-[#00d4ff]/50"
							>
								{#each def.choices || [] as choice}
									<option value={choice}>{choice}</option>
								{/each}
							</select>

						{:else if def.type === 'multi'}
							<div class="flex flex-wrap gap-2">
								{#each def.choices || [] as choice}
									<button
										type="button"
										onclick={() => toggleBenchmark(choice)}
										class="px-3 py-1 rounded text-sm border transition-all {selectedBenchmarks.has(choice)
											? 'bg-[#00d4ff]/20 border-[#00d4ff]/50 text-[#00d4ff]'
											: 'bg-[#0a0a0f] border-gray-700 text-gray-500 hover:border-gray-500'}"
										aria-pressed={selectedBenchmarks.has(choice)}
									>
										{choice.toUpperCase()}
									</button>
								{/each}
							</div>
							<p class="text-xs text-gray-600">Click to toggle benchmarks</p>

						{:else if def.type === 'flag'}
							<label class="flex items-center gap-3 cursor-pointer">
								<input
									type="checkbox"
									bind:checked={formValues[key]}
									class="w-4 h-4 rounded border-gray-700 bg-[#0a0a0f] text-[#00d4ff] focus:ring-[#00d4ff]/50"
								/>
								<span class="text-sm text-gray-400">Enable</span>
							</label>
						{/if}
					</div>
				{/each}
			{/if}
		</div>

		<!-- Footer -->
		<div class="px-6 py-4 border-t border-gray-800 flex gap-3 justify-end">
			<button
				onclick={onCancel}
				class="px-4 py-2 text-sm text-gray-400 hover:text-white border border-gray-700 rounded hover:border-gray-500 transition-colors focus:outline-none focus:ring-2 focus:ring-gray-600"
			>
				Cancel
			</button>
			<button
				onclick={handleSubmit}
				class="px-4 py-2 text-sm bg-[#00d4ff]/20 text-[#00d4ff] border border-[#00d4ff]/50 rounded hover:bg-[#00d4ff]/30 transition-colors focus:outline-none focus:ring-2 focus:ring-[#00d4ff]/50"
			>
				Start
			</button>
		</div>
	</div>
</div>
