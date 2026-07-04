<script lang="ts">
	interface Props {
		value: number;
		label: string;
		min?: number;
		max?: number;
		color?: string;
		unit?: string;
	}

	let {
		value,
		label,
		min = 0,
		max = 100,
		color = '#00d4ff',
		unit = '',
	}: Props = $props();

	let percentage = $derived(Math.min(100, Math.max(0, ((value - min) / (max - min)) * 100)));
	let displayValue = $derived(typeof value === 'number' ? value.toFixed(4) : value);
</script>

<div class="bg-[#1a1a2e] rounded-lg p-4 border border-gray-800">
	<div class="flex items-center justify-between mb-2">
		<span class="text-sm text-gray-400">{label}</span>
		<span class="text-lg font-mono font-bold" style="color: {color}">
			{displayValue}{unit}
		</span>
	</div>
	<div class="h-3 bg-gray-800 rounded-full overflow-hidden">
		<div
			class="h-full rounded-full transition-all duration-500 ease-out"
			style="width: {percentage}%; background-color: {color}"
		></div>
	</div>
	<div class="flex justify-between mt-1 text-xs text-gray-600">
		<span>{min}</span>
		<span>{max}</span>
	</div>
</div>
