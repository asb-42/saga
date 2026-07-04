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
		<span class="text-sm text-gray-400" id={`gauge-label-${label}`}>{label}</span>
		<span class="text-lg font-mono font-bold" style="color: {color}" aria-hidden="true">
			{displayValue}{unit}
		</span>
	</div>
	<div
		class="h-3 bg-gray-800 rounded-full overflow-hidden"
		role="meter"
		aria-valuenow={value}
		aria-valuemin={min}
		aria-valuemax={max}
		aria-labelledby={`gauge-label-${label}`}
		aria-valuetext={`${displayValue}${unit}`}
	>
		<div
			class="h-full rounded-full transition-all duration-500 ease-out"
			style="width: {percentage}%; background-color: {color}"
		></div>
	</div>
	<div class="flex justify-between mt-1 text-xs text-gray-600" aria-hidden="true">
		<span>{min}</span>
		<span>{max}</span>
	</div>
</div>
