<script lang="ts">
	import { onMount, onDestroy } from 'svelte';
	import Chart from 'chart.js/auto';
	import { apiSSE } from '$lib/api';

	let canvas: HTMLCanvasElement;
	let chart: Chart | null = null;
	let eventSource: EventSource | null = null;
	let dataPoints = $state<{ x: number; y: number }[]>([]);

	onMount(() => {
		if (canvas) {
			chart = new Chart(canvas, {
				type: 'line',
				data: {
					datasets: [
						{
							label: 'Training Loss',
							data: dataPoints,
							borderColor: '#00d4ff',
							backgroundColor: 'rgba(0, 212, 255, 0.1)',
							fill: true,
							tension: 0.4,
						},
					],
				},
				options: {
					responsive: true,
					maintainAspectRatio: false,
					animation: { duration: 300 },
					scales: {
						x: {
							type: 'linear',
							title: { display: true, text: 'Step', color: '#808080' },
							grid: { color: 'rgba(255,255,255,0.05)' },
							ticks: { color: '#808080' },
						},
						y: {
							title: { display: true, text: 'Loss', color: '#808080' },
							grid: { color: 'rgba(255,255,255,0.05)' },
							ticks: { color: '#808080' },
						},
					},
					plugins: {
						legend: { labels: { color: '#e0e0e0' } },
					},
				},
			});
		}

		connectSSE();
	});

	onDestroy(() => {
		eventSource?.close();
		chart?.destroy();
	});

	function connectSSE() {
		eventSource = apiSSE('/api/metrics/stream');
		if (!eventSource) return;

		eventSource.onmessage = (event) => {
			try {
				const data = JSON.parse(event.data);
				if (data.type === 'metric' && data.name === 'train/loss') {
					dataPoints = [...dataPoints, { x: data.step, y: data.value }].slice(-200);
					if (chart) {
						chart.data.datasets[0].data = dataPoints;
						chart.update('none');
					}
				}
			} catch (e) {
				console.error('Failed to parse metric');
			}
		};
	}
</script>

<div class="bg-[#1a1a2e] rounded-lg p-4 border border-gray-800">
	<h3 class="text-lg font-semibold text-white mb-4">Training Loss</h3>
	<div class="h-64">
		<canvas bind:this={canvas}></canvas>
	</div>
</div>
