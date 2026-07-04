<script lang="ts">
	interface Props {
		status: string;
	}

	let { status }: Props = $props();

	const statusColors: Record<string, string> = {
		pending: 'bg-gray-600',
		running: 'bg-[#00d4ff]',
		completed: 'bg-[#00ff88]',
		failed: 'bg-[#ff0040]',
		paused: 'bg-[#ffaa00]',
	};

	const statusLabels: Record<string, string> = {
		pending: 'Pending',
		running: 'Running',
		completed: 'Completed',
		failed: 'Failed',
		paused: 'Paused',
	};
</script>

<div class="flex items-center gap-2" role="status" aria-label={`Status: ${statusLabels[status] || status}`}>
	<div
		class="w-2 h-2 rounded-full {statusColors[status] || 'bg-gray-600'}"
		class:animate-pulse={status === 'running'}
		aria-hidden="true"
	></div>
	<span class="text-sm text-gray-400 capitalize">{status}</span>
</div>
