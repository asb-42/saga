<script lang="ts">
	import { addToast, removeToast, subscribeToToasts } from '$lib/toast';
	import { onMount, onDestroy } from 'svelte';
	import { apiSSE } from '$lib/api';
	import type { Toast } from '$lib/toast';

	let eventSource: EventSource | null = null;
	let toasts: Toast[] = [];
	let unsubscribe: (() => void) | null = null;

	onMount(() => {
		// Subscribe to toast changes
		unsubscribe = subscribeToToasts((newToasts) => {
			toasts = newToasts;
		});

		// Subscribe to anomaly events
		eventSource = apiSSE('/api/anomaly/stream');
		if (!eventSource) return;

		eventSource.onmessage = (event) => {
			try {
				const data = JSON.parse(event.data);
				if (data.type === 'anomaly') {
					addToast(
						`Anomaly detected: ${data.alert_type}`,
						'error',
						10000
					);
				}
			} catch (e) {
				console.error('Failed to parse anomaly event');
			}
		};
	});

	onDestroy(() => {
		eventSource?.close();
		unsubscribe?.();
	});

	const typeColors: Record<string, string> = {
		info: 'bg-[#00d4ff]/90 border-[#00d4ff]',
		success: 'bg-[#00ff88]/90 border-[#00ff88]',
		warning: 'bg-[#ffaa00]/90 border-[#ffaa00]',
		error: 'bg-[#ff0040]/90 border-[#ff0040]',
	};

	const typeIcons: Record<string, string> = {
		info: 'ℹ️',
		success: '✅',
		warning: '⚠️',
		error: '🚨',
	};
</script>

<!-- Toast container -->
<div class="fixed top-4 right-4 z-50 flex flex-col gap-2 max-w-sm">
	{#each toasts as toast (toast.id)}
		<div
			class="flex items-center gap-3 px-4 py-3 rounded-lg border shadow-lg backdrop-blur-sm
				{typeColors[toast.type]} animate-slide-in"
			role="alert"
		>
			<span class="text-lg">{typeIcons[toast.type]}</span>
			<span class="text-white flex-1">{toast.message}</span>
			<button
				onclick={() => removeToast(toast.id)}
				class="text-white/70 hover:text-white"
				aria-label="Dismiss"
			>
				×
			</button>
		</div>
	{/each}
</div>

<style>
	@keyframes slide-in {
		from {
			transform: translateX(100%);
			opacity: 0;
		}
		to {
			transform: translateX(0);
			opacity: 1;
		}
	}
	.animate-slide-in {
		animation: slide-in 0.3s ease-out;
	}
</style>
