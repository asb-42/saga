<script lang="ts">
	import '../app.css';
	import Toast from '$lib/components/Toast.svelte';

	let { children } = $props();

	let currentPath = $derived('/');

	const navItems = [
		{ path: '/', label: 'Dashboard', icon: '📊' },
		{ path: '/pipeline', label: 'Pipeline', icon: '📡' },
		{ path: '/metrics', label: 'Metrics', icon: '📈' },
		{ path: '/live', label: 'Live Feed', icon: '🔬' },
		{ path: '/anomaly', label: 'Anomaly', icon: '🔴' },
		{ path: '/logs', label: 'Logs', icon: '📋' },
		{ path: '/models', label: 'Models', icon: '🧠' },
		{ path: '/training', label: 'Training', icon: '🎯' },
		{ path: '/benchmarks', label: 'Benchmarks', icon: '🏆' },
	];
</script>

<svelte:head>
	<meta name="description" content="SAGA Research Lab - Interactive dashboard for AI ensemble research" />
</svelte:head>

<!-- Skip link for keyboard users -->
<a
	href="#main-content"
	class="sr-only focus:not-sr-only focus:absolute focus:top-2 focus:left-2 focus:z-50 focus:bg-[#00d4ff] focus:text-black focus:px-4 focus:py-2 focus:rounded-lg"
>
	Skip to main content
</a>

<div class="min-h-screen flex flex-col">
	<!-- Toast notifications -->
	<Toast />

	<!-- Header -->
	<header class="border-b border-gray-800 bg-[#1a1a2e] px-6 py-4" role="banner">
		<div class="flex items-center justify-between">
			<div class="flex items-center gap-3">
				<span class="text-2xl" aria-hidden="true">🧬</span>
				<div>
					<h1 class="text-xl font-bold text-[#00d4ff]">SAGA Research Lab</h1>
					<p class="text-xs text-gray-500">Selective AI Generation Architecture</p>
				</div>
			</div>
			<div class="flex items-center gap-4">
				<span class="text-sm text-gray-400">v0.1.0</span>
				<div
					class="w-2 h-2 rounded-full bg-[#00ff88] animate-pulse"
					role="status"
					aria-label="System online"
				></div>
			</div>
		</div>
	</header>

	<div class="flex flex-1">
		<!-- Sidebar -->
		<nav class="w-48 border-r border-gray-800 bg-[#1a1a2e]/50 p-4" aria-label="Main navigation">
			<ul class="space-y-2" role="list">
				{#each navItems as item}
					<li>
						<a
							href={item.path}
							aria-current={currentPath === item.path ? 'page' : undefined}
							class="flex items-center gap-3 rounded-lg px-3 py-2 text-sm transition-colors
								{currentPath === item.path
									? 'bg-[#00d4ff]/10 text-[#00d4ff]'
									: 'text-gray-400 hover:bg-gray-800 hover:text-white'}"
						>
							<span aria-hidden="true">{item.icon}</span>
							<span>{item.label}</span>
						</a>
					</li>
				{/each}
			</ul>
		</nav>

		<!-- Main content -->
		<main
			id="main-content"
			class="flex-1 p-6 overflow-auto"
			role="main"
			aria-label="Main content"
			tabindex="-1"
		>
			{@render children()}
		</main>
	</div>

	<!-- Status Bar -->
	<footer class="border-t border-gray-800 bg-[#1a1a2e] px-6 py-2" role="contentinfo">
		<div class="flex items-center justify-between text-xs text-gray-500">
			<div class="flex items-center gap-4">
				<span aria-label="System status: online">🟢 System Online</span>
				<span aria-label="CPU usage">CPU: --</span>
				<span aria-label="Memory usage">RAM: --</span>
			</div>
			<div>
				SAGA Research Lab Dashboard
			</div>
		</div>
	</footer>
</div>

<style>
	/* Screen reader only - hidden visually but accessible */
	.sr-only {
		position: absolute;
		width: 1px;
		height: 1px;
		padding: 0;
		margin: -1px;
		overflow: hidden;
		clip: rect(0, 0, 0, 0);
		white-space: nowrap;
		border-width: 0;
	}

	/* Focus visible for keyboard navigation */
	:global(a:focus-visible),
	:global(button:focus-visible) {
		outline: 2px solid #00d4ff;
		outline-offset: 2px;
	}
</style>
