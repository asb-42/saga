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
	];
</script>

<div class="min-h-screen flex flex-col">
	<!-- Toast notifications -->
	<Toast />

	<!-- Header -->
	<header class="border-b border-gray-800 bg-[#1a1a2e] px-6 py-4">
		<div class="flex items-center justify-between">
			<div class="flex items-center gap-3">
				<span class="text-2xl">🧬</span>
				<div>
					<h1 class="text-xl font-bold text-[#00d4ff]">SAGA Research Lab</h1>
					<p class="text-xs text-gray-500">Selective AI Generation Architecture</p>
				</div>
			</div>
			<div class="flex items-center gap-4">
				<span class="text-sm text-gray-400">v0.1.0</span>
				<div class="w-2 h-2 rounded-full bg-[#00ff88] animate-pulse"></div>
			</div>
		</div>
	</header>

	<div class="flex flex-1">
		<!-- Sidebar -->
		<nav class="w-48 border-r border-gray-800 bg-[#1a1a2e]/50 p-4">
			<ul class="space-y-2">
				{#each navItems as item}
					<li>
						<a
							href={item.path}
							class="flex items-center gap-3 rounded-lg px-3 py-2 text-sm transition-colors
								{currentPath === item.path
									? 'bg-[#00d4ff]/10 text-[#00d4ff]'
									: 'text-gray-400 hover:bg-gray-800 hover:text-white'}"
						>
							<span>{item.icon}</span>
							<span>{item.label}</span>
						</a>
					</li>
				{/each}
			</ul>
		</nav>

		<!-- Main content -->
		<main class="flex-1 p-6 overflow-auto">
			{@render children()}
		</main>
	</div>

	<!-- Status Bar -->
	<footer class="border-t border-gray-800 bg-[#1a1a2e] px-6 py-2">
		<div class="flex items-center justify-between text-xs text-gray-500">
			<div class="flex items-center gap-4">
				<span>🟢 System Online</span>
				<span>CPU: --</span>
				<span>RAM: --</span>
			</div>
			<div>
				SAGA Research Lab Dashboard
			</div>
		</div>
	</footer>
</div>
