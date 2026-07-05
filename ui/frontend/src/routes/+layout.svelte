<script lang="ts">
	import '../app.css';
	import Toast from '$lib/components/Toast.svelte';
	import ThemeToggle from '$lib/components/ThemeToggle.svelte';
	import { t } from '$lib/i18n';
	import { onMount } from 'svelte';
	import { initTheme } from '$lib/theme';

	let { children } = $props();

	let currentPath = $derived('/');
	let sysInfo = $state<any>(null);

	async function fetchSystemInfo() {
		try {
			const resp = await fetch('/api/system');
			if (resp.ok) sysInfo = await resp.json();
		} catch { /* server not ready yet */ }
	}

	onMount(() => {
		initTheme();
		fetchSystemInfo();
		const interval = setInterval(fetchSystemInfo, 5000);
		return () => clearInterval(interval);
	});

	const navItems = [
		{ path: '/', label: t('nav.dashboard'), icon: '📊' },
		{ path: '/pipeline', label: t('nav.pipeline'), icon: '📡' },
		{ path: '/eval', label: t('nav.eval'), icon: '🔬' },
		{ path: '/metrics', label: t('nav.metrics'), icon: '📈' },
		{ path: '/live', label: t('nav.live'), icon: '⚡' },
		{ path: '/anomaly', label: t('nav.anomaly'), icon: '🔴' },
		{ path: '/logs', label: t('nav.logs'), icon: '📋' },
		{ path: '/models', label: t('nav.models'), icon: '🧠' },
		{ path: '/training', label: t('nav.training'), icon: '🎯' },
		{ path: '/benchmarks', label: t('nav.benchmarks'), icon: '🏆' },
	];
</script>

<svelte:head>
	<meta name="description" content="Saga Research Lab - Interactive dashboard for AI ensemble research" />
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
	<header class="header-bg px-6 py-4" role="banner">
		<div class="flex items-center justify-between">
			<div class="flex items-center gap-3">
				<span class="text-2xl" aria-hidden="true">🧬</span>
				<div>
					<h1 class="text-xl font-bold text-[#00d4ff]">{t('app.name')}</h1>
					<p class="text-xs text-secondary">{t('app.description')}</p>
				</div>
			</div>
			<div class="flex items-center gap-4">
				<span class="text-sm text-secondary">{t('app.version')}</span>
				<ThemeToggle />
				<div
					class="w-2 h-2 rounded-full bg-[#00ff88] animate-pulse"
					role="status"
					aria-label={t('status.online')}
				></div>
			</div>
		</div>
	</header>

	<div class="flex flex-1">
		<!-- Sidebar -->
		<nav class="sidebar-bg w-48 border-r p-4" aria-label="Main navigation">
			<ul class="space-y-2" role="list">
				{#each navItems as item}
					<li>
						<a
							href={item.path}
							aria-current={currentPath === item.path ? 'page' : undefined}
							class="nav-link"
							class:nav-link-active={currentPath === item.path}
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
	<footer class="header-bg border-t px-6 py-2" role="contentinfo">
		<div class="flex items-center justify-between text-xs text-secondary">
			<div class="flex items-center gap-4">
				<span aria-label={t('status.online')}>🟢 {t('status.online')}</span>
				{#if sysInfo}
					<span aria-label="CPU usage">CPU: {sysInfo.cpu?.percent ?? '--'}%</span>
					<span aria-label="Memory usage">RAM: {sysInfo.memory?.percent ?? '--'}%</span>
					{#if sysInfo.gpu?.devices}
						{#each sysInfo.gpu.devices as gpu}
							<span aria-label="GPU memory">GPU{gpu.index}: {gpu.allocated_gb}/{gpu.total_mem_gb}GB</span>
						{/each}
					{/if}
				{/if}
			</div>
			<div>
				{t('app.name')} Dashboard
			</div>
		</div>
	</footer>
</div>

<style>
	/* Screen reader only - hidden visually but accessible */
	:global(.sr-only) {
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
		outline: 2px solid var(--color-accent);
		outline-offset: 2px;
	}

	/* Theme-aware backgrounds */
	:global(.header-bg) {
		background-color: var(--color-bg-secondary);
		border-color: var(--color-border);
	}

	:global(.sidebar-bg) {
		background-color: color-mix(in srgb, var(--color-bg-secondary) 50%, transparent);
		border-color: var(--color-border);
	}

	:global(.text-secondary) {
		color: var(--color-text-secondary);
	}

	:global(.nav-link) {
		display: flex;
		align-items: center;
		gap: 0.75rem;
		padding: 0.5rem 0.75rem;
		border-radius: 0.5rem;
		font-size: 0.875rem;
		color: var(--color-text-secondary);
		transition: all 0.15s ease;
	}

	:global(.nav-link:hover) {
		background-color: var(--color-bg-tertiary);
		color: var(--color-text-primary);
	}

	:global(.nav-link-active) {
		background-color: color-mix(in srgb, var(--color-accent) 10%, transparent);
		color: var(--color-accent);
	}
</style>
