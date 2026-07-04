/**
 * Theme store for SAGA Research Lab.
 * Supports dark (default) and light themes.
 */

export type Theme = 'dark' | 'light';

let currentTheme: Theme = 'dark';
let listeners: Array<(theme: Theme) => void> = [];

function notify(): void {
	for (const listener of listeners) {
		listener(currentTheme);
	}
}

export function getTheme(): Theme {
	return currentTheme;
}

export function setTheme(theme: Theme): void {
	currentTheme = theme;
	localStorage.setItem('saga-theme', theme);
	document.documentElement.setAttribute('data-theme', theme);
	notify();
}

export function toggleTheme(): void {
	setTheme(currentTheme === 'dark' ? 'light' : 'dark');
}

export function initTheme(): void {
	const saved = localStorage.getItem('saga-theme') as Theme | null;
	if (saved && (saved === 'dark' || saved === 'light')) {
		currentTheme = saved;
	}
	document.documentElement.setAttribute('data-theme', currentTheme);
}

export function subscribeToTheme(callback: (theme: Theme) => void): () => void {
	listeners = [...listeners, callback];
	callback(currentTheme);
	return () => {
		listeners = listeners.filter(l => l !== callback);
	};
}
