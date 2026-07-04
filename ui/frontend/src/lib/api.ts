/**
 * API client for SAGA Research Lab Backend.
 * Uses browser hostname for LAN access, falls back to localhost for SSR.
 */

import { browser } from '$app/environment';

function getApiBase(): string {
	if (browser) {
		return `http://${window.location.hostname}:8420`;
	}
	// SSR fallback — won't actually be called during SSR
	return 'http://localhost:8420';
}

export function apiFetch(path: string, options?: RequestInit): Promise<Response> {
	return fetch(`${getApiBase()}${path}`, {
		...options,
		headers: {
			'Content-Type': 'application/json',
			...options?.headers,
		},
	});
}

export function apiSSE(path: string): EventSource | null {
	if (!browser) return null;
	return new EventSource(`${getApiBase()}${path}`);
}
