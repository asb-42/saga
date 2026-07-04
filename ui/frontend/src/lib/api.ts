/**
 * API client for SAGA Research Lab Backend.
 * Automatically uses the current hostname for LAN access.
 */

const API_BASE = `http://${window.location.hostname}:8420`;

export async function apiFetch(path: string, options?: RequestInit): Promise<Response> {
	return fetch(`${API_BASE}${path}`, {
		...options,
		headers: {
			'Content-Type': 'application/json',
			...options?.headers,
		},
	});
}

export function apiSSE(path: string): EventSource {
	return new EventSource(`${API_BASE}${path}`);
}

export { API_BASE };
