import { describe, it, expect, vi, beforeEach } from 'vitest';

describe('API Client', () => {
	beforeEach(() => {
		vi.clearAllMocks();
	});

	it('apiFetch is exported', async () => {
		const api = await import('$lib/api');
		expect(typeof api.apiFetch).toBe('function');
	});

	it('apiSSE is exported', async () => {
		const api = await import('$lib/api');
		expect(typeof api.apiSSE).toBe('function');
	});

	it('apiFetch returns a promise', async () => {
		const { apiFetch } = await import('$lib/api');
		// Mock fetch
		global.fetch = vi.fn().mockResolvedValue({
			ok: true,
			json: () => Promise.resolve({ status: 'ok' })
		});
		
		const result = apiFetch('/api/health');
		expect(result).toBeInstanceOf(Promise);
	});

	it('apiSSE returns EventSource or null', async () => {
		const { apiSSE } = await import('$lib/api');
		const result = apiSSE('/api/pipeline/status');
		// In test env (browser=false), returns null
		expect(result === null || result instanceof EventSource).toBe(true);
	});
});
