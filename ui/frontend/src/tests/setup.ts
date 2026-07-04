import '@testing-library/jest-dom';

// Mock window.location
Object.defineProperty(window, 'location', {
	value: {
		hostname: 'localhost',
		protocol: 'http:',
		port: '5173'
	},
	writable: true
});

// Mock EventSource
class MockEventSource {
	url: string;
	onmessage: ((event: MessageEvent) => void) | null = null;
	onerror: ((event: Event) => void) | null = null;
	close = vi.fn();

	constructor(url: string) {
		this.url = url;
	}
}

Object.defineProperty(window, 'EventSource', {
	value: MockEventSource,
	writable: true
});

// Mock fetch
global.fetch = vi.fn().mockResolvedValue({
	ok: true,
	json: () => Promise.resolve({})
});
