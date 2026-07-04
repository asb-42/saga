import { describe, it, expect, vi, beforeEach } from 'vitest';

describe('Toast Store', () => {
	beforeEach(() => {
		vi.clearAllMocks();
	});

	it('addToast is exported', async () => {
		const toast = await import('$lib/toast');
		expect(typeof toast.addToast).toBe('function');
	});

	it('removeToast is exported', async () => {
		const toast = await import('$lib/toast');
		expect(typeof toast.removeToast).toBe('function');
	});

	it('getToasts is exported', async () => {
		const toast = await import('$lib/toast');
		expect(typeof toast.getToasts).toBe('function');
	});

	it('subscribeToToasts is exported', async () => {
		const toast = await import('$lib/toast');
		expect(typeof toast.subscribeToToasts).toBe('function');
	});

	it('addToast creates a toast', async () => {
		const { addToast, getToasts, removeToast } = await import('$lib/toast');
		const toasts = getToasts();
		const initialCount = toasts.length;
		
		addToast('Test', 'info');
		const updated = getToasts();
		expect(updated.length).toBe(initialCount + 1);
		expect(updated[updated.length - 1].message).toBe('Test');
		
		// Cleanup
		removeToast(updated[updated.length - 1].id);
	});

	it('removeToast removes a toast', async () => {
		const { addToast, getToasts, removeToast } = await import('$lib/toast');
		addToast('To remove', 'info');
		const toasts = getToasts();
		const id = toasts[toasts.length - 1].id;
		
		removeToast(id);
		const updated = getToasts();
		expect(updated.find((t: any) => t.id === id)).toBeUndefined();
	});

	it('subscribeToToasts returns unsubscribe function', async () => {
		const { subscribeToToasts } = await import('$lib/toast');
		const unsub = subscribeToToasts(() => {});
		expect(typeof unsub).toBe('function');
		unsub();
	});
});
