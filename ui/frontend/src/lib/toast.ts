/**
 * Toast notification store for SAGA Research Lab.
 * Uses plain JS — no Svelte runes (not allowed in .ts files).
 */

export interface Toast {
	id: number;
	message: string;
	type: 'info' | 'success' | 'warning' | 'error';
	duration?: number;
}

let toastId = 0;
let toasts: Toast[] = [];
let listeners: Array<(toasts: Toast[]) => void> = [];

export function addToast(
	message: string,
	type: Toast['type'] = 'info',
	duration: number = 5000
): void {
	const id = ++toastId;
	const toast: Toast = { id, message, type, duration };
	toasts = [...toasts, toast];
	notify();

	if (duration > 0) {
		setTimeout(() => {
			removeToast(id);
		}, duration);
	}
}

export function removeToast(id: number): void {
	toasts = toasts.filter(t => t.id !== id);
	notify();
}

export function getToasts(): Toast[] {
	return toasts;
}

export function subscribeToToasts(callback: (toasts: Toast[]) => void): () => void {
	listeners = [...listeners, callback];
	return () => {
		listeners = listeners.filter(l => l !== callback);
	};
}

function notify(): void {
	for (const listener of listeners) {
		listener(toasts);
	}
}
