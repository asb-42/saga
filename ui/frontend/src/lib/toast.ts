/**
 * Toast notification store for SAGA Research Lab.
 */

export interface Toast {
	id: number;
	message: string;
	type: 'info' | 'success' | 'warning' | 'error';
	duration?: number;
}

let toastId = 0;
let toasts = $state<Toast[]>([]);

export function addToast(
	message: string,
	type: Toast['type'] = 'info',
	duration: number = 5000
): void {
	const id = ++toastId;
	const toast: Toast = { id, message, type, duration };
	toasts = [...toasts, toast];

	if (duration > 0) {
		setTimeout(() => {
			removeToast(id);
		}, duration);
	}
}

export function removeToast(id: number): void {
	toasts = toasts.filter(t => t.id !== id);
}

export function getToasts(): Toast[] {
	return toasts;
}
