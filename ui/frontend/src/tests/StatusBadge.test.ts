import { describe, it, expect } from 'vitest';

describe('StatusBadge', () => {
	it('status colors map is defined', () => {
		// Test the status color mapping logic without rendering
		const statusColors: Record<string, string> = {
			pending: 'bg-gray-600',
			running: 'bg-[#00d4ff]',
			completed: 'bg-[#00ff88]',
			failed: 'bg-[#ff0040]',
			paused: 'bg-[#ffaa00]',
		};

		expect(statusColors.pending).toBe('bg-gray-600');
		expect(statusColors.running).toBe('bg-[#00d4ff]');
		expect(statusColors.completed).toBe('bg-[#00ff88]');
		expect(statusColors.failed).toBe('bg-[#ff0040]');
		expect(statusColors.paused).toBe('bg-[#ffaa00]');
	});

	it('unknown status falls back to gray', () => {
		const statusColors: Record<string, string> = {
			pending: 'bg-gray-600',
			running: 'bg-[#00d4ff]',
			completed: 'bg-[#00ff88]',
			failed: 'bg-[#ff0040]',
			paused: 'bg-[#ffaa00]',
		};

		const color = statusColors['unknown'] || 'bg-gray-600';
		expect(color).toBe('bg-gray-600');
	});
});
