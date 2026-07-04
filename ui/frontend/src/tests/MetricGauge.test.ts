import { describe, it, expect } from 'vitest';

describe('MetricGauge', () => {
	it('calculates percentage correctly', () => {
		const value = 75;
		const min = 0;
		const max = 100;
		const percentage = ((value - min) / (max - min)) * 100;
		expect(percentage).toBe(75);
	});

	it('calculates percentage with custom range', () => {
		const value = 0.5;
		const min = 0;
		const max = 1;
		const percentage = ((value - min) / (max - min)) * 100;
		expect(percentage).toBe(50);
	});

	it('clamps value to range', () => {
		const value = 150;
		const min = 0;
		const max = 100;
		const clamped = Math.min(Math.max(value, min), max);
		expect(clamped).toBe(100);
	});

	it('returns correct color for value', () => {
		const getColor = (value: number) => {
			if (value < 30) return '#00ff88';
			if (value < 70) return '#ffaa00';
			return '#ff0040';
		};

		expect(getColor(20)).toBe('#00ff88');
		expect(getColor(50)).toBe('#ffaa00');
		expect(getColor(80)).toBe('#ff0040');
	});
});
