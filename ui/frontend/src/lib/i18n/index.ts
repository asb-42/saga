/**
 * Internationalization (i18n) module for SAGA Research Lab.
 * Simple translation system with English as default.
 */

import en from './en.json';

type TranslationKeys = typeof en;

class I18n {
	private locale = 'en';
	private translations: TranslationKeys = en;

	setLocale(locale: string) {
		this.locale = locale;
		// For now, only English is supported
		if (locale !== 'en') {
			console.warn(`Locale "${locale}" not supported, falling back to English`);
			this.locale = 'en';
		}
	}

	getLocale(): string {
		return this.locale;
	}

	/**
	 * Get a translation by dot-separated path.
	 * Example: t('nav.dashboard') returns "Dashboard"
	 */
	t(path: string, params?: Record<string, string | number>): string {
		const keys = path.split('.');
		let value: any = this.translations;

		for (const key of keys) {
			if (value && typeof value === 'object' && key in value) {
				value = value[key];
			} else {
				console.warn(`Translation key not found: ${path}`);
				return path;
			}
		}

		if (typeof value !== 'string') {
			console.warn(`Translation value is not a string: ${path}`);
			return path;
		}

		// Replace parameters like {count}
		if (params) {
			return value.replace(/\{(\w+)\}/g, (_, paramKey) => {
				return params[paramKey]?.toString() ?? `{${paramKey}}`;
			});
		}

		return value;
	}

	/**
	 * Get all translations for a namespace.
	 * Example: tNamespace('nav') returns { dashboard: "Dashboard", ... }
	 */
	tNamespace(namespace: string): Record<string, string> {
		const keys = namespace.split('.');
		let value: any = this.translations;

		for (const key of keys) {
			if (value && typeof value === 'object' && key in value) {
				value = value[key];
			} else {
				return {};
			}
		}

		if (typeof value !== 'object') {
			return {};
		}

		return value as Record<string, string>;
	}
}

export const i18n = new I18n();
export const t = i18n.t.bind(i18n);
