// @ts-check
import { defineConfig } from 'astro/config';
import tailwindcss from '@tailwindcss/vite';

// https://astro.build/config
// Static output (MPA) — every route renders to real HTML at build time for SEO.
export default defineConfig({
  vite: {
    plugins: [tailwindcss()],
  },
});
