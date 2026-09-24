import path from 'node:path'
import { fileURLToPath } from 'node:url'
import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'
import tailwindcss from '@tailwindcss/vite'
import { CONTEXT_SINGLETON_DEDUPE } from '../vite.shared'

const here = path.dirname(fileURLToPath(import.meta.url))

/**
 * The Vite config the story bundle is built with. Deliberately NOT the app's
 * `vite.config.ts` (see `.storybook/main.ts` for why). It carries only the
 * pieces a component needs to resolve and paint at all — the `@` source alias,
 * the context-carrying singleton dedupe list imported from the same module the
 * app build reads, and Tailwind.
 *
 * Tailwind is a PLUGIN here, not an ambient PostCSS step: this package is on
 * Tailwind v4 and has no `postcss.config`, so a config without
 * `@tailwindcss/vite` compiles `src/index.css` with its design tokens but
 * without a single utility class, and every story renders as unstyled text.
 */
export default defineConfig({
  plugins: [react(), tailwindcss()],
  resolve: {
    alias: {
      '@': path.resolve(here, '../src'),
    },
    dedupe: CONTEXT_SINGLETON_DEDUPE,
  },
})
