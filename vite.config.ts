import { defineConfig } from 'vite'
import { resolve } from 'path'

export default defineConfig({
  build: {
    lib: {
      entry: {
        index: resolve(__dirname, 'src/index.ts'),
        'core/index': resolve(__dirname, 'src/core/index.ts'),
        'web/index': resolve(__dirname, 'src/web/index.ts'),
      },
      name: 'OpenGaze',
      formats: ['es', 'cjs'],
      fileName: (format, entryName) => `${entryName}.${format === 'es' ? 'js' : 'cjs'}`,
    },
    rollupOptions: {
      external: ['@mediapipe/tasks-vision'],
      output: {
        globals: {
          '@mediapipe/tasks-vision': 'TasksVision',
        },
      },
    },
    sourcemap: true,
  },
})
