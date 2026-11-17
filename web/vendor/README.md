# Vendored browser libraries

- `recogito-text-4.3.7.min.js`, `recogito-text-4.3.7.css`: [@recogito/text-annotator](https://github.com/recogito/text-annotator-js)
  4.3.7 (BSD-3-Clause, `RECOGITO-LICENSE`), bundled with its dependencies into one IIFE (`window.RecogitoText`
  exposing `createTextAnnotator` and `W3CTextFormat`) with esbuild 0.24.0:
  `export { createTextAnnotator, W3CTextFormat } from '@recogito/text-annotator'` →
  `esbuild entry.js --bundle --format=iife --global-name=RecogitoText --minify`.
