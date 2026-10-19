/**
 * ESLint flat config for the KISS Sorcar webapp (media/).
 * Uses the Google style rules bundled with gts, with browser globals.
 */
import eslint from "@eslint/js";
import prettierConfig from "eslint-config-prettier";
import pluginN from "eslint-plugin-n";
import pluginPrettier from "eslint-plugin-prettier";

export default [
  {
    ignores: [
      "node_modules/**",
      "media/marked.min.js",
      "media/highlight.min.js",
      "media/vosk.js",
    ],
  },
  eslint.configs.recommended,
  prettierConfig,
  {
    plugins: {
      n: pluginN,
      prettier: pluginPrettier,
    },
    rules: {
      "prettier/prettier": "error",
      "block-scoped-var": "error",
      eqeqeq: "error",
      "no-var": "error",
      "prefer-const": "error",
      "eol-last": "error",
      "prefer-arrow-callback": "error",
      "no-trailing-spaces": "error",
      quotes: ["warn", "single", {avoidEscape: true}],
      "no-restricted-properties": [
        "error",
        {object: "describe", property: "only"},
        {object: "it", property: "only"},
      ],
    },
  },
  {
    files: ["media/**/*.js"],
    languageOptions: {
      globals: {
        document: "readonly",
        window: "readonly",
        setTimeout: "readonly",
        clearTimeout: "readonly",
        setInterval: "readonly",
        clearInterval: "readonly",
        requestAnimationFrame: "readonly",
        cancelAnimationFrame: "readonly",
        MutationObserver: "readonly",
        IntersectionObserver: "readonly",
        ResizeObserver: "readonly",
        HTMLElement: "readonly",
        Event: "readonly",
        KeyboardEvent: "readonly",
        MouseEvent: "readonly",
        DragEvent: "readonly",
        ClipboardEvent: "readonly",
        FileReader: "readonly",
        FormData: "readonly",
        URL: "readonly",
        Blob: "readonly",
        Image: "readonly",
        navigator: "readonly",
        localStorage: "readonly",
        CustomEvent: "readonly",
        MessageEvent: "readonly",
        location: "readonly",
        console: "readonly",
        fetch: "readonly",
        crypto: "readonly",
        acquireTransport: "readonly",
        createSorcarApi: "readonly",
        marked: "readonly",
        hljs: "readonly",
        NodeFilter: "readonly",
      },
    },
    rules: {
      "no-var": "warn",
      "block-scoped-var": "warn",
      "no-empty": ["error", {allowEmptyCatch: true}],
      "no-redeclare": "warn",
      eqeqeq: ["error", "always", {null: "ignore"}],
      "no-unused-vars": [
        "error",
        {
          argsIgnorePattern: "^_",
          varsIgnorePattern: "^_",
          caughtErrorsIgnorePattern: "^_",
        },
      ],
    },
  },
];
