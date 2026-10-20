/**
 * ESLint flat config for the KISS Sorcar webapp (media/): the
 * recommended rules plus prettier (gts's .prettierrc), with browser
 * globals.  Formatting is prettier's alone; eslint-config-prettier
 * turns the overlapping stylistic rules off.
 */
import eslint from "@eslint/js";
import prettierConfig from "eslint-config-prettier";
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
      prettier: pluginPrettier,
    },
    rules: {
      "prettier/prettier": "error",
      "prefer-const": "error",
      "prefer-arrow-callback": "error",
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
        FileReader: "readonly",
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
