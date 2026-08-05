/**
 * ESLint flat config for KISS Sorcar VS Code extension.
 * Uses Google TypeScript Style (gts) rules for TypeScript and JavaScript,
 * with project-specific overrides for the VS Code extension environment.
 */
import eslint from "@eslint/js";
import tseslint from "typescript-eslint";
import prettierConfig from "eslint-config-prettier";
import pluginN from "eslint-plugin-n";
import pluginPrettier from "eslint-plugin-prettier";

export default [
  {
    ignores: [
      "out/**",
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
  ...tseslint.configs.recommended.map((cfg) => ({
    ...cfg,
    files: ["src/**/*.ts"],
  })),
  {
    files: ["src/**/*.ts"],
    languageOptions: {
      parser: tseslint.parser,
      parserOptions: {
        ecmaVersion: 2018,
        sourceType: "module",
        project: "./tsconfig.json",
      },
    },
    rules: {
      "@typescript-eslint/ban-ts-comment": "warn",
      "@typescript-eslint/no-floating-promises": "error",
      "@typescript-eslint/no-non-null-assertion": "off",
      "@typescript-eslint/no-use-before-define": "off",
      "@typescript-eslint/no-warning-comments": "off",
      "@typescript-eslint/no-empty-function": "off",
      "@typescript-eslint/no-var-requires": "off",
      "@typescript-eslint/no-require-imports": "off",
      "@typescript-eslint/no-empty-object-type": "off",
      "@typescript-eslint/explicit-function-return-type": "off",
      "@typescript-eslint/explicit-module-boundary-types": "off",
      "@typescript-eslint/ban-types": "off",
      "@typescript-eslint/camelcase": "off",
      "@typescript-eslint/no-explicit-any": "warn",
      "n/no-missing-import": "off",
      "n/no-empty-function": "off",
      "n/no-unsupported-features/es-syntax": "off",
      "n/no-missing-require": "off",
      "n/shebang": "off",
      "no-dupe-class-members": "off",
      "no-empty": ["error", {allowEmptyCatch: true}],
      "require-atomic-updates": "off",
      "@typescript-eslint/no-unused-vars": [
        "error",
        {argsIgnorePattern: "^_", varsIgnorePattern: "^_"},
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
        location: "readonly",
        console: "readonly",
        fetch: "readonly",
        crypto: "readonly",
        acquireVsCodeApi: "readonly",
        createSorcarApi: "readonly",
        SORCAR_API_COMMANDS: "readonly",
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
