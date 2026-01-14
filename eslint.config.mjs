// Root ESLint flat config for the repo
// - Lints plain JS in analysis/static
// - Provides Jest globals for frontend tests

import js from "@eslint/js";
import globals from "globals";
import jestPlugin from "eslint-plugin-jest";
import prettierConfig from "eslint-config-prettier";
import prettierPlugin from "eslint-plugin-prettier";
import sonarjsPlugin from "eslint-plugin-sonarjs";

export default [
  // Base recommended rules for all JS
  js.configs.recommended,

  // Disable rules that conflict with Prettier formatting
  prettierConfig,

  // Surface Prettier formatting issues through ESLint
  {
    plugins: {
      prettier: prettierPlugin,
      sonarjs: sonarjsPlugin,
    },
    rules: {
      "prettier/prettier": "error",
      "sonarjs/cognitive-complexity": ["warn", 15],
      "sonarjs/no-all-duplicated-branches": "warn",
      "sonarjs/no-identical-functions": "warn",
      "sonarjs/no-duplicate-string": ["warn", { threshold: 3 }],
    },
  },

  // Global ignores
  {
    ignores: [
      "**/node_modules/**",
      "**/dist/**",
      "logs/**",
      "results/**",
      "env-delusions/**",
      "annotation_outputs/**",
      "annotations/**",
      "transcripts_data/**",
      "transcripts/**",
      "transcripts_de_ided/**",
      "analysis/data/**",
      "analysis/figures/**",
      "scripts/**",
      "src/**",
    ],
  },


  // plain browser JS
  {
    files: ["analysis/**/*.js"],
    languageOptions: {
      globals: {
        ...globals.browser,
      },
    },
  },
];
