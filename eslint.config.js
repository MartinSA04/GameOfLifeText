import js from "@eslint/js";
import globals from "globals";
import tseslint from "typescript-eslint";

export default tseslint.config(
  { ignores: ["dist/**", "node_modules/**", ".venv/**"] },
  js.configs.recommended,
  ...tseslint.configs.recommendedTypeChecked,
  {
    languageOptions: {
      parserOptions: {
        // One project per lib: the page gets DOM, the worker gets WebWorker.
        project: ["./tsconfig.test.json", "./tsconfig.worker.json"],
        tsconfigRootDir: import.meta.dirname
      }
    },
    rules: {
      eqeqeq: ["error", "smart"],
      "no-console": ["warn", { allow: ["error", "warn"] }],
      "object-shorthand": "error",
      "@typescript-eslint/consistent-type-imports": "error",
      "@typescript-eslint/no-non-null-assertion": "error"
    }
  },
  {
    files: ["web/src/**/*.ts"],
    languageOptions: { globals: globals.browser }
  },
  {
    files: ["web/engine-worker.ts"],
    languageOptions: { globals: globals.worker }
  },
  // Config files: plain JS, and outside every tsconfig, so type-aware rules
  // (and the parser's project lookup) have to be switched off for them.
  { files: ["**/*.js"], ...tseslint.configs.disableTypeChecked },
  { files: ["**/*.js"], languageOptions: { globals: globals.node } }
);
