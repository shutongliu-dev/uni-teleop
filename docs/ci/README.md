# Optional GitHub Actions

`github-actions.example.yml` runs the SDK-free launcher tests and integration
inventory check. To enable it, copy it to `.github/workflows/check.yml` using a
GitHub credential authorized to write workflows. It is a template, not an active
CI run. The current publishing credential does not include the `workflow` scope.

Native simulator and hardware tests still require their separate environments.
