---
description: Prepare a voxmd release (version, changelog, checks) and hand the tag to the user
argument-hint: <new version, e.g. 0.1.1>
---

Prepare voxmd release `$ARGUMENTS`. Publishing is triggered by pushing a `v*` tag. `.github/workflows/release.yml` then tests, builds, and publishes to PyPI after the user approves the `pypi` environment. A PyPI version can never be replaced or reused, so every step below happens before the tag.

1. **Preconditions.** Stop and report if any fails:
   - `git status` is clean and on `main`, in sync with `origin/main`.
   - `$ARGUMENTS` is a valid PEP 440 version, greater than `__version__` in `src/voxmd/__init__.py` and not already on PyPI. Check with `curl -s https://pypi.org/pypi/voxmd/json`.
   - `git tag --list "v$ARGUMENTS"` is empty.
2. **Version.** Set `__version__ = "$ARGUMENTS"` in `src/voxmd/__init__.py`. That is the only place the version lives; `pyproject.toml` reads it from there.
3. **Changelog.**
   - Add `## [$ARGUMENTS] - YYYY-MM-DD` (today) at the top of `CHANGELOG.md`, with Added / Changed / Fixed sections written from `git log v<previous>..HEAD`, in user-facing language.
   - Add the matching link reference at the bottom.
   - A pre-release (`rcN`) gets its own entry too.
4. **README.** PyPI shows the README exactly as it is at the tag, so README changes reach PyPI only with a release. Keep every link absolute (relative links break on PyPI), and keep the install instructions right for this version.
5. **Checks.**
   ```sh
   uv run pytest && uv run ruff check && uv run ruff format --check
   rm -rf /tmp/voxmd-dist && uv build -o /tmp/voxmd-dist
   uvx twine check --strict /tmp/voxmd-dist/*
   ```
   Then list both files:
   - The wheel should contain only `voxmd/`, `voxmd/templates/note.md.j2` and `dist-info`.
   - The sdist should contain only the `only-include` paths from `pyproject.toml`.
   - Neither may contain memo audio, notes, `voxmd.yaml` or `entities.json`.
6. **Dependencies.** If any pin changed since the last release, run `uvx pip-audit` on `uv export --locked --no-dev --no-hashes --no-emit-project`, and note the changes in the changelog.
7. **Stop.** Show the user the diff. Commit and push only when they ask, ending the message with the Claude co-author trailer. Then give them the tag commands; don't push the tag yourself:
   ```sh
   git tag -a v$ARGUMENTS -m "voxmd $ARGUMENTS"
   git push origin v$ARGUMENTS
   ```
   Remind them to approve the `pypi` deployment in the Actions tab. The build job fails if the tag and `voxmd --version` disagree.
8. **After publishing**, if the user asks: check that https://pypi.org/project/voxmd/ shows the version, and that `uv tool upgrade voxmd` followed by `voxmd --version` prints it. To pull a bad release, yank it on PyPI; don't delete it.
