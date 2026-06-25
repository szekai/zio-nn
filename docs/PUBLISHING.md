# Publishing zio-nn to Maven Central

## Prerequisites

1. **Sonatype Central account**: Sign up at [central.sonatype.com](https://central.sonatype.com)
2. **Generate a Sonatype user token**: Central → Your Name → User Tokens → Generate
3. **GPG key**: Generate with `gpg --gen-key`, publish to a keyserver (see below)

## GPG Setup

```bash
# Generate key (RSA 4096, no expiry, use szekai@users.noreply.github.com)
gpg --gen-key

# List keys to get your KEY_ID
gpg --list-keys --keyid-format SHORT
# Example: pub rsa4096/ABC12345

# Publish to keyserver so Maven Central can verify
gpg --keyserver keyserver.ubuntu.com --send-keys ABC12345

# Export the private key in ASCII-armored format (for GitHub secret)
gpg --armor --export-secret-keys ABC12345 | base64
# → copy the output, this is your PGP_SECRET
```

## Setting Up GitHub Secrets

The CI pipeline (`.github/workflows/ci.yml`) uses four repository secrets.
Add them in GitHub → Settings → Secrets and variables → Actions:

| Secret | Value | How to get |
|--------|-------|------------|
| `PGP_PASSPHRASE` | GPG key passphrase | The passphrase you used when generating the GPG key |
| `PGP_SECRET` | Base64-encoded private key | `gpg --armor --export-secret-keys KEY_ID \| base64` |
| `SONATYPE_USERNAME` | Sonatype user token username | Sonatype Central → User Tokens |
| `SONATYPE_PASSWORD` | Sonatype user token password | Sonatype Central → User Tokens |

## Release via CI (recommended)

Version is managed automatically by `sbt-dynver` from git tags.
No version string in `build.sbt` — just tag and push:

```bash
# Ensure tests pass
sbt test

# Tag the release (version derived from tag name)
git tag v0.9.0
git push origin v0.9.0

# GitHub Actions runs: test → release
# Artifact lands on Maven Central in ~10 minutes
```

The CI workflow (`.github/workflows/ci.yml`) handles everything:
- `test` job: runs `sbt test` on every PR/push
- `release` job: runs `sbt ci-release` when a `v*` tag is pushed

## Local Test Publish

```bash
# Publish to local Ivy (no signing needed)
sbt publishLocal

# Publish to local Maven (tests publication process)
sbt publishM2

# Verify artifacts
ls ~/.m2/repository/io/github/szekai/zio-nn-core_3/0.8.0/
```

## Manual Sonatype Staging (if CI is unavailable)

```bash
# Stage to Sonatype without releasing
CI=true sbt ci-release

# This does: publishSigned → sonatypeBundleRelease
# To abort before release: use sonatypeDrop instead
```

## Maven Central Coordinates

```scala
// build.sbt
libraryDependencies ++= Seq(
  "io.github.szekai" %% "zio-nn-core"  % "<version>",
  "io.github.szekai" %% "zio-nn-djl"   % "<version>",
  // or "io.github.szekai" %% "zio-nn-dl4j"  % "<version>"
  // or "io.github.szekai" %% "zio-nn-storch" % "<version>"
)
```

Searchable at: https://search.maven.org/artifact/io.github.szekai/zio-nn-core_3

## Troubleshooting

| Problem | Fix |
|---------|-----|
| `gpg: signing failed: Inappropriate ioctl` | `export GPG_TTY=$(tty)` |
| `401 Unauthorized` | Check SONATYPE_USERNAME/PASSWORD in GitHub secrets |
| `No public key` | Run `gpg --keyserver keyserver.ubuntu.com --send-keys KEY_ID` |
| `sbt-dynver` wrong version | Ensure the latest tag matches `v<semver>` (run `git tag --sort=-v:refname`) |
| Release job skipped | Tags must match `v*` pattern (e.g., `v0.9.0` not `0.9.0`) |
