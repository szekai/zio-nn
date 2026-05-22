# Publishing zio-nn to Maven Central

## Prerequisites

1. **Sonatype JIRA account**: Sign up at [issues.sonatype.org](https://issues.sonatype.org)
2. **Claim `io.github.szekai` group ID**: Create a JIRA ticket requesting `io.github.szekai` namespace
3. **GPG key**: Generate with `gpg --gen-key`, publish with `gpg --keyserver keyserver.ubuntu.com --send-keys YOUR_KEY_ID`

## GPG Setup

```bash
# Generate key
gpg --gen-key
# Use: RSA 4096, no expiry, real name + email

# List keys to get your KEY_ID
gpg --list-keys --keyid-format SHORT
# Example output: pub rsa4096/ABC12345

# Publish to keyserver
gpg --keyserver keyserver.ubuntu.com --send-keys ABC12345

# Export for sbt (optional — gpg-agent handles this automatically on most systems)
# gpg --export-secret-keys > ~/.sbt/gpg/secring.asc
```

## Local Test Publish

```bash
# Publish to local Ivy (no signing needed)
sbt publishLocal

# Publish to local Maven (tests publication process)
sbt publishM2

# Verify artifacts
ls ~/.m2/repository/dev/zio/zio-nn-core_3/0.5.0/
```

## Sonatype Staging (Dry Run)

```bash
# Stage to Sonatype without releasing
sbt publishSigned
sbt sonatypeBundleRelease  # use sonatypeDrop to abort
```

## Release Steps

```bash
# 1. Set version in build.sbt (remove -SNAPSHOT)
# 2. Commit and tag
git add build.sbt
git commit -m "Release v0.5.0"
git tag -a v0.5.0 -m "zio-nn v0.5.0"
git push origin main --tags

# 3. Publish signed artifacts
sbt publishSigned

# 4. Release to Maven Central
sbt sonatypeBundleRelease

# 5. Bump version for next development cycle
# Edit build.sbt: version := "0.5.1-SNAPSHOT"
git add build.sbt
git commit -m "Bump to v0.5.1-SNAPSHOT"
git push
```

## Maven Central Coordinates

After release, dependencies are available at:

```scala
// build.sbt
libraryDependencies ++= Seq(
  "io.github.szekai" %% "zio-nn-core" % "0.5.0",
  "io.github.szekai" %% "zio-nn-djl"  % "0.5.0",
  // or "io.github.szekai" %% "zio-nn-dl4j" % "0.5.0"
)
```

Searchable at: https://search.maven.org/artifact/io.github.szekai/zio-nn-core_3

## CI/CD (GitHub Actions)

```yaml
# .github/workflows/release.yml
name: Release
on:
  push:
    tags: ['v*']
jobs:
  publish:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-java@v4
        with:
          java-version: '21'
          distribution: 'temurin'
      - uses: sbt/setup-sbt@v1
      - name: Import GPG key
        run: echo "$GPG_SECRET_KEY" | gpg --import
        env:
          GPG_SECRET_KEY: ${{ secrets.GPG_SECRET_KEY }}
      - name: Publish
        run: sbt publishSigned sonatypeBundleRelease
        env:
          SONATYPE_USERNAME: ${{ secrets.SONATYPE_USERNAME }}
          SONATYPE_PASSWORD: ${{ secrets.SONATYPE_PASSWORD }}
```

## Troubleshooting

| Problem | Fix |
|---------|-----|
| `gpg: signing failed: Inappropriate ioctl` | `export GPG_TTY=$(tty)` |
| `401 Unauthorized` | Check SONATYPE_USERNAME/PASSWORD in `~/.sbt/sonatype_credentials` |
| `Already claimed` | Group ID already taken — choose a different one |
| `No public key` | Run `gpg --keyserver keyserver.ubuntu.com --send-keys KEY_ID` |
