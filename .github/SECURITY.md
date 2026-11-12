# Security Policy

## Supported versions

Prism is developed against the latest `main` branch. Older releases are not actively patched; please upgrade to the latest release.

## Security model

**LAN-only, no authentication.**

- Both the PC server (`pc/unified_server.py`) and the Android app bind `0.0.0.0:8101` on your local network.
- There is **no auth** on either server. Any device on the same LAN can call any endpoint.
- **Do not expose ports 8101 / 8102 to the internet.** Use Tailscale, WireGuard or your LAN firewall to restrict access.
- The NSFW safety checker is disabled in the PC launchers.
- Models are downloaded lazily and never bundled; no model weights are shipped with the software.
- No telemetry, no phone-home, no cloud calls after model download.

## Reporting a vulnerability

Please email the maintainer (DracieVajko via GitHub) or open a private issue. Include:

- Prism version (`VERSION` file / `GET /v1/settings`)
- Platform (PC Windows/Linux/macOS or Android version + device)
- Steps to reproduce
- Expected vs. actual behavior

We do not offer bounties, but we will acknowledge and fix valid reports as quickly as possible.

## What is out of scope

- Exposing Prism ports directly to the public internet (against the documented security model)
- Modifications you make to the source (custom forks, third-party models)
- Physical access to the device running Prism
