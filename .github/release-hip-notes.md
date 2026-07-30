# audio.cpp for Windows + AMD GPU (HIP/ROCm)

Prebuilt `audiocpp_cli` / `audiocpp_server` binaries for AMD GPUs.
**No ROCm/HIP SDK installation required** — the ROCm runtime DLLs and GPU kernel
libraries are bundled in the package. A recent AMD Adrenalin driver is the only
prerequisite. A CPU fallback backend is included in the same binaries
(`--backend cpu`).

## Which package do I need?

| Package | ROCm | Supported GPUs (gfx) |
|---|---|---|
| `audiocpp-bin-win-hip-rocm7.1-x64.zip` | 7.1 | gfx1100, gfx1101, gfx1103, gfx1150, gfx1151, gfx1200, gfx1201 |
| `audiocpp-bin-win-hip-rocm6.4-x64.zip` | 6.4 | gfx1100, gfx1101, **gfx1102**, gfx1103, gfx1150, gfx1151, gfx1200, gfx1201 |

- **RX 7600 (gfx1102): use the ROCm 6.4 package.** ROCm 7.1 ships no hipBLASLt
  kernels for gfx1102, so the 7.1 package cannot serve it.
- **Radeon 780M / Strix Point / Strix Halo iGPUs** (gfx1103 / gfx1150 / gfx1151):
  either package works.
- **RDNA4 (RX 9070 series, gfx1200/1201)**: either works; the 7.1 package is
  recommended (better-tuned RDNA4 kernels).

## Quick start

```powershell
# TTS via HIP backend
.\audiocpp_cli.exe --backend hip --task tts --family <family> --model C:\path\to\model [options]

# Server
.\audiocpp_server.exe --config C:\path\to\server.json
```

`--backend rocm` is accepted as an alias for `--backend hip`.

Keep all DLL files and the `rocblas\` and `hipblaslt\` directories next to the
`.exe` files — the math libraries locate their kernel files relative to the DLL
path. If GPU initialization fails, update your AMD driver first.

## Notes

- **Versioning**: HIP releases are versioned to match the upstream audio.cpp
  Windows releases (e.g. this `v0.4.3` corresponds to upstream `release-0.4.3`),
  built from the fork's `main` synced to upstream at release time. For CPU/CUDA
  packages and the upstream release history, see
  <https://github.com/0xShug0/audio.cpp/releases>.
- Built on GitHub Actions from this fork's `main` (workflow:
  `.github/workflows/release-hip.yml`); see
  `docs/build/windows-hip-distribution.md` for the full build/packaging story.
- HIP build support itself is merged upstream
  ([0xShug0/audio.cpp#153](https://github.com/0xShug0/audio.cpp/pull/153));
  these prebuilt packages are published from the fork because upstream does not
  maintain AMD releases (no hardware to test).
- Windows 10/11 x64 only. Model files are not included; download them separately
  per `docs/models/`.
