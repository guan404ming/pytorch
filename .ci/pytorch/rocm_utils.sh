#!/bin/bash
# ROCm-specific utility functions shared across CI scripts

ROCM_UTILS_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Build rocm-composable-kernel (ck4inductor) wheel
# Usage: build_rocm_ck_wheel <output_dir>
build_rocm_ck_wheel() {
  local raw_output_dir="${1:?build_rocm_ck_wheel: output directory required}"
  mkdir -p "$raw_output_dir"
  local output_dir
  output_dir="$(cd "$raw_output_dir" && pwd)"
  if [[ -z "$output_dir" ]]; then
    echo "build_rocm_ck_wheel: failed to resolve output directory '$raw_output_dir'" >&2
    return 1
  fi

  echo "Building rocm-composable-kernel (ck4inductor) wheel at $(date)"

  local pin_file="${ROCM_UTILS_DIR}/../docker/ci_commit_pins/rocm-composable-kernel.txt"
  if [[ ! -f "$pin_file" ]]; then
    echo "build_rocm_ck_wheel: pin file not found at $pin_file" >&2
    return 1
  fi
  local ck_commit
  ck_commit=$(tr -d '[:space:]' < "$pin_file")
  echo "CK commit: $ck_commit"

  # Use a fresh tmp dir keyed by commit prefix; remove any stale state from prior runs
  # so `git init` / `git fetch` start clean.
  local ck_dir="/tmp/ck-${ck_commit:0:12}"
  rm -rf "$ck_dir"

  # Cleanup runs on every exit path from this function (success, return, or set -e abort
  # via the caller's ERR/EXIT), via the explicit rm at the end and via the trap below.
  # `trap RETURN` alone is unreliable under `set -e` because the shell may exit before
  # the function returns, so we also rm at the end of the success path. Expand $ck_dir
  # at trap-set time (rather than trap-fire time) since it is a local that may go out
  # of scope, and its value never changes within this function.
  # shellcheck disable=SC2064
  trap "rm -rf '$ck_dir'" RETURN

  git init "$ck_dir"
  pushd "$ck_dir" >/dev/null || return 1
  git fetch --depth 1 https://github.com/ROCm/composable_kernel.git "$ck_commit"
  git checkout FETCH_HEAD
  python -m build --wheel --no-isolation --outdir "$output_dir"
  popd >/dev/null || return 1

  # Verify the wheel actually landed so downstream copies don't silently skip it.
  if ! compgen -G "$output_dir/rocm_composable_kernel*.whl" >/dev/null; then
    echo "build_rocm_ck_wheel: no rocm_composable_kernel wheel produced in $output_dir" >&2
    return 1
  fi

  rm -rf "$ck_dir"
  echo "Finished building rocm-composable-kernel (ck4inductor) wheel at $(date)"
}

# Encode the theRock nightly sysdeps + ROCm lib dirs into the torch shared
# objects of a freshly-built wheel so the fix travels with the wheel.
#
# theRock nightly ROCm is installed under /opt/rocm with its OS-side sysdeps
# (libdrm, liblzma, ...) under /opt/rocm/lib/rocm_sysdeps/lib. The torch
# binaries built against it carry NEEDED entries such as
# librocm_sysdeps_liblzma.so.5, but neither that dir nor /opt/rocm/lib is on the
# default loader search path. As a result `import torch` only works when
# LD_LIBRARY_PATH (via /etc/rocm_env.sh) or a global ldconfig entry points at
# them -- which is not the case for e.g. the distributed pre-flight that runs
# python directly.
#
# Mirroring how .ci/manywheel/repair_wheel.py encodes RPATHs on the bundled ROCm
# nightly wheels, we append these dirs to the RPATH of the torch .so files in
# the produced wheel (preserving torch's own $ORIGIN entries) so it is
# self-resolving regardless of the environment. Guarded by the caller on the
# existence of the sysdeps dir so the apt-based ROCm path is unaffected.
# Usage: patch_rocm_sysdeps_rpath <wheel_dir>
patch_rocm_sysdeps_rpath() {
  local wheel_dir="${1:?patch_rocm_sysdeps_rpath: wheel directory required}"
  local extra_rpath="/opt/rocm/lib/rocm_sysdeps/lib:/opt/rocm/lib"

  # `wheel` (unpack/pack) and `patchelf` may not be present in the build image.
  python -m pip install wheel patchelf

  local tmp
  tmp="$(mktemp -d)"
  # shellcheck disable=SC2064
  trap "rm -rf '$tmp'" RETURN

  local whl
  for whl in "${wheel_dir}"/torch-*.whl; do
    [ -f "$whl" ] || continue
    echo "Encoding ROCm sysdeps RPATH into $(basename "$whl")"
    local unpack_dir="${tmp}/unpack"
    rm -rf "$unpack_dir"
    wheel unpack "$whl" -d "$unpack_dir"

    local torch_dir
    torch_dir="$(echo "${unpack_dir}"/torch-*/torch)"
    local sofile cur
    for sofile in "${torch_dir}"/*.so* "${torch_dir}"/lib/*.so*; do
      [ -f "$sofile" ] || continue
      cur="$(patchelf --print-rpath "$sofile" 2>/dev/null || true)"
      if [ -n "$cur" ]; then
        patchelf --force-rpath --set-rpath "${cur}:${extra_rpath}" "$sofile"
      else
        patchelf --force-rpath --set-rpath "${extra_rpath}" "$sofile"
      fi
    done

    rm -f "$whl"
    wheel pack "${unpack_dir}"/torch-*/ -d "$wheel_dir"
  done
}
