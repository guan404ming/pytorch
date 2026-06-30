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

# Encode the TheRock ROCm runtime + sysdeps lib dirs into the torch shared
# objects of a freshly-built wheel so the fix travels with the wheel.
#
# TheRock ships OS-side sysdeps (libdrm, liblzma, ...) under
# .../lib/rocm_sysdeps/lib and the torch binaries built against it carry NEEDED
# entries such as librocm_sysdeps_liblzma.so.5. Those dirs are not on the
# default loader search path, so `import torch` only works when LD_LIBRARY_PATH
# (via /etc/rocm_env.sh) points at them -- which is NOT the case for e.g. the
# distributed pre-flight that runs python directly. Encode them as RPATH on the
# torch .so files instead (preserving torch's own $ORIGIN entries), mirroring
# how .ci/manywheel/repair_wheel.py / cuda_rpaths() encode RPATHs on the bundled
# wheels, so the wheel is self-resolving regardless of the environment.
#
# Two ROCm layouts are supported so this works before AND after the tarball ->
# wheels migration (pytorch/pytorch#188429) without depending on /opt/rocm:
#   * TheRock wheels (pytorch/pytorch#188429): the `rocm` pip package unpacks
#     under <site-packages>/_rocm_sdk_core (verified by torch/utils/cpp_extension.py),
#     a sibling of torch/ in site-packages -> reachable via an $ORIGIN-relative
#     RPATH so the produced wheel stays self-contained.
#   * Tarball install (current rocm-nightly): libs live at /opt/rocm/lib and
#     /opt/rocm/lib/rocm_sysdeps/lib -> added as an absolute RPATH only if that
#     dir exists.
# Both entries may be added together and are harmless when one layout is absent,
# so neither breaks the other across the migration. No-ops (and patches nothing)
# when neither layout is present, e.g. the apt-based ROCm path.
# Usage: patch_rocm_sysdeps_rpath <wheel_dir>
patch_rocm_sysdeps_rpath() {
  local wheel_dir="${1:?patch_rocm_sysdeps_rpath: wheel directory required}"

  local extra_rpath=""

  # TheRock wheel layout: derive an $ORIGIN-relative path from torch/lib to the
  # _rocm_sdk_core lib dir. relpath is computed at build time so the wheel does
  # not hardcode the absolute site-packages location.
  local rocm_rel
  rocm_rel="$(python3 -c '
import importlib.util, os, sysconfig
spec = importlib.util.find_spec("_rocm_sdk_core")
if spec and spec.origin:
    root_lib = os.path.join(os.path.dirname(spec.origin), "lib")
    torch_lib = os.path.join(sysconfig.get_path("purelib"), "torch", "lib")
    print(os.path.relpath(root_lib, torch_lib))
' 2>/dev/null || true)"
  if [ -n "$rocm_rel" ]; then
    extra_rpath="\$ORIGIN/${rocm_rel}:\$ORIGIN/${rocm_rel}/rocm_sysdeps/lib"
  fi

  # Tarball layout: absolute /opt/rocm dirs (only present in the tarball world).
  if [ -d /opt/rocm/lib/rocm_sysdeps/lib ]; then
    extra_rpath="${extra_rpath:+${extra_rpath}:}/opt/rocm/lib/rocm_sysdeps/lib:/opt/rocm/lib"
  fi

  if [ -z "$extra_rpath" ]; then
    echo "patch_rocm_sysdeps_rpath: no TheRock ROCm sysdeps layout detected; skipping"
    return 0
  fi
  echo "patch_rocm_sysdeps_rpath: appending RPATH entries: ${extra_rpath}"

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
