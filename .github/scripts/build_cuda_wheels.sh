#!/usr/bin/env bash
set -euo pipefail

# Reuse the supplied venv. No torch/CUDA upgrades, no parallel native builds.
export MAX_JOBS=1
export CMAKE_BUILD_PARALLEL_LEVEL=1
export NVCC_PREPEND_FLAGS="--threads 1"
export TORCH_CUDA_ARCH_LIST="8.9"
export BUILD_TARGET=cuda
export MAKEFLAGS=-j1
export OMP_NUM_THREADS=1
export CCACHE_DISABLE=1

py="$TF_PYTHON"
export PATH="$(dirname "$py"):$PATH"
out="$TF_RESULTS/wheel-artifact"
mkdir -p "$out"
log="$TF_RESULTS/native-build.log"
touch "$log"
export CUDA_HOME="$("$py" -c 'from torch.utils.cpp_extension import CUDA_HOME; print(CUDA_HOME or "")')"
if [[ -z "$CUDA_HOME" || ! -x "$CUDA_HOME/bin/nvcc" ]]; then
  echo "::error::CUDA toolkit nvcc is missing; a CUDA-enabled PyTorch wheel alone cannot compile extensions."
  exit 1
fi
export PATH="$CUDA_HOME/bin:$PATH"
"$CUDA_HOME/bin/nvcc" --version
free -h
"$py" -c 'import sys, torch; assert sys.version_info[:2]==(3,12); assert torch.__version__.split("+")[0]=="2.7.1" and torch.version.cuda=="12.8"; assert torch.cuda.get_device_capability()==(8,9); print("Python/PyTorch/CUDA/SM verified")'
if ! "$CUDA_HOME/bin/nvcc" --version | grep -q 'release 12.8'; then
  echo "::error::Expected CUDA Toolkit 12.8 to match the existing PyTorch wheel."
  exit 1
fi
if "$py" -m pip --version >/dev/null 2>&1; then
  install=("$py" -m pip install --no-deps)
elif command -v uv >/dev/null; then
  install=(uv pip install --python "$py" --no-deps)
else
  echo "::error::Neither pip in the venv nor uv is available."
  exit 1
fi

# Check lightweight prerequisites individually. Do not replace already installed packages.
for item in 'setuptools:setuptools' 'wheel:wheel' 'ninja:ninja' 'plyfile:plyfile' 'zstandard:zstandard' 'easydict:easydict' 'cv2:opencv-python-headless'; do
  module="${item%%:*}"
  package="${item#*:}"
  if ! "$py" -c "import $module" >/dev/null 2>&1; then
    "${install[@]}" "$package" >>"$log" 2>&1
  fi
done

abi="$("$py" -c 'import sys, torch, platform; print(f"cp{sys.version_info.major}{sys.version_info.minor}-torch{torch.__version__.split(chr(43))[0]}-cu{torch.version.cuda.replace(chr(46),chr(45))}-abi{int(torch.compiled_with_cxx11_abi())}-{platform.machine()}")')"
cache="$HOME/.cache/toothforge3d-wheels/$abi-sm89"
mkdir -p "$cache"
exec 9>"$cache/build.lock"
flock 9

build_wheel() {
  local name="$1" url="$2" ref="$3" subdir="$4" publish="$5" module="$6"
  local dir="$cache/$name-$ref"
  mkdir -p "$dir"
  local -a wheels=()
  shopt -s nullglob
  wheels=("$dir"/*.whl)
  if (( ${#wheels[@]} != 1 )); then
    echo "Building $name from $ref (MAX_JOBS=$MAX_JOBS)"
    local src="$dir/source"
    if [[ ! -d "$src/.git" ]]; then
      git init -q "$src"
      git -C "$src" remote add origin "$url"
    fi
    # Reuse a previously fetched exact revision; shallow SHA fetch is not
    # supported by every proxy. Fall back to fetching advertised branches/tags.
    git -C "$src" remote set-url origin "$url"
    if ! git -C "$src" cat-file -e "$ref^{commit}" 2>/dev/null; then
      echo "$name: fetching pinned public source"
      if ! git -C "$src" -c http.version=HTTP/1.1 fetch --no-tags --depth=1 origin "$ref" >>"$log" 2>&1; then
        echo "$name: direct revision fetch failed; trying advertised refs"
        tail -n 12 "$log"
        if ! git -C "$src" -c http.version=HTTP/1.1 fetch --tags origin '+refs/heads/*:refs/remotes/origin/*' >>"$log" 2>&1; then
          echo "::error::$name source fetch failed"
          tail -n 25 "$log"
          return 1
        fi
      else
        # A successful fetch of a tag may only populate FETCH_HEAD.
        if [[ "$ref" == refs/tags/* ]]; then
          git -C "$src" update-ref "$ref" FETCH_HEAD
        fi
      fi
    fi
    if ! git -C "$src" checkout --detach "$ref" >>"$log" 2>&1; then
      echo "::error::$name pinned revision checkout failed"
      tail -n 25 "$log"
      return 1
    fi
    echo "$name: fetching pinned submodules"
    git -C "$src" submodule sync --recursive >>"$log" 2>&1
    if ! git -C "$src" -c http.version=HTTP/1.1 submodule update --init --recursive --jobs 1 >>"$log" 2>&1; then
      tail -n 20 "$log"
      echo "$name: repairing incomplete submodule checkout in dedicated build cache"
      # Deinitialize only submodules of our disposable cached upstream source,
      # not the user's project or installed environment. Recreate nested git
      # metadata as well as worktrees; retrying alone cannot fix a broken HEAD.
      git -C "$src" submodule deinit -f --all >>"$log" 2>&1 || true
      if [[ -d "$src/.git/modules" ]]; then
        mv "$src/.git/modules" "$dir/modules-incomplete-$(date +%s)-$"
      fi
      # A failed nested checkout can leave worktrees after deinit; move each
      # declared top-level gitlink aside rather than deleting source files.
      while IFS= read -r subpath; do
        [[ -n "$subpath" ]] || continue
        if [[ -e "$src/$subpath" ]]; then
          mv "$src/$subpath" "$dir/submodule-incomplete-$(basename "$subpath")-$(date +%s)-$"
        fi
      done < <(git -C "$src" config -f .gitmodules --get-regexp '^submodule\..*\.path
        echo "::error::$name submodule checkout failed"
        tail -n 30 "$log"
        return 1
      fi
    fi
    local actual
    actual="$(git -C "$src" rev-parse HEAD)"
    echo "$name: source ready at $actual; starting native compiler"
    # Preserve successful wheels and reuse the source/build directory on subsequent runs.
    local staging="$dir/staging"
    mkdir -p "$staging"
    if ! (cd "$src/$subdir" && "$py" setup.py bdist_wheel --dist-dir "$staging") >>"$log" 2>&1; then
      echo "::error::$name compilation failed. Full log is local: $log"
      # Native compiler output contains no patient data; only show the last failure diagnostics.
      tail -n 35 "$log"
      return 1
    fi
    local -a built=("$staging"/*.whl)
    [[ "${#built[@]}" == 1 ]]
    mv "${built[0]}" "$dir/"
    printf '%s %s\n' "$name" "$actual" >"$dir/source-version.txt"
    wheels=("$dir"/*.whl)
  else
    echo "Reusing cached $name wheel"
  fi
  # Publish even if a later dependency or import fails; manifest records verification separately.
  if [[ "$publish" == yes ]]; then
    cp "${wheels[0]}" "$out/"
    if [[ -f "$dir/source-version.txt" ]]; then
      cat "$dir/source-version.txt" >>"$out/sources.txt"
    fi
    (cd "$out" && sha256sum ./*.whl >SHA256SUMS)
    printf '%s\n' "Python 3.12; PyTorch 2.7.1+cu128; CUDA Toolkit 12.8; Linux x86_64; SM 8.9; ABI $abi" >"$out/environment.txt"
  fi
  "${install[@]}" --force-reinstall "${wheels[0]}" >>"$log" 2>&1
  if "$py" -c "import torch; import $module" >>"$log" 2>&1; then
    echo "$name import verified"
    if [[ "$publish" == yes ]]; then
      printf '%s import passed\n' "$name" >>"$out/verification.txt"
    fi
  else
    echo "::error::$name import failed after wheel installation."
    tail -n 25 "$log"
    return 1
  fi
}

build_wheel cumesh https://github.com/JeffreyXiang/CuMesh.git 12289e1062f0603f2f0d0771b02e1395d247f26f . yes cumesh
# O-Voxel imports its upstream postprocessing module eagerly; these imports are needed even when only using FDG.
if ! "$py" -c 'import torch; from flex_gemm.ops.grid_sample import grid_sample_3d' >>"$log" 2>&1; then
  build_wheel flex_gemm https://github.com/JeffreyXiang/FlexGEMM.git 6dd94a859c26ee8246888502eada3dd8ad85532e . yes flex_gemm
fi
if ! "$py" -c 'import torch; import nvdiffrast.torch' >>"$log" 2>&1; then
  # Local runtime dependency only; do not redistribute the nvdiffrast wheel.
  build_wheel nvdiffrast https://github.com/NVlabs/nvdiffrast.git refs/tags/v0.4.0 . no nvdiffrast.torch
fi
build_wheel o_voxel https://github.com/microsoft/TRELLIS.2.git 75fbf0183001ed9876c8dbb35de6b68552ee08bd o-voxel yes o_voxel

# Verify the actual FDG CUDA decode, not merely package import.
"$py" - <<'PY' >>"$log" 2>&1
import torch
from o_voxel.convert import mesh_to_flexible_dual_grid, flexible_dual_grid_to_mesh
v = torch.tensor([[-.2,-.2,0.],[.2,-.2,0.],[0.,.2,0.],[0.,0.,.3]], dtype=torch.float32)
f = torch.tensor([[0,1,3],[1,2,3],[2,0,3]], dtype=torch.int32)
aabb = torch.tensor([[-.5,-.5,-.5],[.5,.5,.5]], dtype=torch.float32)
coords, dual, flags = mesh_to_flexible_dual_grid(v, f, grid_size=32, aabb=aabb)
offsets = (dual * 32 - coords.float()).cuda().requires_grad_(True)
vertices, faces = flexible_dual_grid_to_mesh(coords.int().cuda(), offsets, flags.bool().cuda(), split_weight=None, grid_size=32, aabb=aabb, train=False)
assert len(vertices) and len(faces)
vertices.square().mean().backward()
assert offsets.grad is not None and torch.isfinite(offsets.grad).all() and offsets.grad.abs().sum() > 0
torch.cuda.synchronize()
PY
echo "O-Voxel CUDA round trip and gradient passed" | tee -a "$out/verification.txt"
 | cut -d ' ' -f2-)
      if ! git -C "$src" -c http.version=HTTP/1.1 submodule update --init --recursive --jobs 1 >>"$log" 2>&1; then
        echo "::error::$name submodule checkout failed"
        tail -n 30 "$log"
        return 1
      fi
    fi
    local actual
    actual="$(git -C "$src" rev-parse HEAD)"
    echo "$name: source ready at $actual; starting native compiler"
    # Preserve successful wheels and reuse the source/build directory on subsequent runs.
    local staging="$dir/staging"
    mkdir -p "$staging"
    if ! (cd "$src/$subdir" && "$py" setup.py bdist_wheel --dist-dir "$staging") >>"$log" 2>&1; then
      echo "::error::$name compilation failed. Full log is local: $log"
      # Native compiler output contains no patient data; only show the last failure diagnostics.
      tail -n 35 "$log"
      return 1
    fi
    local -a built=("$staging"/*.whl)
    [[ "${#built[@]}" == 1 ]]
    mv "${built[0]}" "$dir/"
    printf '%s %s\n' "$name" "$actual" >"$dir/source-version.txt"
    wheels=("$dir"/*.whl)
  else
    echo "Reusing cached $name wheel"
  fi
  # Publish even if a later dependency or import fails; manifest records verification separately.
  if [[ "$publish" == yes ]]; then
    cp "${wheels[0]}" "$out/"
    if [[ -f "$dir/source-version.txt" ]]; then
      cat "$dir/source-version.txt" >>"$out/sources.txt"
    fi
    (cd "$out" && sha256sum ./*.whl >SHA256SUMS)
    printf '%s\n' "Python 3.12; PyTorch 2.7.1+cu128; CUDA Toolkit 12.8; Linux x86_64; SM 8.9; ABI $abi" >"$out/environment.txt"
  fi
  "${install[@]}" --force-reinstall "${wheels[0]}" >>"$log" 2>&1
  if "$py" -c "import torch; import $module" >>"$log" 2>&1; then
    echo "$name import verified"
    if [[ "$publish" == yes ]]; then
      printf '%s import passed\n' "$name" >>"$out/verification.txt"
    fi
  else
    echo "::error::$name import failed after wheel installation."
    tail -n 25 "$log"
    return 1
  fi
}

build_wheel cumesh https://github.com/JeffreyXiang/CuMesh.git 12289e1062f0603f2f0d0771b02e1395d247f26f . yes cumesh
# O-Voxel imports its upstream postprocessing module eagerly; these imports are needed even when only using FDG.
if ! "$py" -c 'import torch; from flex_gemm.ops.grid_sample import grid_sample_3d' >>"$log" 2>&1; then
  build_wheel flex_gemm https://github.com/JeffreyXiang/FlexGEMM.git 6dd94a859c26ee8246888502eada3dd8ad85532e . yes flex_gemm
fi
if ! "$py" -c 'import torch; import nvdiffrast.torch' >>"$log" 2>&1; then
  # Local runtime dependency only; do not redistribute the nvdiffrast wheel.
  build_wheel nvdiffrast https://github.com/NVlabs/nvdiffrast.git refs/tags/v0.4.0 . no nvdiffrast.torch
fi
build_wheel o_voxel https://github.com/microsoft/TRELLIS.2.git 75fbf0183001ed9876c8dbb35de6b68552ee08bd o-voxel yes o_voxel

# Verify the actual FDG CUDA decode, not merely package import.
"$py" - <<'PY' >>"$log" 2>&1
import torch
from o_voxel.convert import mesh_to_flexible_dual_grid, flexible_dual_grid_to_mesh
v = torch.tensor([[-.2,-.2,0.],[.2,-.2,0.],[0.,.2,0.],[0.,0.,.3]], dtype=torch.float32)
f = torch.tensor([[0,1,3],[1,2,3],[2,0,3]], dtype=torch.int32)
aabb = torch.tensor([[-.5,-.5,-.5],[.5,.5,.5]], dtype=torch.float32)
coords, dual, flags = mesh_to_flexible_dual_grid(v, f, grid_size=32, aabb=aabb)
offsets = (dual * 32 - coords.float()).cuda().requires_grad_(True)
vertices, faces = flexible_dual_grid_to_mesh(coords.int().cuda(), offsets, flags.bool().cuda(), split_weight=None, grid_size=32, aabb=aabb, train=False)
assert len(vertices) and len(faces)
vertices.square().mean().backward()
assert offsets.grad is not None and torch.isfinite(offsets.grad).all() and offsets.grad.abs().sum() > 0
torch.cuda.synchronize()
PY
echo "O-Voxel CUDA round trip and gradient passed" | tee -a "$out/verification.txt"
