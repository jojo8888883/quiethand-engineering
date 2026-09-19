#!/usr/bin/env bash
set -euo pipefail

ROOT=/llm_jzm/dty_user/quiethand_m3
CONDA=/mnt/sdc/dty_user/miniconda3/bin/conda
CUDA_124=/usr/local/cuda-12.4
RUNTIME_ARCHIVES="$ROOT/external_data/quiethand_m3_runtime_sources"
RUNTIME_SOURCES="$ROOT/runtime_sources"

export PYTHONNOUSERSITE=1
export PYTHONSAFEPATH=1
export MAX_JOBS=8

materialize_runtime_source() {
  local archive_name="$1"
  local root_name="$2"
  local expected_bytes="$3"
  local expected_sha256="$4"
  local archive_path="$RUNTIME_ARCHIVES/$archive_name"
  local source_path="$RUNTIME_SOURCES/$root_name"
  local actual_bytes
  local actual_sha256
  local member

  test -f "$archive_path"
  test ! -L "$archive_path"
  actual_bytes="$(stat -c '%s' "$archive_path")"
  test "$actual_bytes" = "$expected_bytes"
  actual_sha256="$(sha256sum "$archive_path" | cut -d ' ' -f 1)"
  test "$actual_sha256" = "$expected_sha256"

  while IFS= read -r member; do
    case "$member" in
      "$root_name"|"$root_name/"|"$root_name/"*) ;;
      *) return 1 ;;
    esac
  done < <(tar -tzf "$archive_path")
  tar -tvzf "$archive_path" | awk '
    substr($1, 1, 1) != "-" && substr($1, 1, 1) != "d" { exit 1 }
  '

  mkdir -p "$RUNTIME_SOURCES"
  case "$source_path" in
    "$RUNTIME_SOURCES/"*) ;;
    *) return 1 ;;
  esac
  if [[ -e "$source_path" ]]; then
    rm -rf -- "$source_path"
  fi
  tar -xzf "$archive_path" --no-same-owner --no-same-permissions \
    -C "$RUNTIME_SOURCES"
  test -d "$source_path"
  printf '%s\n' "$source_path"
}

setup_raw_hand() {
  local env="$ROOT/envs/raw_hand"
  local repo="$ROOT/external_repos/quiethand_m3/HaWoR"
  local constraints="$ROOT/scripts/quiethand/m3_raw_runtime_constraints.txt"
  local requirements="$ROOT/scripts/quiethand/m3_raw_runtime_requirements.txt"
  local pytorch3d_source
  local chumpy_source
  test -x "$env/bin/python"
  test -f "$repo/requirements.txt"
  test -f "$constraints"
  test -f "$requirements"
  pytorch3d_source="$(materialize_runtime_source \
    pytorch3d-75ebeeaea0908c5527e7b1e305fbc7681382db47.tar.gz \
    pytorch3d-75ebeeaea0908c5527e7b1e305fbc7681382db47 \
    36380402 \
    09378f154554ccdf02a8a0373ff715edb44ddfc3814d2cfdc308f4f49af67ab9)"
  chumpy_source="$(materialize_runtime_source \
    chumpy-580566eafc9ac68b2614b64d6f7aaa84eebb70da.tar.gz \
    chumpy-580566eafc9ac68b2614b64d6f7aaa84eebb70da \
    54967 \
    51253baf90c79a14d69ed4ad911da48e194a6ccad160db80dd7c6cae5e7d14cf)"
  "$CONDA" install --override-channels \
    -c nvidia/label/cuda-11.7.1 -c https://conda.anaconda.org/conda-forge \
    -p "$env" cuda-nvcc=11.7.99 cuda-cudart-dev=11.7.99 \
    cuda-libraries-dev=11.7.1 \
    gcc_linux-64=11 gxx_linux-64=11 cmake ninja -y
  export CUDA_HOME="$env"
  export PATH="$CUDA_HOME/bin:$PATH"
  "$env/bin/python" -I -m pip install setuptools==69.5.1
  "$env/bin/python" -I -m pip install \
    torch==1.13.0+cu117 torchvision==0.14.0+cu117 \
    --extra-index-url https://download.pytorch.org/whl/cu117
  "$env/bin/python" -I -m pip install --no-build-isolation "$pytorch3d_source"
  "$env/bin/python" -I -m pip install --no-build-isolation "$chumpy_source"
  "$env/bin/python" -I -m pip install --no-build-isolation \
    --constraint "$constraints" \
    --extra-index-url https://download.pytorch.org/whl/cu117 \
    -r "$requirements"
  "$env/bin/python" -I -m pip install pytorch-lightning==2.2.4 --no-deps
  "$env/bin/python" -I -m pip install lightning-utilities torchmetrics==1.4.0
  "$env/bin/python" -I -m pip check
  "$env/bin/python" -I - <<'PY'
import cv2
import numpy
import pytorch3d
import pytorch_lightning
import smplx
import torch
import torchvision
assert torch.__version__ == "1.13.0+cu117"
assert torchvision.__version__ == "0.14.0+cu117"
assert numpy.__version__ == "1.26.4"
print("RAW_HAND_ENV_READY", torch.__version__, torch.version.cuda, cv2.__version__)
PY
}

setup_perception() {
  local env="$ROOT/envs/perception"
  local foundation="$ROOT/external_repos/quiethand_m3/FoundationPose"
  local sam="$ROOT/external_repos/quiethand_m3/sam2"
  local pytorch3d_source
  local nvdiffrast_source
  test -d "$foundation"
  test -d "$sam"
  test -x "$CUDA_124/bin/nvcc"
  pytorch3d_source="$(materialize_runtime_source \
    pytorch3d-fdaf9bd6fed7977e4c2056e7c77c640781e58fcd.tar.gz \
    pytorch3d-fdaf9bd6fed7977e4c2056e7c77c640781e58fcd \
    36384373 \
    b81ede673a9e655b381221c2c25abe977260f049933e077e41f606ef095e3ec8)"
  nvdiffrast_source="$(materialize_runtime_source \
    nvdiffrast-253ac4fcea7de5f396371124af597e6cc957bfae.tar.gz \
    nvdiffrast-253ac4fcea7de5f396371124af597e6cc957bfae \
    10718549 \
    351a15c952448fdc24c069eeedb2f7836071ff87ab30792265cd0f43cc5fb0f8)"
  if [[ ! -x "$env/bin/python" ]]; then
    "$CONDA" create --override-channels -c https://conda.anaconda.org/conda-forge \
      -p "$env" python=3.11 pip cmake ninja eigen boost-cpp pybind11 \
      gcc_linux-64=13 gxx_linux-64=13 -y
  fi
  "$CONDA" install --override-channels -c https://conda.anaconda.org/conda-forge \
    -p "$env" gcc_linux-64=13 gxx_linux-64=13 -y
  export CUDA_HOME="$CUDA_124"
  export PATH="$CUDA_HOME/bin:$env/bin:$PATH"
  "$env/bin/python" -I -m pip install \
    torch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1 \
    --index-url https://download.pytorch.org/whl/cu124
  "$env/bin/python" -I -m pip install --no-build-isolation "$pytorch3d_source"
  "$env/bin/python" -I -m pip install --no-build-isolation "$nvdiffrast_source"
  "$env/bin/python" -I -m pip install -r "$foundation/requirements.txt"
  (
    cd "$foundation"
    CONDA_PREFIX="$env" bash build_all_conda.sh
  )
  (
    cd "$sam"
    SAM2_BUILD_ALLOW_ERRORS=0 "$env/bin/python" -I -m pip install \
      --no-build-isolation -v -e .
  )
  "$env/bin/python" -I -m pip check
  "$env/bin/python" -I - <<'PY'
import cv2
import nvdiffrast.torch
import numpy
import pytorch3d
import sam2
import torch
import torchvision
assert torch.__version__ == "2.5.1+cu124"
assert torchvision.__version__ == "0.20.1+cu124"
print("PERCEPTION_ENV_READY", torch.__version__, torch.version.cuda, numpy.__version__, cv2.__version__)
PY
}

case "${1:-}" in
  raw_hand) setup_raw_hand ;;
  perception) setup_perception ;;
  *) echo "usage: $0 raw_hand|perception" >&2; exit 2 ;;
esac
