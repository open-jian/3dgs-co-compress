#!/usr/bin/env bash

EVAL_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(dirname "${EVAL_DIR}")"
CODE_ROOT="$(dirname "${REPO_ROOT}")"
WORKSPACE_ROOT="$(dirname "${CODE_ROOT}")"
OUTPUT_ROOT="${OUTPUT_ROOT:-${WORKSPACE_ROOT}/Output}"
DATA_ROOT="${DATA_ROOT:-${WORKSPACE_ROOT}/Data}"
COLASPLAT_OUTPUT="${OUTPUT_ROOT}/colasplat"
AE_CKPT_ROOT="${COLASPLAT_OUTPUT}/autoencoder/ckpt"
export OUTPUT_ROOT DATA_ROOT
