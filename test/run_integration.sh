#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd)"

# Fill in before running, or export DEEPCELL_ACCESS_TOKEN in your shell instead.
# export DEEPCELL_ACCESS_TOKEN=""

if [[ -z "${DEEPCELL_ACCESS_TOKEN}" ]]; then
    echo "Error: DEEPCELL_ACCESS_TOKEN is not set. Fill it in at the top of test/run_integration.sh or export it before running." >&2
    exit 1
fi

cd "${REPO_ROOT}"

if [[ ! -f test/test_mibi.tiff || ! -f test/test_ome.tiff || ! -f test/test_plain.tiff ]]; then
    python test/create_test_data.py --output-dir test
fi

tiffs=(
    "test/test_mibi.tiff"
    "test/test_ome.tiff"
    "test/test_plain.tiff"
)

compartments=(
    "whole-cell"
    "nuclear"
)

for tiff in "${tiffs[@]}"; do
    stem="$(basename "${tiff}" .tiff)"
    nuclear_channel="nuclear"
    membrane_channel="membrane"

    if [[ "${stem}" == "test_plain" ]]; then
        nuclear_channel="Channel_0"
        membrane_channel="Channel_1"
    fi

    for compartment in "${compartments[@]}"; do
        output_path="test/${stem}_${compartment}.tiff"
        echo "Running ${stem} (${compartment}) -> ${output_path}"
        mesmer-segment "${tiff}" \
            --nuclear-channel "${nuclear_channel}" \
            --membrane-channel "${membrane_channel}" \
            --compartment "${compartment}" > "${output_path}"
    done
done
