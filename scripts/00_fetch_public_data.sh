#!/bin/sh
# Rasters released with Sendhilnathan, Ipata & Goldberg 2021 (Nat Commun 12:6475).
# The repository has no licence file, so the data are fetched, not redistributed here.
set -e
cd "$(dirname "$0")/.."
tmp=$(mktemp -d)
git clone --depth 1 https://github.com/naveen-7/Cerebellum_reward.git "$tmp/repo"
mkdir -p data/public
cp "$tmp/repo/Data/CS_DATASET.mat" data/public/
rm -rf "$tmp"
echo "data/public/CS_DATASET.mat"
