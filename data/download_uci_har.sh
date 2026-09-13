#!/usr/bin/env bash
# Downloads the UCI HAR dataset (Anguita et al., 2013) into data/UCI_HAR_Dataset.
# 30 subjects, 6 activities, 50 Hz accel+gyro, 128-sample windows, subject-grouped split.
set -euo pipefail
cd "$(dirname "$0")"
URL="https://archive.ics.uci.edu/static/public/240/human+activity+recognition+using+smartphones.zip"
if [ -d UCI_HAR_Dataset ]; then echo "already downloaded"; exit 0; fi
curl -L -o har.zip "$URL"
unzip -q -o har.zip
unzip -q -o "UCI HAR Dataset.zip"
mv "UCI HAR Dataset" UCI_HAR_Dataset
rm -rf har.zip "UCI HAR Dataset.zip" "UCI HAR Dataset.names" __MACOSX
echo "done"; ls UCI_HAR_Dataset
