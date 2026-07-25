#!/usr/bin/env bash
set -euo pipefail
mkdir -p external
if [ ! -d external/mujoco_menagerie ]; then
  git clone --depth 1 https://github.com/google-deepmind/mujoco_menagerie.git external/mujoco_menagerie
fi
printf 'Allegro model available at: %s\n' "external/mujoco_menagerie/wonik_allegro"
printf 'The MVP currently uses models/three_finger_rod.xml. The next integration step is to compose right_hand.xml with the rod/task XML.\n'
