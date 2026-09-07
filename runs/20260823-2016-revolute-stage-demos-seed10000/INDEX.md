# Accepted Revolute Mass-Stage Demo Videos

- Artifact directory: `/data/ypx/allegro_rod_mvp/runs/20260823-2016-revolute-stage-demos-seed10000`
- Curriculum state snapshot: `/data/ypx/allegro_rod_mvp/runs/curricula/20260823-2015-proportional-physics-C-seed0/state.json`
- Protocol: deterministic PPO, fixed seed 10000, 20.0 s / 500 steps, 25 FPS
- Physics: proportional full-vector friction and proportional rod-joint dynamics
- Saved palm pose: `/data/ypx/allegro_rod_mvp/configs/hand_poses/my_grasp.json` (SHA-256 `2d8ac7f17a6693855543395d52022524c1a2956422915ee2962544019f6e7c9f`)
- Recovered grasp: `/data/ypx/allegro_rod_mvp/configs/hand_grasps/my_grasp_revolute_shared.json` (SHA-256 `fdd9ead60842eca3167f98ecda3c496ea752cff3fa9e053de47a4544e061b03a`)
- Success gate: positive net unwrapped angle >= 180 deg, tilt/tip/stability/drop checks; contact is diagnostic.
- Every MP4 was decoded from first through final frame after encoding.

## Checkpoint provenance and results

### s=400
- Video: [revolute_s400.mp4](revolute_s400.mp4)
- Exact path: `/data/ypx/allegro_rod_mvp/runs/20260823-2016-revolute-stage-demos-seed10000/revolute_s400.mp4`
- Accepted run record: `20260823-2015-proportional-physics-C-seed0-R00-s400-mu4-seed0`
- Checkpoint: `/data/ypx/allegro_rod_mvp/runs/20260823-1812-finger-gait-contact-scale010-C-s400-seed0/checkpoints/final_model.zip`
- VecNormalize: `/data/ypx/allegro_rod_mvp/runs/20260823-1812-finger-gait-contact-scale010-C-s400-seed0/checkpoints/vecnormalize.pkl`
- Friction scale/vector: 4 / `[7.2, 7.2, 0.2, 0.004, 0.004]`
- Seed-10000 result: net rotation 563.783 deg; tilt 0.000 deg; tip error 0.000000 mm; success True; final/mean contacts 2/3 / 2.030
- Integrity: 501 decoded frames; 640x480; 25 FPS; 20.04 s

### s=200
- Video: [revolute_s200.mp4](revolute_s200.mp4)
- Exact path: `/data/ypx/allegro_rod_mvp/runs/20260823-2016-revolute-stage-demos-seed10000/revolute_s200.mp4`
- Accepted run record: `20260823-2015-proportional-physics-C-seed0-R01-s200-mu2-seed0`
- Checkpoint: `/data/ypx/allegro_rod_mvp/runs/20260823-2015-proportional-physics-C-seed0-R01-s200-mu2-seed0/checkpoints/final_model.zip`
- VecNormalize: `/data/ypx/allegro_rod_mvp/runs/20260823-2015-proportional-physics-C-seed0-R01-s200-mu2-seed0/checkpoints/vecnormalize.pkl`
- Friction scale/vector: 2 / `[3.6, 3.6, 0.1, 0.002, 0.002]`
- Seed-10000 result: net rotation 780.387 deg; tilt 0.000 deg; tip error 0.000000 mm; success True; final/mean contacts 1/3 / 2.016
- Integrity: 501 decoded frames; 640x480; 25 FPS; 20.04 s

### s=100
- Video: [revolute_s100.mp4](revolute_s100.mp4)
- Exact path: `/data/ypx/allegro_rod_mvp/runs/20260823-2016-revolute-stage-demos-seed10000/revolute_s100.mp4`
- Accepted run record: `20260823-2015-proportional-physics-C-seed0-R02-s100-mu1-seed0`
- Checkpoint: `/data/ypx/allegro_rod_mvp/runs/20260823-2015-proportional-physics-C-seed0-R02-s100-mu1-seed0/checkpoints/final_model.zip`
- VecNormalize: `/data/ypx/allegro_rod_mvp/runs/20260823-2015-proportional-physics-C-seed0-R02-s100-mu1-seed0/checkpoints/vecnormalize.pkl`
- Friction scale/vector: 1 / `[1.8, 1.8, 0.05, 0.001, 0.001]`
- Seed-10000 result: net rotation 1516.543 deg; tilt 0.000 deg; tip error 0.000000 mm; success True; final/mean contacts 1/3 / 1.260
- Integrity: 501 decoded frames; 640x480; 25 FPS; 20.04 s

### s=50
- Video: [revolute_s50.mp4](revolute_s50.mp4)
- Exact path: `/data/ypx/allegro_rod_mvp/runs/20260823-2016-revolute-stage-demos-seed10000/revolute_s50.mp4`
- Accepted run record: `20260823-2015-proportional-physics-C-seed0-R03-s50-mu0.5-seed0`
- Checkpoint: `/data/ypx/allegro_rod_mvp/runs/20260823-2015-proportional-physics-C-seed0-R03-s50-mu0.5-seed0/checkpoints/final_model.zip`
- VecNormalize: `/data/ypx/allegro_rod_mvp/runs/20260823-2015-proportional-physics-C-seed0-R03-s50-mu0.5-seed0/checkpoints/vecnormalize.pkl`
- Friction scale/vector: 0.5 / `[0.9, 0.9, 0.025, 0.0005, 0.0005]`
- Seed-10000 result: net rotation 4455.688 deg; tilt 0.000 deg; tip error 0.000000 mm; success True; final/mean contacts 1/3 / 0.834
- Integrity: 501 decoded frames; 640x480; 25 FPS; 20.04 s

### s=25
- Video: [revolute_s25.mp4](revolute_s25.mp4)
- Exact path: `/data/ypx/allegro_rod_mvp/runs/20260823-2016-revolute-stage-demos-seed10000/revolute_s25.mp4`
- Accepted run record: `20260823-2015-proportional-physics-C-seed0-R04-s25-mu0.25-seed0`
- Checkpoint: `/data/ypx/allegro_rod_mvp/runs/20260823-2015-proportional-physics-C-seed0-R04-s25-mu0.25-seed0/checkpoints/final_model.zip`
- VecNormalize: `/data/ypx/allegro_rod_mvp/runs/20260823-2015-proportional-physics-C-seed0-R04-s25-mu0.25-seed0/checkpoints/vecnormalize.pkl`
- Friction scale/vector: 0.25 / `[0.45, 0.45, 0.0125, 0.00025, 0.00025]`
- Seed-10000 result: net rotation 5194.422 deg; tilt 0.000 deg; tip error 0.000000 mm; success True; final/mean contacts 0/3 / 0.606
- Integrity: 501 decoded frames; 640x480; 25 FPS; 20.04 s

