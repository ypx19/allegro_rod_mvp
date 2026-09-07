# Accepted Revolute Mass-Stage Demo Videos

- Artifact directory: `/data/ypx/allegro_rod_mvp/runs/20260823-2037-revolute-full-corrected-stage-demos-seed10000`
- Curriculum state snapshot: `/data/ypx/allegro_rod_mvp/runs/curricula/20260823-2015-proportional-physics-C-seed0/state.json`
- Checkpoint provenance: exact ten accepted corrected-lineage revolute records; the s=400 record points to accepted C and s=200 through s=1 point to their sequential corrected-curriculum checkpoints.
- Protocol: deterministic PPO, fixed seed 10000, 20.0 s / 500 steps, 25 FPS
- Physics: proportional full-vector friction and proportional rod-joint dynamics
- Saved palm pose: `/data/ypx/allegro_rod_mvp/configs/hand_poses/my_grasp.json` (SHA-256 `2d8ac7f17a6693855543395d52022524c1a2956422915ee2962544019f6e7c9f`)
- Recovered grasp: `/data/ypx/allegro_rod_mvp/configs/hand_grasps/my_grasp_revolute_shared.json` (SHA-256 `fdd9ead60842eca3167f98ecda3c496ea752cff3fa9e053de47a4544e061b03a`)
- Success gate: positive net unwrapped angle >= 180 deg, tilt/tip/stability/drop checks; contact is diagnostic.
- Every MP4 was decoded from first through final frame after encoding.

## Checkpoint provenance and results

### s=400
- Video: [revolute_s400.mp4](revolute_s400.mp4)
- Exact path: `/data/ypx/allegro_rod_mvp/runs/20260823-2037-revolute-full-corrected-stage-demos-seed10000/revolute_s400.mp4`
- Accepted run record: `20260823-2015-proportional-physics-C-seed0-R00-s400-mu4-seed0`
- Checkpoint: `/data/ypx/allegro_rod_mvp/runs/20260823-1812-finger-gait-contact-scale010-C-s400-seed0/checkpoints/final_model.zip`
- VecNormalize: `/data/ypx/allegro_rod_mvp/runs/20260823-1812-finger-gait-contact-scale010-C-s400-seed0/checkpoints/vecnormalize.pkl`
- Friction scale/vector: 4 / `[7.2, 7.2, 0.2, 0.004, 0.004]`
- Seed-10000 result: net rotation 563.783 deg; tilt 0.000 deg; tip error 0.000000 mm; success True; final/mean contacts 2/3 / 2.030
- Integrity: 501 decoded frames; 640x480; 25 FPS; 20.04 s

### s=200
- Video: [revolute_s200.mp4](revolute_s200.mp4)
- Exact path: `/data/ypx/allegro_rod_mvp/runs/20260823-2037-revolute-full-corrected-stage-demos-seed10000/revolute_s200.mp4`
- Accepted run record: `20260823-2015-proportional-physics-C-seed0-R01-s200-mu2-seed0`
- Checkpoint: `/data/ypx/allegro_rod_mvp/runs/20260823-2015-proportional-physics-C-seed0-R01-s200-mu2-seed0/checkpoints/final_model.zip`
- VecNormalize: `/data/ypx/allegro_rod_mvp/runs/20260823-2015-proportional-physics-C-seed0-R01-s200-mu2-seed0/checkpoints/vecnormalize.pkl`
- Friction scale/vector: 2 / `[3.6, 3.6, 0.1, 0.002, 0.002]`
- Seed-10000 result: net rotation 780.387 deg; tilt 0.000 deg; tip error 0.000000 mm; success True; final/mean contacts 1/3 / 2.016
- Integrity: 501 decoded frames; 640x480; 25 FPS; 20.04 s

### s=100
- Video: [revolute_s100.mp4](revolute_s100.mp4)
- Exact path: `/data/ypx/allegro_rod_mvp/runs/20260823-2037-revolute-full-corrected-stage-demos-seed10000/revolute_s100.mp4`
- Accepted run record: `20260823-2015-proportional-physics-C-seed0-R02-s100-mu1-seed0`
- Checkpoint: `/data/ypx/allegro_rod_mvp/runs/20260823-2015-proportional-physics-C-seed0-R02-s100-mu1-seed0/checkpoints/final_model.zip`
- VecNormalize: `/data/ypx/allegro_rod_mvp/runs/20260823-2015-proportional-physics-C-seed0-R02-s100-mu1-seed0/checkpoints/vecnormalize.pkl`
- Friction scale/vector: 1 / `[1.8, 1.8, 0.05, 0.001, 0.001]`
- Seed-10000 result: net rotation 1516.543 deg; tilt 0.000 deg; tip error 0.000000 mm; success True; final/mean contacts 1/3 / 1.260
- Integrity: 501 decoded frames; 640x480; 25 FPS; 20.04 s

### s=50
- Video: [revolute_s50.mp4](revolute_s50.mp4)
- Exact path: `/data/ypx/allegro_rod_mvp/runs/20260823-2037-revolute-full-corrected-stage-demos-seed10000/revolute_s50.mp4`
- Accepted run record: `20260823-2015-proportional-physics-C-seed0-R03-s50-mu0.5-seed0`
- Checkpoint: `/data/ypx/allegro_rod_mvp/runs/20260823-2015-proportional-physics-C-seed0-R03-s50-mu0.5-seed0/checkpoints/final_model.zip`
- VecNormalize: `/data/ypx/allegro_rod_mvp/runs/20260823-2015-proportional-physics-C-seed0-R03-s50-mu0.5-seed0/checkpoints/vecnormalize.pkl`
- Friction scale/vector: 0.5 / `[0.9, 0.9, 0.025, 0.0005, 0.0005]`
- Seed-10000 result: net rotation 4455.688 deg; tilt 0.000 deg; tip error 0.000000 mm; success True; final/mean contacts 1/3 / 0.834
- Integrity: 501 decoded frames; 640x480; 25 FPS; 20.04 s

### s=25
- Video: [revolute_s25.mp4](revolute_s25.mp4)
- Exact path: `/data/ypx/allegro_rod_mvp/runs/20260823-2037-revolute-full-corrected-stage-demos-seed10000/revolute_s25.mp4`
- Accepted run record: `20260823-2015-proportional-physics-C-seed0-R04-s25-mu0.25-seed0`
- Checkpoint: `/data/ypx/allegro_rod_mvp/runs/20260823-2015-proportional-physics-C-seed0-R04-s25-mu0.25-seed0/checkpoints/final_model.zip`
- VecNormalize: `/data/ypx/allegro_rod_mvp/runs/20260823-2015-proportional-physics-C-seed0-R04-s25-mu0.25-seed0/checkpoints/vecnormalize.pkl`
- Friction scale/vector: 0.25 / `[0.45, 0.45, 0.0125, 0.00025, 0.00025]`
- Seed-10000 result: net rotation 5194.422 deg; tilt 0.000 deg; tip error 0.000000 mm; success True; final/mean contacts 0/3 / 0.606
- Integrity: 501 decoded frames; 640x480; 25 FPS; 20.04 s

### s=12.5
- Video: [revolute_s12p5.mp4](revolute_s12p5.mp4)
- Exact path: `/data/ypx/allegro_rod_mvp/runs/20260823-2037-revolute-full-corrected-stage-demos-seed10000/revolute_s12p5.mp4`
- Accepted run record: `20260823-2015-proportional-physics-C-seed0-R05-s12.5-mu0.125-seed0`
- Checkpoint: `/data/ypx/allegro_rod_mvp/runs/20260823-2015-proportional-physics-C-seed0-R05-s12.5-mu0.125-seed0/checkpoints/final_model.zip`
- VecNormalize: `/data/ypx/allegro_rod_mvp/runs/20260823-2015-proportional-physics-C-seed0-R05-s12.5-mu0.125-seed0/checkpoints/vecnormalize.pkl`
- Friction scale/vector: 0.125 / `[0.225, 0.225, 0.00625, 0.000125, 0.000125]`
- Seed-10000 result: net rotation 8157.868 deg; tilt 0.000 deg; tip error 0.000000 mm; success True; final/mean contacts 1/3 / 0.520
- Integrity: 501 decoded frames; 640x480; 25 FPS; 20.04 s

### s=6.25
- Video: [revolute_s6p25.mp4](revolute_s6p25.mp4)
- Exact path: `/data/ypx/allegro_rod_mvp/runs/20260823-2037-revolute-full-corrected-stage-demos-seed10000/revolute_s6p25.mp4`
- Accepted run record: `20260823-2015-proportional-physics-C-seed0-R06-s6.25-mu0.0625-seed0`
- Checkpoint: `/data/ypx/allegro_rod_mvp/runs/20260823-2015-proportional-physics-C-seed0-R06-s6.25-mu0.0625-seed0/checkpoints/final_model.zip`
- VecNormalize: `/data/ypx/allegro_rod_mvp/runs/20260823-2015-proportional-physics-C-seed0-R06-s6.25-mu0.0625-seed0/checkpoints/vecnormalize.pkl`
- Friction scale/vector: 0.0625 / `[0.1125, 0.1125, 0.003125, 6.25e-05, 6.25e-05]`
- Seed-10000 result: net rotation 9589.580 deg; tilt 0.000 deg; tip error 0.000000 mm; success True; final/mean contacts 0/3 / 0.528
- Integrity: 501 decoded frames; 640x480; 25 FPS; 20.04 s

### s=3.125
- Video: [revolute_s3p125.mp4](revolute_s3p125.mp4)
- Exact path: `/data/ypx/allegro_rod_mvp/runs/20260823-2037-revolute-full-corrected-stage-demos-seed10000/revolute_s3p125.mp4`
- Accepted run record: `20260823-2015-proportional-physics-C-seed0-R07-s3.125-mu0.03125-seed0`
- Checkpoint: `/data/ypx/allegro_rod_mvp/runs/20260823-2015-proportional-physics-C-seed0-R07-s3.125-mu0.03125-seed0/checkpoints/final_model.zip`
- VecNormalize: `/data/ypx/allegro_rod_mvp/runs/20260823-2015-proportional-physics-C-seed0-R07-s3.125-mu0.03125-seed0/checkpoints/vecnormalize.pkl`
- Friction scale/vector: 0.03125 / `[0.05625, 0.05625, 0.0015625, 3.125e-05, 3.125e-05]`
- Seed-10000 result: net rotation 10225.610 deg; tilt 0.000 deg; tip error 0.000000 mm; success True; final/mean contacts 1/3 / 0.540
- Integrity: 501 decoded frames; 640x480; 25 FPS; 20.04 s

### s=1.5625
- Video: [revolute_s1p5625.mp4](revolute_s1p5625.mp4)
- Exact path: `/data/ypx/allegro_rod_mvp/runs/20260823-2037-revolute-full-corrected-stage-demos-seed10000/revolute_s1p5625.mp4`
- Accepted run record: `20260823-2015-proportional-physics-C-seed0-R08-s1.5625-mu0.015625-seed0`
- Checkpoint: `/data/ypx/allegro_rod_mvp/runs/20260823-2015-proportional-physics-C-seed0-R08-s1.5625-mu0.015625-seed0/checkpoints/final_model.zip`
- VecNormalize: `/data/ypx/allegro_rod_mvp/runs/20260823-2015-proportional-physics-C-seed0-R08-s1.5625-mu0.015625-seed0/checkpoints/vecnormalize.pkl`
- Friction scale/vector: 0.015625 / `[0.028125, 0.028125, 0.00078125, 1.5625e-05, 1.5625e-05]`
- Seed-10000 result: net rotation 11142.942 deg; tilt 0.000 deg; tip error 0.000000 mm; success True; final/mean contacts 1/3 / 0.514
- Integrity: 501 decoded frames; 640x480; 25 FPS; 20.04 s

### s=1
- Video: [revolute_s1.mp4](revolute_s1.mp4)
- Exact path: `/data/ypx/allegro_rod_mvp/runs/20260823-2037-revolute-full-corrected-stage-demos-seed10000/revolute_s1.mp4`
- Accepted run record: `20260823-2015-proportional-physics-C-seed0-R09-s1-mu0.01-seed0`
- Checkpoint: `/data/ypx/allegro_rod_mvp/runs/20260823-2015-proportional-physics-C-seed0-R09-s1-mu0.01-seed0/checkpoints/final_model.zip`
- VecNormalize: `/data/ypx/allegro_rod_mvp/runs/20260823-2015-proportional-physics-C-seed0-R09-s1-mu0.01-seed0/checkpoints/vecnormalize.pkl`
- Friction scale/vector: 0.01 / `[0.018000000000000002, 0.018000000000000002, 0.0005, 1e-05, 1e-05]`
- Seed-10000 result: net rotation 11270.938 deg; tilt 0.000 deg; tip error 0.000000 mm; success True; final/mean contacts 1/3 / 0.484
- Integrity: 501 decoded frames; 640x480; 25 FPS; 20.04 s

