# Curriculum Stage Videos

- Source state: `/data/ypx/allegro_rod_mvp/runs/curricula/20260823-0405-allegro-tip-bottom-smoke-seed0/state.json`
- Policy: deterministic PPO
- Preferred evaluation seed: 0
- Episode limit: 20 seconds
- Each video uses the selected checkpoint and selected VecNormalize file recorded in state.json.

## A0-revolute
- Video: `/data/ypx/allegro_rod_mvp/runs/curricula/20260823-0405-allegro-tip-bottom-smoke-seed0/stage_videos_20260823-130800/00_A0-revolute_seed0_ppo_rod_20000_steps.mp4`
- Checkpoint: `/data/ypx/allegro_rod_mvp/runs/20260823-0405-allegro-tip-bottom-smoke-seed0-A0-revolute-s10-retry0/checkpoints/ppo_rod_20000_steps.zip`
- VecNormalize: `/data/ypx/allegro_rod_mvp/runs/20260823-0405-allegro-tip-bottom-smoke-seed0-A0-revolute-s10-retry0/checkpoints/ppo_rod_20000_steps_vecnormalize.pkl`
- Seed: 0 (preferred seed completed several seconds)
- Final: rotation 0.12°, tip error 0.00 mm, contacts 2/3 (mean 2.974), tilt 0.00°, success False, termination `contact_support`, duration 13.76 s

## B0-soft-connect
- Video: `/data/ypx/allegro_rod_mvp/runs/curricula/20260823-0405-allegro-tip-bottom-smoke-seed0/stage_videos_20260823-130800/01_B0-soft-connect_seed0_ppo_rod_10000_steps.mp4`
- Checkpoint: `/data/ypx/allegro_rod_mvp/runs/20260823-0405-allegro-tip-bottom-smoke-seed0-B0-soft-connect-s10-retry0/checkpoints/ppo_rod_10000_steps.zip`
- VecNormalize: `/data/ypx/allegro_rod_mvp/runs/20260823-0405-allegro-tip-bottom-smoke-seed0-B0-soft-connect-s10-retry0/checkpoints/ppo_rod_10000_steps_vecnormalize.pkl`
- Seed: 0 (all checked seeds failed similarly; retained representative preferred seed)
- Final: rotation 51.03°, tip error 7.92 mm, contacts 2/3 (mean 2.884), tilt 8.64°, success False, termination `contact_support`, duration 2.76 s

## B1-medium-connect
- Video: `/data/ypx/allegro_rod_mvp/runs/curricula/20260823-0405-allegro-tip-bottom-smoke-seed0/stage_videos_20260823-130800/02_B1-medium-connect_seed0_final.mp4`
- Checkpoint: `/data/ypx/allegro_rod_mvp/runs/20260823-0405-allegro-tip-bottom-smoke-seed0-B1-medium-connect-s10-retry0/checkpoints/final_model.zip`
- VecNormalize: `/data/ypx/allegro_rod_mvp/runs/20260823-0405-allegro-tip-bottom-smoke-seed0-B1-medium-connect-s10-retry0/checkpoints/vecnormalize.pkl`
- Seed: 0 (preferred seed completed several seconds)
- Final: rotation 48.97°, tip error 1.63 mm, contacts 2/3 (mean 2.931), tilt 8.79°, success False, termination `contact_support`, duration 4.64 s

## B2-hard-connect
- Video: `/data/ypx/allegro_rod_mvp/runs/curricula/20260823-0405-allegro-tip-bottom-smoke-seed0/stage_videos_20260823-130800/03_B2-hard-connect_seed0_final.mp4`
- Checkpoint: `/data/ypx/allegro_rod_mvp/runs/20260823-0405-allegro-tip-bottom-smoke-seed0-B2-hard-connect-s10-retry0/checkpoints/final_model.zip`
- VecNormalize: `/data/ypx/allegro_rod_mvp/runs/20260823-0405-allegro-tip-bottom-smoke-seed0-B2-hard-connect-s10-retry0/checkpoints/vecnormalize.pkl`
- Seed: 0 (preferred seed completed several seconds)
- Final: rotation 71.62°, tip error 0.65 mm, contacts 2/3 (mean 2.946), tilt 14.22°, success False, termination `contact_support`, duration 5.92 s

## B3-zero-stabilizer
- Video: `/data/ypx/allegro_rod_mvp/runs/curricula/20260823-0405-allegro-tip-bottom-smoke-seed0/stage_videos_20260823-130800/04_B3-zero-stabilizer_seed0_ppo_rod_10000_steps.mp4`
- Checkpoint: `/data/ypx/allegro_rod_mvp/runs/20260823-0405-allegro-tip-bottom-smoke-seed0-B3-zero-stabilizer-s10-retry0/checkpoints/ppo_rod_10000_steps.zip`
- VecNormalize: `/data/ypx/allegro_rod_mvp/runs/20260823-0405-allegro-tip-bottom-smoke-seed0-B3-zero-stabilizer-s10-retry0/checkpoints/ppo_rod_10000_steps_vecnormalize.pkl`
- Seed: 0 (preferred seed completed several seconds)
- Final: rotation 67.16°, tip error 0.53 mm, contacts 2/3 (mean 2.944), tilt 11.97°, success False, termination `contact_support`, duration 5.76 s

## C1-mass-4
- Video: `/data/ypx/allegro_rod_mvp/runs/curricula/20260823-0405-allegro-tip-bottom-smoke-seed0/stage_videos_20260823-130800/05_C1-mass-4_seed7_ppo_rod_10000_steps.mp4`
- Checkpoint: `/data/ypx/allegro_rod_mvp/runs/20260823-0405-allegro-tip-bottom-smoke-seed0-C1-mass-4-s4-retry0/checkpoints/ppo_rod_10000_steps.zip`
- VecNormalize: `/data/ypx/allegro_rod_mvp/runs/20260823-0405-allegro-tip-bottom-smoke-seed0-C1-mass-4-s4-retry0/checkpoints/ppo_rod_10000_steps_vecnormalize.pkl`
- Seed: 7 (preferred seed failed immediately; selected longest fixed alternate)
- Final: rotation 66.99°, tip error 0.62 mm, contacts 2/3 (mean 2.935), tilt 11.34°, success False, termination `contact_support`, duration 4.92 s

## C2-mass-2
- Video: `/data/ypx/allegro_rod_mvp/runs/curricula/20260823-0405-allegro-tip-bottom-smoke-seed0/stage_videos_20260823-130800/06_C2-mass-2_seed0_ppo_rod_10000_steps.mp4`
- Checkpoint: `/data/ypx/allegro_rod_mvp/runs/20260823-0405-allegro-tip-bottom-smoke-seed0-C2-mass-2-s2-retry0/checkpoints/ppo_rod_10000_steps.zip`
- VecNormalize: `/data/ypx/allegro_rod_mvp/runs/20260823-0405-allegro-tip-bottom-smoke-seed0-C2-mass-2-s2-retry0/checkpoints/ppo_rod_10000_steps_vecnormalize.pkl`
- Seed: 0 (preferred seed completed several seconds)
- Final: rotation 53.30°, tip error 1.24 mm, contacts 3/3 (mean 2.986), tilt 11.30°, success False, termination `none`, duration 20.00 s

## C3-mass-1
- Video: `/data/ypx/allegro_rod_mvp/runs/curricula/20260823-0405-allegro-tip-bottom-smoke-seed0/stage_videos_20260823-130800/07_C3-mass-1_seed0_final.mp4`
- Checkpoint: `/data/ypx/allegro_rod_mvp/runs/20260823-0405-allegro-tip-bottom-smoke-seed0-C3-mass-1-s1-retry0/checkpoints/final_model.zip`
- VecNormalize: `/data/ypx/allegro_rod_mvp/runs/20260823-0405-allegro-tip-bottom-smoke-seed0-C3-mass-1-s1-retry0/checkpoints/vecnormalize.pkl`
- Seed: 0 (preferred seed completed several seconds)
- Final: rotation 39.01°, tip error 1.25 mm, contacts 3/3 (mean 3.000), tilt 7.66°, success False, termination `none`, duration 20.00 s

