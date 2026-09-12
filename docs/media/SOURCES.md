# Demo clip sources

| File | Run / path | Role |
|---|---|---|
| stage-d0-top-revolute.mp4 | `20260808-0124-…-C0-revolute` final | Top-hang D0 revolute @s=400, μ≈7.2 |
| stage-d1-top-tip.mp4 | `20260808-0124-…-C1-tip` final | Top-hang D1 first tip-connect transfer |
| stage-d2-top-tip.mp4 | `20260808-0124-…-C2-retry1-tip` final | Tip retry; rotation ↑ |
| stage-d3-top-tip.mp4 | `20260808-0124-…-C3-retry2-tip` final | Tip retry; near-final gait |
| stage-d4-top-tip.mp4 | `20260808-0124-…-C4-retry3-tip` final | Tip-connect two-finger gait (~5000°) |
| success-top-tip-c4-twisting.mp4 | same as D4 | Hero / gallery alias of D4 |
| success-revolute-a1-omega-hold.mp4 | `20260802-1439-exp-a1-revolute` ckpt_800k | Revolute hinge + sustained ω hold |
| success-stage0-hanging-tip.mp4 | `20260802-0220` ckpt_5e6 | Stage 0 hanging tip connect |
| success-bottom-revolute-3touch.mp4 | `20260808-0224-…-C0-revolute` | Bottom tip revolute with 3-touch reward |
| partial-bottom-tip-c4-stand.mp4 | `20260808-0232-…-C4` | Bottom tip stands 20s, slow spin (~56°) — gate fail |
| fail-tip-connect-tilt-cliff.mp4 | `20260802-1555-exp-b2` | Tip-connect tilt collapse after revolute transfer |
| fail-light-mass-s40-tilt.mp4 | `20260808-0115-…-C1` s=40 | Light-mass tip: spins then axis_tilt kill |
| fail-bottom-contact-support.mp4 | `20260808-0232-…-C2` | Lost 3-finger contact window |
| fail-free-tip-c5-collapse.mp4 | `20260808-0232-…-C5` | Free tip transfer dies in ~6 steps |
| fail-stabilizer-assist-cliff.mp4 | `20260723-1600` Stage2 final | External axis-stabilizer removal cliff |
| ablation/without_curriculum_seed1.mp4 | `20260817-2332-ablation-fixed-s1` | No-curriculum eval @ s=1 |
| ablation/with_curriculum_seed1.mp4 | `20260817-2332-ablation-curr-s400to1` | Curriculum eval @ s=1 |
| ablation/side_by_side_seed0.mp4 | ablation comparison | Seed 0 side-by-side |
| ablation/side_by_side_seed1.mp4 | ablation comparison | Seed 1 side-by-side |
| ablation/side_by_side_seed2.mp4 | ablation comparison | Seed 2 side-by-side |

| cinematic-allegro-tip-bottom.mp4 | T00 `20260823-2040-…-T00-s400-mu4-lr3e5` seed 6; flipped `my_grasp.json`; policy from frame 0; 30-frame non-tilt prefix looped 10×; 仰视 orbit `elev 10±4°` | Hero / homepage left clip |
| cinematic-allegro-tip-bottom.jpg | frame ~2.4 s of the 仰视 loop | Hero / gallery poster |
| cinematic-allegro-tip-bottom-unflipped-c3.mp4 | archived C3 `20260823-0405-…-C3-mass-1` seed 0; default unflipped palm | Unused hero (palm on floor) |
| cinematic-allegro-tip-bottom-unflipped-c3.jpg | hold frame of the unflipped C3 fly-in | Unused poster |
| cinematic-allegro-tip-c3.mp4 | same unflipped C3 policy, close side orbit (superseded) | Unused hero candidate |
| cinematic-allegro-tip-c3.jpg | frame ~7.2 s of the orbit clip | Unused poster |
| partial-allegro-t00-flipped.mp4 | T00 eval `tip_connect_best_00_seed10000` (34 steps, axis_tilt) | Overlay of the flipped T00 episode |
| partial-allegro-tip-c3-stand.mp4 | `stage_videos_20260823-130800/07_C3-mass-1_seed0_final.mp4` | Overlay eval of unflipped C3 |
| hand_surrogate_45h.png | `reports/decks/assets/hand_surrogate_45h.png` | v1 9-DoF planar surrogate, 45° |
| hand_allegro_45h.png | `reports/decks/assets/hand_allegro_45h.png` | v2 12-DoF Allegro-like wrap, 45° |
| cinematic-screwdriver.mp4 | `20260818-2345-cinematic-c0-screwdriver-seed0` | Retired hero / homepage front clip (9-DoF revolute) |
| cinematic-screwdriver.jpg | first frame of `cinematic-screwdriver.mp4` | Hero / gallery poster |
| cinematic-screwdriver-oval.mp4 | `20260818-2350-cinematic-c0-screwdriver-oval-seed0` | Hero / homepage front clip |
| cinematic-isaacgym.mp4 | `20260818-2325-cinematic-viz-seed0` | Hero / homepage right clip: Isaac Gym mosaic, n=16 |
| cinematic-isaacgym.jpg | frame 1/120 of `cinematic-isaacgym.mp4` | Hero / gallery poster (4×4 mosaic) |
| og-isaacgym-parallel.jpg | 2×2 of close third-person crops from `cinematic-isaacgym.mp4` hold (not top-down mosaic) | Open Graph / Twitter share image |
| front-demo.mp4 | hstack of screwdriver + isaacgym (1920×540) | README / GitHub repo front demo |
| front-demo.webp | animated still of `front-demo.mp4` (1280×360) | README-safe looping preview (GitHub strips `<video>`) |

Ablation clips live under `docs/media/ablation/` for GitHub Pages.
