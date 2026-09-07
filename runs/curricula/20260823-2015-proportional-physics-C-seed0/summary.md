# Corrected proportional-physics C curriculum

- Status: Phase R completed; Phase T failed at `s=400`.
- Parent: `runs/20260823-1812-finger-gait-contact-scale010-C-s400-seed0/checkpoints/final_model.zip`
- Gate: fixed and unseen net-angle success each >=0.50.
- Policy force: diagnostic only.
- Friction: full vector, `scale=4*s/400`, no adaptation.
- Rod passive dynamics: damping/armature/frictionloss scaled by `s/400`.

All revolute stages `400,200,100,50,25,12.5,6.25,3.125,1.5625,1` pass
fixed/unseen at 1.0/1.0. Final checkpoint:
`runs/20260823-2015-proportional-physics-C-seed0-R09-s1-mu0.01-seed0/checkpoints/final_model.zip`.

Final revolute fixed/unseen rotation is 11,384.68°/11,337.89°, but >=2-contact
fraction is only 0.0146/0.0180. This is accepted by the declared angle gate, not
claimed to be supported gaiting.

Heavy tip-connect fails after the initial transfer, one standard extension, and
one predeclared low-LR retry. The final retry has 0.0/0.0 success and all 20
episodes terminate on axis tilt. Tip-connect lower masses were not started.

See `reports/comparisons/20260823-proportional-physics-C-curriculum.md`.
