# s25 Low-Friction Bound Diagnostic
## Scope
This is a new diagnostic after the preserved
`20260823-1855-net-angle-C-force-curriculum-seed0` saturation result. Two and
only two additional scales, 0.05 and 0.025, were independently trained for 100k
steps from the accepted `s=50` checkpoint.

## Friction Vectors
The rod and each of `tip0`, `tip1`, and `tip2` use identical vectors:
- scale 0.10: `[0.18,0.05,0.001]`;
- scale 0.05: `[0.09,0.05,0.001]`;
- scale 0.025: `[0.045,0.05,0.001]`.

Only sliding friction changes. Torsional and rolling coefficients remain 0.05
and 0.001. All values are finite, strictly positive, and accepted by MuJoCo.

## Protocol
Revolute `s=25`; C reward and contact settings; net-angle success; identical
training seed, fixed seeds 10000–10009, unseen seeds 20000–20009, optimizer,
normalization parent, and 100k budget. The force gate is fixed median total
normal force conditioned on `omega >0.5 rad/s` and at least two contacts, with
target band `[78.447,117.670] N`.

## Scale 0.05
- Fixed/unseen success: 1.0/1.0.
- Net angle: 3026.01°/2802.79°.
- Conditioned force median: 54.669/54.972 N.
- Conditioned force p95: 100.573/105.380 N.
- Eligible fraction: 0.1760/0.1698.
- All-step force p95: 74.628/78.996 N.
- Contact distribution fixed: 0/1/2/3 =
  `0.1786/0.5614/0.2368/0.0232`.
- Contact distribution unseen: `0.1790/0.5552/0.2390/0.0268`.
- >=1 and >=2 fractions: 0.8214/0.2600 fixed,
  0.8210/0.2658 unseen.
- Axial-slip proxy p95: `1.886e-15/1.850e-15 m/s`.
- Maximum tip error below `6.6e-17 m`; numerical instability 0/0.

## Scale 0.025
- Fixed/unseen success: 1.0/1.0.
- Net angle: 3073.89°/3191.96°.
- Conditioned force median: 49.427/47.272 N.
- Conditioned force p95: 95.461/80.244 N.
- Eligible fraction: 0.0996/0.1050.
- All-step force p95: 62.193/61.157 N.
- Contact distribution fixed: `0.2710/0.5724/0.1432/0.0134`.
- Contact distribution unseen: `0.2820/0.5660/0.1404/0.0116`.
- >=1 and >=2 fractions: 0.7290/0.1566 fixed,
  0.7180/0.1520 unseen.
- Axial-slip proxy p95: `1.925e-15/1.928e-15 m/s`.
- Maximum tip error below `5.1e-17 m`; numerical instability 0/0.

## Monotonicity
Plot: `reports/comparisons/20260823-s25-low-friction-bound-extension.png`.

Fixed conditioned force changes from 53.386 N at scale 0.10 to 54.669 N at
0.05, then 49.427 N at 0.025. The first decrease in friction produces a small
force increase, but the second produces a clear decrease. Therefore force is not
monotone in the physically expected direction. Support also degrades: >=2-contact
fraction falls from 0.260 at 0.05 to 0.157 at 0.025.

## Conclusion
Neither predeclared scale reaches the force band despite 1.0/1.0 net-angle
success. Friction-only adaptation is insufficient at `s=25` under this protocol.
No stage is accepted, so no acceptance video is required. Lower masses and
tip-connect remain blocked.
