# Controlled preload validation B

- Method: stationary angular trajectory under proportional axial disturbance.
- Learned policy: none.
- `tau(s)=0.5*s/400 N m`.
- Full friction vector scale: `4*s/400`.
- Preload multipliers: `0,0.25,0.5,0.75,1,1.5,2,3`.
- Pose SHA-256: `2d8ac7f17a6693855543395d52022524c1a2956422915ee2962544019f6e7c9f`.
- Grasp SHA-256: `fdd9ead60842eca3167f98ecda3c496ea752cff3fa9e053de47a4544e061b03a`.

Analytical `N_min` is 1.838 N at every mass. No multiplier through 3 passes at
`s=400` or `s=25`; upper-bound median forces are 24.144 and 19.561 N. `s=1`
first passes at multiplier 1.5 with 9.314 N median force. Therefore exact
simulator-level invariance is not established. Raw results are in `results.json`
and `trials.csv`.
