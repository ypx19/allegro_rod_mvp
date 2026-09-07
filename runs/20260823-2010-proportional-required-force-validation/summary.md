# Controlled preload validation A

- Method: stationary angular trajectory under proportional axial disturbance.
- Learned policy: none.
- `tau(s)=5*s/400 N m`.
- Full friction vector scale: `4*s/400`.
- Preload multipliers: `0,0.25,0.5,0.75,1,1.5,2,3`.
- Pose SHA-256: `2d8ac7f17a6693855543395d52022524c1a2956422915ee2962544019f6e7c9f`.
- Grasp SHA-256: `fdd9ead60842eca3167f98ecda3c496ea752cff3fa9e053de47a4544e061b03a`.

All trials lose support and fail tracking. The 5 N m reference disturbance is
outside the validated holding regime, so no minimum-force estimate is inferred.
Raw trials remain in `results.json` and `trials.csv`.
