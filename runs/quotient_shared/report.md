
### main

| variant | bits | solved | estimator phase: exact / given the true digits | after the switch: exact / given / hard | steps after the switch to exact (median) | subtractor rows right (of 200) |
|---|---|---|---|---|---|---|
| shared subtractor | 6 | 3/4 | 0.05–0.86 / 0.05–0.99 | 0.07–1.00 / 0.06–1.00 / 0.04–1.00 | 200 (3 of 4 reached it) | 0.045–0.810 |
| shared subtractor | 7 | 1/4 | 0.02–0.85 / 0.01–0.98 | 0.02–0.99 / 0.02–1.00 / 0.02–0.99 | 1700 (1 of 4 reached it) | 0.060–0.735 |
| shared subtractor | 10 | 2/4 | 0.00–0.78 / 0.00–1.00 | 0.00–1.00 / 0.00–1.00 / 0.00–1.00 | 200 (2 of 4 reached it) | 0.090–1.000 |
| learned slots (longdiv's cell) | 6 | 0/3 | 0.42–0.48 / 0.02–0.03 | 0.45–0.71 / 0.02 / 0.32–0.68 | · (0 of 3 reached it) | · |
| learned slots (longdiv's cell) | 7 | 0/3 | 0.25–0.41 / 0.01–0.06 | 0.05–0.06 / 0.02–0.04 / 0.03–0.05 | · (0 of 3 reached it) | · |
| learned slots (longdiv's cell) | 10 | 0/3 | 0.09–0.17 / 0.00 | 0.00–0.01 / 0.00 / 0.00 | · (0 of 3 reached it) | · |
| shared subtractor, always from the top w digits | 6 | 0/2 | 0.04–0.32 / 0.07–0.67 | 0.07–0.27 / 0.16–0.54 / 0.04–0.05 | · (0 of 2 reached it) | 0.050–0.215 |
| shared subtractor, always from the top w digits | 7 | 0/2 | 0.03–0.67 / 0.01–0.96 | 0.03–0.06 / 0.02–0.65 / 0.02 | · (0 of 2 reached it) | 0.095–0.445 |
| shared subtractor, always from the top w digits | 10 | 0/2 | 0.00 / 0.00 | 0.00 / 0.01 / 0.00 | · (0 of 2 reached it) | 0.080 |
| shared subtractor, choice held uniform for 400 steps | 6 | 4/4 | 0.92–0.97 / 1.00 | 1.00 / 1.00 / 1.00 | 200 (4 of 4 reached it) | 0.805–0.810 |
| shared subtractor, choice held uniform for 400 steps | 7 | 2/8 | 0.82–0.85 / 0.98–1.00 | 0.50–0.99 / 0.72–1.00 / 0.13–0.99 | · (0 of 8 reached it) | 0.410–0.975 |
| shared subtractor, choice held uniform for 400 steps | 10 | 4/4 | 0.73–0.76 / 1.00 | 1.00 / 1.00 / 1.00 | 100 (4 of 4 reached it) | 1.000 |
| shared subtractor, choice held uniform for 400 steps, borrow a bit between places | 6 | 4/4 | 0.88–0.97 / 0.94–1.00 | 1.00 / 1.00 / 1.00 | 200 (4 of 4 reached it) | 0.805–0.810 |
| shared subtractor, choice held uniform for 400 steps, borrow a bit between places | 7 | 1/4 | 0.79–0.86 / 0.88–1.00 | 0.03–1.00 / 0.28–1.00 / 0.00–1.00 | 100 (1 of 4 reached it) | 0.400–1.000 |
| shared subtractor, choice held uniform for 400 steps, borrow a bit between places | 10 | 0/4 | 0.65–0.72 / 0.89–1.00 | 0.00 / 0.09–0.18 / 0.00 | · (0 of 4 reached it) | 0.395–0.505 |
| shared subtractor, choice held uniform for 400 steps, handover over 500 steps | 6 | 3/4 | 0.92–0.96 / 0.99–1.00 | 0.97–1.00 / 0.99–1.00 / 0.97–1.00 | 500 (3 of 4 reached it) | 0.795–0.810 |
| shared subtractor, choice held uniform for 400 steps, handover over 500 steps | 7 | 0/4 | 0.81–0.85 / 0.96–1.00 | 0.40–0.99 / 0.72–1.00 / 0.13–0.99 | 1700 (1 of 4 reached it) | 0.390–0.705 |
| shared subtractor, choice held uniform for 400 steps, handover over 500 steps | 10 | 4/4 | 0.73–0.75 / 1.00 | 1.00 / 1.00 / 1.00 | 500 (4 of 4 reached it) | 1.000 |
| shared subtractor, choice held uniform for 400 steps, lr 0.0003 after the switch | 7 | 0/4 | 0.82–0.85 / 0.98–1.00 | 0.05–0.95 / 0.91–1.00 / 0.05–0.96 | · (0 of 4 reached it) | 0.300–0.645 |
| shared subtractor, choice held uniform for 400 steps, lr 0.001 after the switch | 7 | 0/4 | 0.82–0.84 / 0.98 | 0.32–0.98 / 0.70–1.00 / 0.18–0.98 | · (0 of 4 reached it) | 0.490–0.625 |
| shared subtractor, choice held uniform for 400 steps, phase B only, from S4 weights, phase B hard_borrow=all, borrow a bit between places | 6 | 0/4 | 0.88–0.97 / 0.94–1.00 | 0.02–0.04 / 0.05–0.17 / 0.03–0.04 | · (0 of 4 reached it) | 0.040–0.160 |
| shared subtractor, choice held uniform for 400 steps, phase B only, from S4 weights, phase B hard_borrow=all, borrow a bit between places | 7 | 2/4 | 0.79–0.86 / 0.88–1.00 | 0.01–1.00 / 0.14–1.00 / 0.01–1.00 | 100 (2 of 4 reached it) | 0.430–1.000 |
| shared subtractor, choice held uniform for 400 steps, phase B only, from S4 weights, phase B hard_borrow=all, borrow a bit between places | 10 | 1/4 | 0.65–0.72 / 0.89–1.00 | 0.00–1.00 / 0.00–1.00 / 0.00–1.00 | 0 (1 of 4 reached it) | 0.235–1.000 |
| shared subtractor, not staged | 6 | 0/2 | 0.04–0.21 / 0.06 | 0.10–0.16 / 0.20–0.21 / 0.05–0.06 | · (0 of 2 reached it) | 0.095–0.160 |
| shared subtractor, not staged | 7 | 0/2 | 0.03–0.04 / 0.03 | 0.03 / 0.03–0.08 / 0.02–0.03 | · (0 of 2 reached it) | 0.070–0.080 |
| shared subtractor, not staged | 10 | 0/2 | 0.00–0.01 / 0.00 | 0.00 / 0.01 / 0.00 | · (0 of 2 reached it) | 0.065–0.085 |
| shared subtractor, own borrow from the start | 6 | 2/2 | 1.00 / 1.00 | 1.00 / 1.00 / 1.00 | 0 (2 of 2 reached it) | 0.775–0.810 |
| shared subtractor, own borrow from the start | 7 | 0/2 | 0.04–0.98 / 0.17–1.00 | 0.94–0.99 / 0.98–1.00 / 0.94–0.99 | 100 (1 of 2 reached it) | 0.510–0.515 |
| shared subtractor, own borrow from the start | 10 | 0/2 | 0.00 / 0.01–0.14 | 0.00–0.01 / 0.00–0.16 / 0.00–0.01 | · (0 of 2 reached it) | 0.080–0.280 |
| shared subtractor, window 2/1 | 6 | 3/3 | 0.68–0.78 / 0.95–0.98 | 1.00 / 1.00 / 1.00 | 200 (3 of 3 reached it) | 0.805–0.810 |
| shared subtractor, window 2/1 | 7 | 0/3 | 0.74–0.75 / 0.87–0.88 | 0.53–0.98 / 0.79–0.99 / 0.06–0.98 | · (0 of 3 reached it) | 0.290–0.545 |
| shared subtractor, window 2/1 | 10 | 0/3 | 0.00–0.62 / 0.01–0.89 | 0.00–0.01 / 0.01–0.68 / 0.00–0.01 | · (0 of 3 reached it) | 0.100–0.605 |

### held-out moduli

| variant | bits | solved (all moduli) | estimator phase, held-out moduli (trained) | after the switch, held-out moduli (trained) | given the true digits, held-out moduli |
|---|---|---|---|---|---|
| shared subtractor | 6 | 2/4 | 0.03–0.44 (·) | 0.02–1.00 (0.11–1.00) | 0.02–1.00 |
| shared subtractor | 7 | 1/4 | 0.03–0.78 (0.02–0.84) | 0.03–1.00 (0.03–1.00) | 0.03–1.00 |
| shared subtractor | 10 | 1/4 | 0.01–0.75 (0.00–0.77) | 0.00–1.00 (0.00–1.00) | 0.00–1.00 |
| shared subtractor, choice held uniform for 400 steps | 6 | 4/4 | 0.50–0.65 (·) | 1.00 (1.00) | 1.00 |
| shared subtractor, choice held uniform for 400 steps | 7 | 0/4 | 0.73–0.78 (0.80–0.83) | 0.02–0.99 (0.01–0.98) | 0.10–1.00 |
| shared subtractor, choice held uniform for 400 steps | 10 | 4/4 | 0.72–0.76 (0.74–0.75) | 1.00 (1.00) | 1.00 |
| shared subtractor, choice held uniform for 400 steps, borrow a bit between places | 6 | 3/4 | 0.43–0.59 (·) | 0.02–1.00 (0.20–1.00) | 0.59–1.00 |
| shared subtractor, choice held uniform for 400 steps, borrow a bit between places | 7 | 1/4 | 0.68–0.74 (0.80–0.88) | 0.02–1.00 (0.05–1.00) | 0.37–1.00 |
| shared subtractor, choice held uniform for 400 steps, borrow a bit between places | 10 | 0/4 | 0.60–0.70 (0.59–0.72) | 0.00–0.01 (0.00) | 0.01–0.11 |
| shared subtractor, choice held uniform for 400 steps, handover over 500 steps | 6 | 4/4 | 0.50–0.68 (·) | 1.00 (1.00) | 1.00 |
| shared subtractor, choice held uniform for 400 steps, handover over 500 steps | 7 | 0/4 | 0.73–0.77 (0.82–0.86) | 0.02–0.95 (0.01–0.97) | 0.03–1.00 |
| shared subtractor, choice held uniform for 400 steps, handover over 500 steps | 10 | 4/4 | 0.72–0.76 (0.74–0.75) | 1.00 (1.00) | 1.00 |
| shared subtractor, choice held uniform for 400 steps, phase B only, from S4 weights, phase B hard_borrow=all, borrow a bit between places | 6 | 0/4 | 0.43–0.59 (0.91–0.98) | 0.01–0.03 (0.04–0.05) | 0.02–0.04 |
| shared subtractor, choice held uniform for 400 steps, phase B only, from S4 weights, phase B hard_borrow=all, borrow a bit between places | 7 | 0/4 | 0.68–0.74 (0.79–0.84) | 0.02 (0.01) | 0.05–0.72 |
| shared subtractor, choice held uniform for 400 steps, phase B only, from S4 weights, phase B hard_borrow=all, borrow a bit between places | 10 | 0/4 | 0.60–0.70 (0.59–0.72) | 0.01 (0.00) | 0.00–0.27 |

### widths

| variant | trained at | holdout | 6 bits | 7 bits | 10 bits | 14 bits | 20 bits |
|---|---|---|---|---|---|---|---|
| shared subtractor | 6 | 0 | 0.07–1.00 (0.06–1.00) | 0.01–0.81 (0.01–0.81) | 0.00–0.50 (0.00–0.54) | 0.00–0.16 (0.00–0.25) | 0.00–0.02 (0.00–0.05) |
| shared subtractor | 6 | 4 | 0.08–1.00 (0.04–1.00) | 0.00–0.80 (0.00–0.80) | 0.00–0.50 (0.00–0.54) | 0.00–0.16 (0.00–0.25) | 0.00–0.02 (0.00–0.05) |
| shared subtractor | 7 | 0 | 0.04–0.85 (0.03–1.00) | 0.02–0.99 (0.02–1.00) | 0.00–0.74 (0.00–0.87) | 0.00–0.24 (0.00–0.61) | 0.00–0.01 (0.00–0.19) |
| shared subtractor | 7 | 4 | 0.02–1.00 (0.04–1.00) | 0.03–1.00 (0.03–1.00) | 0.00–1.00 (0.01–1.00) | 0.00–1.00 (0.00–1.00) | 0.00–1.00 (0.00–1.00) |
| shared subtractor | 10 | 0 | 0.01–1.00 (0.06–1.00) | 0.00–1.00 (0.02–1.00) | 0.00–1.00 (0.00–1.00) | 0.00–1.00 (0.00–1.00) | 0.00–1.00 (0.00–1.00) |
| shared subtractor | 10 | 4 | 0.01–1.00 (0.05–1.00) | 0.01–1.00 (0.02–1.00) | 0.00–1.00 (0.00–1.00) | 0.00–1.00 (0.00–1.00) | 0.00–1.00 (0.00–1.00) |
| learned slots (longdiv's cell) | 6 | 0 | 0.45–0.71 (0.02) | 0.01 (0.00–0.01) | 0.00 (0.00) | 0.00 (0.00) | 0.00 (0.00) |
| learned slots (longdiv's cell) | 7 | 0 | 0.02 (0.01–0.04) | 0.05–0.06 (0.02–0.04) | 0.00 (0.00) | 0.00 (0.00) | 0.00 (0.00) |
| learned slots (longdiv's cell) | 10 | 0 | 0.03–0.04 (0.01–0.04) | 0.02 (0.01–0.02) | 0.00–0.01 (0.00) | 0.00 (0.00) | 0.00 (0.00) |
| shared subtractor, always from the top w digits | 6 | 0 | 0.07–0.27 (0.16–0.54) | 0.01 (0.04–0.11) | 0.00 (0.00–0.02) | 0.00 (0.00) | 0.00 (0.00) |
| shared subtractor, always from the top w digits | 7 | 0 | 0.04–0.05 (0.03–0.74) | 0.03–0.06 (0.02–0.65) | 0.00 (0.00–0.08) | 0.00 (0.00–0.01) | 0.00 (0.00) |
| shared subtractor, always from the top w digits | 10 | 0 | 0.03 (0.05–0.07) | 0.02 (0.01–0.02) | 0.00 (0.01) | 0.00 (0.00) | 0.00 (0.00) |
| shared subtractor, choice held uniform for 400 steps | 6 | 0 | 1.00 (1.00) | 0.80–0.81 (0.80–0.81) | 0.49–0.51 (0.53–0.55) | 0.15–0.16 (0.24–0.25) | 0.02 (0.05–0.06) |
| shared subtractor, choice held uniform for 400 steps | 6 | 4 | 1.00 (1.00) | 0.78–0.81 (0.79–0.81) | 0.46–0.51 (0.52–0.55) | 0.13–0.16 (0.23–0.26) | 0.01–0.02 (0.05–0.06) |
| shared subtractor, choice held uniform for 400 steps | 7 | 0 | 0.08–0.94 (0.83–1.00) | 0.50–0.99 (0.72–1.00) | 0.01–0.93 (0.10–0.97) | 0.00–0.77 (0.01–0.90) | 0.00–0.44 (0.00–0.68) |
| shared subtractor, choice held uniform for 400 steps | 7 | 4 | 0.02–0.84 (0.35–1.00) | 0.00–0.98 (0.10–1.00) | 0.00–0.19 (0.03–0.64) | 0.00–0.01 (0.00–0.26) | 0.00 (0.00–0.02) |
| shared subtractor, choice held uniform for 400 steps | 10 | 0 | 1.00 (1.00) | 1.00 (1.00) | 1.00 (1.00) | 0.97–1.00 (1.00) | 1.00 (1.00) |
| shared subtractor, choice held uniform for 400 steps | 10 | 4 | 1.00 (1.00) | 1.00 (1.00) | 1.00 (1.00) | 0.97–1.00 (1.00) | 1.00 (1.00) |
| shared subtractor, choice held uniform for 400 steps, borrow a bit between places | 6 | 0 | 1.00 (1.00) | 0.80–0.81 (0.80–0.81) | 0.49–0.51 (0.54–0.55) | 0.15–0.17 (0.25–0.26) | 0.02 (0.05–0.06) |
| shared subtractor, choice held uniform for 400 steps, borrow a bit between places | 6 | 4 | 0.15–1.00 (0.60–1.00) | 0.01–0.81 (0.15–0.81) | 0.00–0.51 (0.04–0.55) | 0.00–0.16 (0.01–0.26) | 0.00–0.02 (0.00–0.06) |
| shared subtractor, choice held uniform for 400 steps, borrow a bit between places | 7 | 0 | 0.01–1.00 (0.52–1.00) | 0.03–1.00 (0.28–1.00) | 0.00–1.00 (0.10–1.00) | 0.00–1.00 (0.01–1.00) | 0.00–1.00 (0.00–1.00) |
| shared subtractor, choice held uniform for 400 steps, borrow a bit between places | 7 | 4 | 0.01–1.00 (0.64–1.00) | 0.04–1.00 (0.34–1.00) | 0.00–1.00 (0.14–1.00) | 0.00–1.00 (0.03–1.00) | 0.00–1.00 (0.00–1.00) |
| shared subtractor, choice held uniform for 400 steps, borrow a bit between places | 10 | 0 | 0.01–0.02 (0.58–0.68) | 0.01–0.02 (0.27–0.41) | 0.00 (0.09–0.18) | 0.00 (0.02–0.05) | 0.00 (0.00–0.01) |
| shared subtractor, choice held uniform for 400 steps, borrow a bit between places | 10 | 4 | 0.02 (0.13–0.64) | 0.00–0.01 (0.04–0.32) | 0.00 (0.01–0.11) | 0.00 (0.00–0.02) | 0.00 (0.00) |
| shared subtractor, choice held uniform for 400 steps, handover over 500 steps | 6 | 0 | 0.97–1.00 (0.99–1.00) | 0.41–0.81 (0.69–0.81) | 0.12–0.51 (0.38–0.55) | 0.02–0.16 (0.12–0.26) | 0.00–0.02 (0.01–0.06) |
| shared subtractor, choice held uniform for 400 steps, handover over 500 steps | 6 | 4 | 1.00 (1.00) | 0.68–0.82 (0.74–0.82) | 0.33–0.52 (0.45–0.56) | 0.07–0.17 (0.18–0.27) | 0.00–0.02 (0.03–0.06) |
| shared subtractor, choice held uniform for 400 steps, handover over 500 steps | 7 | 0 | 0.13–0.79 (0.84–1.00) | 0.40–0.99 (0.72–1.00) | 0.01–0.70 (0.11–0.85) | 0.00–0.27 (0.01–0.58) | 0.00–0.01 (0.00–0.14) |
| shared subtractor, choice held uniform for 400 steps, handover over 500 steps | 7 | 4 | 0.01–0.38 (0.05–1.00) | 0.02–0.96 (0.03–1.00) | 0.00–0.25 (0.01–0.73) | 0.00–0.07 (0.00–0.35) | 0.00 (0.00–0.03) |
| shared subtractor, choice held uniform for 400 steps, handover over 500 steps | 10 | 0 | 1.00 (1.00) | 1.00 (1.00) | 1.00 (1.00) | 1.00 (1.00) | 1.00 (1.00) |
| shared subtractor, choice held uniform for 400 steps, handover over 500 steps | 10 | 4 | 1.00 (1.00) | 1.00 (1.00) | 1.00 (1.00) | 1.00 (1.00) | 1.00 (1.00) |
| shared subtractor, choice held uniform for 400 steps, lr 0.0003 after the switch | 7 | 0 | 0.06–0.77 (0.97–1.00) | 0.05–0.95 (0.91–1.00) | 0.01–0.45 (0.16–0.80) | 0.00–0.07 (0.01–0.39) | 0.00 (0.00–0.05) |
| shared subtractor, choice held uniform for 400 steps, lr 0.001 after the switch | 7 | 0 | 0.09–0.76 (0.86–1.00) | 0.32–0.98 (0.70–1.00) | 0.01–0.16 (0.11–0.81) | 0.00–0.01 (0.01–0.32) | 0.00 (0.00–0.02) |
| shared subtractor, choice held uniform for 400 steps, phase B only, from S4 weights, phase B hard_borrow=all, borrow a bit between places | 6 | 0 | 0.02–0.04 (0.05–0.17) | 0.00–0.01 (0.00–0.09) | 0.00 (0.00–0.01) | 0.00 (0.00) | 0.00 (0.00) |
| shared subtractor, choice held uniform for 400 steps, phase B only, from S4 weights, phase B hard_borrow=all, borrow a bit between places | 6 | 4 | 0.04 (0.02–0.06) | 0.00–0.01 (0.00–0.01) | 0.00 (0.00) | 0.00 (0.00) | 0.00 (0.00) |
| shared subtractor, choice held uniform for 400 steps, phase B only, from S4 weights, phase B hard_borrow=all, borrow a bit between places | 7 | 0 | 0.03–1.00 (0.29–1.00) | 0.01–1.00 (0.14–1.00) | 0.00–1.00 (0.02–1.00) | 0.00–1.00 (0.00–1.00) | 0.00–1.00 (0.00–1.00) |
| shared subtractor, choice held uniform for 400 steps, phase B only, from S4 weights, phase B hard_borrow=all, borrow a bit between places | 7 | 4 | 0.02–0.07 (0.08–0.87) | 0.01 (0.03–0.69) | 0.00 (0.00–0.40) | 0.00 (0.00–0.13) | 0.00 (0.00–0.01) |
| shared subtractor, choice held uniform for 400 steps, phase B only, from S4 weights, phase B hard_borrow=all, borrow a bit between places | 10 | 0 | 0.02–1.00 (0.03–1.00) | 0.01–1.00 (0.00–1.00) | 0.00–1.00 (0.00–1.00) | 0.00–1.00 (0.00–1.00) | 0.00–1.00 (0.00–1.00) |
| shared subtractor, choice held uniform for 400 steps, phase B only, from S4 weights, phase B hard_borrow=all, borrow a bit between places | 10 | 4 | 0.02–0.13 (0.04–0.74) | 0.01 (0.01–0.57) | 0.00 (0.00–0.29) | 0.00 (0.00–0.07) | 0.00 (0.00) |
| shared subtractor, not staged | 6 | 0 | 0.10–0.16 (0.20–0.21) | 0.01 (0.03–0.05) | 0.00 (0.01) | 0.00 (0.00) | 0.00 (0.00) |
| shared subtractor, not staged | 7 | 0 | 0.04 (0.03–0.09) | 0.03 (0.03–0.08) | 0.00–0.01 (0.00–0.01) | 0.00 (0.00) | 0.00 (0.00) |
| shared subtractor, not staged | 10 | 0 | 0.03 (0.05) | 0.00–0.01 (0.00–0.02) | 0.00 (0.01) | 0.00 (0.00) | 0.00 (0.00) |
| shared subtractor, own borrow from the start | 6 | 0 | 1.00 (1.00) | 0.53–0.81 (0.68–0.81) | 0.17–0.51 (0.33–0.55) | 0.01–0.16 (0.09–0.26) | 0.00–0.02 (0.01–0.06) |
| shared subtractor, own borrow from the start | 7 | 0 | 0.01–0.02 (1.00) | 0.94–0.99 (0.98–1.00) | 0.00 (0.01–0.23) | 0.00 (0.01) | 0.00 (0.00) |
| shared subtractor, own borrow from the start | 10 | 0 | 0.02–0.04 (0.03–0.63) | 0.01 (0.01–0.29) | 0.00–0.01 (0.00–0.16) | 0.00 (0.00–0.01) | 0.00 (0.00) |
| shared subtractor, window 2/1 | 6 | 0 | 1.00 (1.00) | 0.80–0.81 (0.80–0.81) | 0.49–0.50 (0.53–0.54) | 0.14–0.16 (0.23–0.25) | 0.01–0.02 (0.05) |
| shared subtractor, window 2/1 | 7 | 0 | 0.02–0.58 (0.87–1.00) | 0.53–0.98 (0.79–0.99) | 0.00–0.01 (0.11–0.18) | 0.00 (0.01–0.03) | 0.00 (0.00) |
| shared subtractor, window 2/1 | 10 | 0 | 0.02–0.03 (0.08–0.97) | 0.01 (0.01–0.89) | 0.00–0.01 (0.01–0.68) | 0.00 (0.00–0.17) | 0.00 (0.00–0.01) |

### stages

| variant | bits | seed | holdout | stages: steps / exact on the stage's validation sample | own borrow after the switch, every 100 steps (first 6) |
|---|---|---|---|---|---|
| shared subtractor | 6 | 0 | 0 | k1: 1000/0.74 k2: 1500/0.87 | 0.03 0.99 1.00 1.00 1.00 1.00 |
| shared subtractor | 6 | 1 | 0 | k1: 1000/0.06 k2: 1500/0.04 | 0.04 0.04 0.04 0.05 0.04 0.05 |
| shared subtractor | 6 | 2 | 0 | k1: 1000/0.22 k2: 1500/0.41 | 0.03 0.11 0.97 1.00 1.00 1.00 |
| shared subtractor | 6 | 3 | 0 | k1: 1000/0.51 k2: 1500/0.85 | 0.02 0.95 1.00 1.00 1.00 1.00 |
| shared subtractor | 6 | 0 | 4 | k1: 1000/0.34 k2: 1500/0.95 | 0.04 0.99 1.00 1.00 1.00 1.00 |
| shared subtractor | 6 | 1 | 4 | k1: 1000/0.20 k2: 1500/0.15 | 0.03 0.06 0.10 0.15 0.19 0.80 |
| shared subtractor | 6 | 2 | 4 | k1: 1000/0.08 k2: 1500/0.04 | 0.05 0.05 0.05 0.05 0.06 0.06 |
| shared subtractor | 6 | 3 | 4 | k1: 1000/0.28 k2: 1500/0.40 | 0.03 0.54 0.99 0.99 1.00 1.00 |
| shared subtractor | 7 | 0 | 0 | k1: 1000/0.04 k2: 1000/0.01 k3: 1500/0.03 | 0.02 0.03 0.03 0.03 0.03 0.03 |
| shared subtractor | 7 | 1 | 0 | k1: 1000/0.73 k2: 1000/0.80 k3: 1500/0.84 | 0.02 0.03 0.03 0.03 0.03 0.02 |
| shared subtractor | 7 | 2 | 0 | k1: 1000/0.79 k2: 1000/0.82 k3: 1500/0.86 | 0.06 0.03 0.03 0.02 0.02 0.02 |
| shared subtractor | 7 | 3 | 0 | k1: 1000/0.56 k2: 1000/0.79 k3: 1500/0.86 | 0.20 0.03 0.02 0.03 0.03 0.04 |
| shared subtractor | 7 | 0 | 4 | k1: 1000/0.04 k2: 1000/0.03 k3: 1500/0.03 | 0.02 0.03 0.04 0.02 0.03 0.03 |
| shared subtractor | 7 | 1 | 4 | k1: 1000/0.58 k2: 1000/0.82 k3: 1500/0.87 | 0.12 0.01 0.02 0.03 0.03 0.03 |
| shared subtractor | 7 | 2 | 4 | k1: 1000/0.71 k2: 1000/0.79 k3: 1500/0.87 | 0.17 0.02 0.02 0.02 0.02 0.03 |
| shared subtractor | 7 | 3 | 4 | k1: 1000/0.59 k2: 1000/0.83 k3: 1500/0.87 | 0.12 0.04 0.15 0.87 1.00 1.00 |
| shared subtractor | 10 | 0 | 0 | k1: 1000/0.12 k2: 1000/0.02 k3: 1000/0.00 k4: 1500/0.01 | 0.00 0.01 0.00 0.00 0.00 0.00 |
| shared subtractor | 10 | 1 | 0 | k1: 1000/0.59 k2: 1000/0.68 k3: 1000/0.64 k4: 1500/0.76 | 0.00 0.00 0.00 0.00 0.00 0.00 |
| shared subtractor | 10 | 2 | 0 | k1: 1000/0.83 k2: 1000/0.73 k3: 1000/0.66 k4: 1500/0.76 | 0.00 0.12 1.00 1.00 1.00 1.00 |
| shared subtractor | 10 | 3 | 0 | k1: 1000/0.80 k2: 1000/0.71 k3: 1000/0.66 k4: 1500/0.77 | 0.00 1.00 1.00 1.00 1.00 1.00 |
| shared subtractor | 10 | 0 | 4 | k1: 1000/0.10 k2: 1000/0.01 k3: 1000/0.00 k4: 1500/0.00 | 0.00 0.00 0.00 0.00 0.00 0.00 |
| shared subtractor | 10 | 1 | 4 | k1: 1000/0.01 k2: 1000/0.01 k3: 1000/0.00 k4: 1500/0.00 | 0.00 0.00 0.00 0.00 0.00 0.00 |
| shared subtractor | 10 | 2 | 4 | k1: 1000/0.84 k2: 1000/0.71 k3: 1000/0.65 k4: 1500/0.75 | 0.00 0.00 0.00 1.00 1.00 1.00 |
| shared subtractor | 10 | 3 | 4 | k1: 1000/0.59 k2: 1000/0.67 k3: 1000/0.61 k4: 1500/0.70 | 0.00 0.01 0.00 0.00 0.00 0.00 |
| learned slots (longdiv's cell) | 6 | 0 | 0 | k1: 1000/0.32 k2: 1500/0.50 | 0.04 0.13 0.32 0.43 0.49 0.53 |
| learned slots (longdiv's cell) | 6 | 1 | 0 | k1: 1000/0.40 k2: 1500/0.43 | 0.04 0.11 0.26 0.30 0.37 0.40 |
| learned slots (longdiv's cell) | 6 | 2 | 0 | k1: 1000/0.39 k2: 1500/0.42 | 0.05 0.12 0.15 0.19 0.23 0.27 |
| learned slots (longdiv's cell) | 7 | 0 | 0 | k1: 1000/0.23 k2: 1000/0.12 k3: 1500/0.42 | 0.02 0.02 0.02 0.02 0.02 0.02 |
| learned slots (longdiv's cell) | 7 | 1 | 0 | k1: 1000/0.16 k2: 1000/0.07 k3: 1500/0.30 | 0.03 0.03 0.03 0.02 0.03 0.03 |
| learned slots (longdiv's cell) | 7 | 2 | 0 | k1: 1000/0.16 k2: 1000/0.08 k3: 1500/0.25 | 0.03 0.02 0.03 0.02 0.03 0.03 |
| learned slots (longdiv's cell) | 10 | 0 | 0 | k1: 1000/0.24 k2: 1000/0.14 k3: 1000/0.08 k4: 1500/0.13 | 0.00 0.00 0.00 0.00 0.00 0.00 |
| learned slots (longdiv's cell) | 10 | 1 | 0 | k1: 1000/0.34 k2: 1000/0.16 k3: 1000/0.08 k4: 1500/0.17 | 0.00 0.00 0.00 0.00 0.00 0.00 |
| learned slots (longdiv's cell) | 10 | 2 | 0 | k1: 1000/0.29 k2: 1000/0.10 k3: 1000/0.05 k4: 1500/0.09 | 0.00 0.00 0.00 0.00 0.00 0.00 |
| shared subtractor, always from the top w digits | 6 | 0 | 0 | k1: 1000/0.24 k2: 1500/0.33 | 0.04 0.06 0.08 0.11 0.12 0.13 |
| shared subtractor, always from the top w digits | 6 | 1 | 0 | k1: 1000/0.04 k2: 1500/0.04 | 0.04 0.04 0.04 0.04 0.03 0.04 |
| shared subtractor, always from the top w digits | 7 | 0 | 0 | k1: 1000/0.03 k2: 1000/0.02 k3: 1500/0.03 | 0.03 0.03 0.03 0.03 0.03 0.03 |
| shared subtractor, always from the top w digits | 7 | 1 | 0 | k1: 1000/0.04 k2: 1000/0.06 k3: 1500/0.68 | 0.06 0.03 0.03 0.02 0.03 0.03 |
| shared subtractor, always from the top w digits | 10 | 0 | 0 | k1: 1000/0.00 k2: 1000/0.00 k3: 1000/0.00 k4: 1500/0.00 | 0.00 0.00 0.00 0.01 0.00 0.00 |
| shared subtractor, always from the top w digits | 10 | 1 | 0 | k1: 1000/0.00 k2: 1000/0.00 k3: 1000/0.00 k4: 1500/0.00 | 0.00 0.00 0.00 0.00 0.00 0.00 |
| shared subtractor, choice held uniform for 400 steps | 6 | 0 | 0 | k1: 1000/0.92 k2: 1500/0.91 | 0.04 1.00 1.00 1.00 1.00 1.00 |
| shared subtractor, choice held uniform for 400 steps | 6 | 1 | 0 | k1: 1000/0.95 k2: 1500/0.98 | 0.04 0.46 1.00 0.99 1.00 1.00 |
| shared subtractor, choice held uniform for 400 steps | 6 | 2 | 0 | k1: 1000/0.91 k2: 1500/0.92 | 0.13 1.00 1.00 1.00 1.00 1.00 |
| shared subtractor, choice held uniform for 400 steps | 6 | 3 | 0 | k1: 1000/0.94 k2: 1500/0.97 | 0.05 0.95 1.00 1.00 1.00 1.00 |
| shared subtractor, choice held uniform for 400 steps | 6 | 0 | 4 | k1: 1000/0.97 k2: 1500/0.98 | 0.07 1.00 1.00 1.00 1.00 1.00 |
| shared subtractor, choice held uniform for 400 steps | 6 | 1 | 4 | k1: 1000/0.96 k2: 1500/0.96 | 0.05 0.98 1.00 1.00 1.00 1.00 |
| shared subtractor, choice held uniform for 400 steps | 6 | 2 | 4 | k1: 1000/0.94 k2: 1500/0.93 | 0.56 0.98 1.00 1.00 1.00 1.00 |
| shared subtractor, choice held uniform for 400 steps | 6 | 3 | 4 | k1: 1000/0.89 k2: 1500/0.91 | 0.14 0.99 0.99 1.00 1.00 1.00 |
| shared subtractor, choice held uniform for 400 steps | 7 | 0 | 0 | k1: 1000/0.75 k2: 1000/0.76 k3: 1500/0.82 | 0.10 0.02 0.06 0.09 0.15 0.19 |
| shared subtractor, choice held uniform for 400 steps | 7 | 0 | 0 | k1: 1000/0.74 k2: 1000/0.78 k3: 1500/0.84 | 0.20 0.03 0.07 0.21 0.30 0.41 |
| shared subtractor, choice held uniform for 400 steps | 7 | 1 | 0 | k1: 1000/0.86 k2: 1000/0.80 k3: 1500/0.85 | 0.03 0.04 0.62 0.81 0.34 0.66 |
| shared subtractor, choice held uniform for 400 steps | 7 | 1 | 0 | k1: 1000/0.86 k2: 1000/0.80 k3: 1500/0.85 | 0.03 0.04 0.62 0.81 0.34 0.66 |
| shared subtractor, choice held uniform for 400 steps | 7 | 2 | 0 | k1: 1000/0.81 k2: 1000/0.76 k3: 1500/0.82 | 0.12 0.31 0.74 0.93 0.94 0.53 |
| shared subtractor, choice held uniform for 400 steps | 7 | 2 | 0 | k1: 1000/0.81 k2: 1000/0.76 k3: 1500/0.82 | 0.12 0.31 0.74 0.93 0.94 0.53 |
| shared subtractor, choice held uniform for 400 steps | 7 | 3 | 0 | k1: 1000/0.81 k2: 1000/0.77 k3: 1500/0.83 | 0.27 0.07 0.83 0.99 0.99 0.99 |
| shared subtractor, choice held uniform for 400 steps | 7 | 3 | 0 | k1: 1000/0.81 k2: 1000/0.77 k3: 1500/0.83 | 0.27 0.07 0.83 0.99 0.99 0.99 |
| shared subtractor, choice held uniform for 400 steps | 7 | 0 | 4 | k1: 1000/0.78 k2: 1000/0.80 k3: 1500/0.86 | 0.02 0.03 0.03 0.11 0.24 0.28 |
| shared subtractor, choice held uniform for 400 steps | 7 | 1 | 4 | k1: 1000/0.72 k2: 1000/0.80 k3: 1500/0.84 | 0.06 0.02 0.01 0.01 0.01 0.01 |
| shared subtractor, choice held uniform for 400 steps | 7 | 2 | 4 | k1: 1000/0.72 k2: 1000/0.79 k3: 1500/0.83 | 0.04 0.06 0.12 0.28 0.58 0.94 |
| shared subtractor, choice held uniform for 400 steps | 7 | 3 | 4 | k1: 1000/0.78 k2: 1000/0.76 k3: 1500/0.83 | 0.09 0.02 0.24 0.44 0.71 0.88 |
| shared subtractor, choice held uniform for 400 steps | 10 | 0 | 0 | k1: 1000/0.83 k2: 1000/0.69 k3: 1000/0.65 k4: 1500/0.74 | 0.00 1.00 1.00 1.00 1.00 1.00 |
| shared subtractor, choice held uniform for 400 steps | 10 | 1 | 0 | k1: 1000/0.83 k2: 1000/0.70 k3: 1000/0.65 k4: 1500/0.72 | 0.00 1.00 1.00 1.00 1.00 1.00 |
| shared subtractor, choice held uniform for 400 steps | 10 | 2 | 0 | k1: 1000/0.80 k2: 1000/0.71 k3: 1000/0.64 k4: 1500/0.73 | 0.12 1.00 1.00 1.00 1.00 1.00 |
| shared subtractor, choice held uniform for 400 steps | 10 | 3 | 0 | k1: 1000/0.84 k2: 1000/0.69 k3: 1000/0.63 k4: 1500/0.76 | 0.88 1.00 1.00 1.00 1.00 1.00 |
| shared subtractor, choice held uniform for 400 steps | 10 | 0 | 4 | k1: 1000/0.86 k2: 1000/0.70 k3: 1000/0.62 k4: 1500/0.72 | 0.00 0.01 1.00 1.00 1.00 1.00 |
| shared subtractor, choice held uniform for 400 steps | 10 | 1 | 4 | k1: 1000/0.86 k2: 1000/0.69 k3: 1000/0.62 k4: 1500/0.73 | 0.00 1.00 1.00 1.00 1.00 1.00 |
| shared subtractor, choice held uniform for 400 steps | 10 | 2 | 4 | k1: 1000/0.84 k2: 1000/0.70 k3: 1000/0.62 k4: 1500/0.73 | 0.00 1.00 1.00 1.00 1.00 1.00 |
| shared subtractor, choice held uniform for 400 steps | 10 | 3 | 4 | k1: 1000/0.85 k2: 1000/0.69 k3: 1000/0.62 k4: 1500/0.74 | 0.00 1.00 1.00 1.00 1.00 1.00 |
| shared subtractor, choice held uniform for 400 steps, borrow a bit between places | 6 | 0 | 0 | k1: 1000/0.88 k2: 1500/0.89 | 0.10 0.96 1.00 1.00 0.97 1.00 |
| shared subtractor, choice held uniform for 400 steps, borrow a bit between places | 6 | 1 | 0 | k1: 1000/0.90 k2: 1500/0.94 | 0.03 0.19 1.00 1.00 1.00 1.00 |
| shared subtractor, choice held uniform for 400 steps, borrow a bit between places | 6 | 2 | 0 | k1: 1000/0.95 k2: 1500/0.98 | 0.05 0.07 1.00 1.00 1.00 1.00 |
| shared subtractor, choice held uniform for 400 steps, borrow a bit between places | 6 | 3 | 0 | k1: 1000/0.84 k2: 1500/0.96 | 0.04 0.92 1.00 1.00 1.00 1.00 |
| shared subtractor, choice held uniform for 400 steps, borrow a bit between places | 6 | 0 | 4 | k1: 1000/0.53 k2: 1500/0.92 | 0.02 0.06 0.11 0.13 0.14 0.17 |
| shared subtractor, choice held uniform for 400 steps, borrow a bit between places | 6 | 1 | 4 | k1: 1000/0.85 k2: 1500/0.94 | 0.04 0.93 1.00 1.00 1.00 1.00 |
| shared subtractor, choice held uniform for 400 steps, borrow a bit between places | 6 | 2 | 4 | k1: 1000/0.92 k2: 1500/0.90 | 0.05 0.56 1.00 1.00 1.00 1.00 |
| shared subtractor, choice held uniform for 400 steps, borrow a bit between places | 6 | 3 | 4 | k1: 1000/0.75 k2: 1500/0.97 | 0.06 0.44 1.00 1.00 1.00 1.00 |
| shared subtractor, choice held uniform for 400 steps, borrow a bit between places | 7 | 0 | 0 | k1: 1000/0.87 k2: 1000/0.83 k3: 1500/0.86 | 0.01 0.02 0.02 0.02 0.03 0.02 |
| shared subtractor, choice held uniform for 400 steps, borrow a bit between places | 7 | 1 | 0 | k1: 1000/0.84 k2: 1000/0.80 k3: 1500/0.86 | 0.65 1.00 1.00 1.00 1.00 1.00 |
| shared subtractor, choice held uniform for 400 steps, borrow a bit between places | 7 | 2 | 0 | k1: 1000/0.87 k2: 1000/0.71 k3: 1500/0.85 | 0.01 0.02 0.03 0.02 0.02 0.03 |
| shared subtractor, choice held uniform for 400 steps, borrow a bit between places | 7 | 3 | 0 | k1: 1000/0.87 k2: 1000/0.55 k3: 1500/0.78 | 0.03 0.03 0.02 0.03 0.02 0.03 |
| shared subtractor, choice held uniform for 400 steps, borrow a bit between places | 7 | 0 | 4 | k1: 1000/0.88 k2: 1000/0.85 k3: 1500/0.84 | 0.91 0.41 0.03 0.03 0.03 0.02 |
| shared subtractor, choice held uniform for 400 steps, borrow a bit between places | 7 | 1 | 4 | k1: 1000/0.86 k2: 1000/0.77 k3: 1500/0.84 | 0.02 0.53 1.00 1.00 1.00 1.00 |
| shared subtractor, choice held uniform for 400 steps, borrow a bit between places | 7 | 2 | 4 | k1: 1000/0.87 k2: 1000/0.79 k3: 1500/0.78 | 0.25 0.20 0.08 0.25 0.29 0.65 |
| shared subtractor, choice held uniform for 400 steps, borrow a bit between places | 7 | 3 | 4 | k1: 1000/0.81 k2: 1000/0.76 k3: 1500/0.82 | 0.06 0.04 0.03 0.04 0.04 0.04 |
| shared subtractor, choice held uniform for 400 steps, borrow a bit between places | 10 | 0 | 0 | k1: 1000/0.81 k2: 1000/0.66 k3: 1000/0.57 k4: 1500/0.64 | 0.00 0.00 0.00 0.00 0.00 0.00 |
| shared subtractor, choice held uniform for 400 steps, borrow a bit between places | 10 | 1 | 0 | k1: 1000/0.80 k2: 1000/0.69 k3: 1000/0.56 k4: 1500/0.71 | 0.00 0.00 0.00 0.00 0.00 0.00 |
| shared subtractor, choice held uniform for 400 steps, borrow a bit between places | 10 | 2 | 0 | k1: 1000/0.82 k2: 1000/0.66 k3: 1000/0.60 k4: 1500/0.71 | 0.00 0.00 0.00 0.00 0.00 0.00 |
| shared subtractor, choice held uniform for 400 steps, borrow a bit between places | 10 | 3 | 0 | k1: 1000/0.82 k2: 1000/0.67 k3: 1000/0.58 k4: 1500/0.67 | 0.00 0.01 0.00 0.00 0.00 0.00 |
| shared subtractor, choice held uniform for 400 steps, borrow a bit between places | 10 | 0 | 4 | k1: 1000/0.83 k2: 1000/0.70 k3: 1000/0.55 k4: 1500/0.65 | 0.00 0.00 0.00 0.00 0.00 0.00 |
| shared subtractor, choice held uniform for 400 steps, borrow a bit between places | 10 | 1 | 4 | k1: 1000/0.84 k2: 1000/0.65 k3: 1000/0.56 k4: 1500/0.67 | 0.00 0.00 0.00 0.00 0.00 0.00 |
| shared subtractor, choice held uniform for 400 steps, borrow a bit between places | 10 | 2 | 4 | k1: 1000/0.82 k2: 1000/0.66 k3: 1000/0.59 k4: 1500/0.58 | 0.00 0.00 0.00 0.00 0.00 0.00 |
| shared subtractor, choice held uniform for 400 steps, borrow a bit between places | 10 | 3 | 4 | k1: 1000/0.82 k2: 1000/0.67 k3: 1000/0.57 k4: 1500/0.70 | 0.00 0.00 0.00 0.00 0.00 0.00 |
| shared subtractor, choice held uniform for 400 steps, handover over 500 steps | 6 | 0 | 0 | k1: 1000/0.92 k2: 1500/0.91 | 0.91 0.92 0.94 0.99 1.00 1.00 |
| shared subtractor, choice held uniform for 400 steps, handover over 500 steps | 6 | 1 | 0 | k1: 1000/0.95 k2: 1500/0.94 | 0.94 0.82 0.96 0.96 0.95 0.96 |
| shared subtractor, choice held uniform for 400 steps, handover over 500 steps | 6 | 2 | 0 | k1: 1000/0.91 k2: 1500/0.91 | 0.91 0.92 0.94 1.00 1.00 1.00 |
| shared subtractor, choice held uniform for 400 steps, handover over 500 steps | 6 | 3 | 0 | k1: 1000/0.94 k2: 1500/0.97 | 0.97 0.96 0.98 1.00 1.00 0.99 |
| shared subtractor, choice held uniform for 400 steps, handover over 500 steps | 6 | 0 | 4 | k1: 1000/0.97 k2: 1500/0.97 | 0.97 0.97 0.98 1.00 1.00 1.00 |
| shared subtractor, choice held uniform for 400 steps, handover over 500 steps | 6 | 1 | 4 | k1: 1000/0.96 k2: 1500/0.96 | 0.96 0.95 0.97 1.00 1.00 1.00 |
| shared subtractor, choice held uniform for 400 steps, handover over 500 steps | 6 | 2 | 4 | k1: 1000/0.94 k2: 1500/0.93 | 0.93 0.94 0.95 1.00 0.97 1.00 |
| shared subtractor, choice held uniform for 400 steps, handover over 500 steps | 6 | 3 | 4 | k1: 1000/0.92 k2: 1500/0.94 | 0.94 0.94 0.95 0.97 0.99 1.00 |
| shared subtractor, choice held uniform for 400 steps, handover over 500 steps | 7 | 0 | 0 | k1: 1000/0.74 k2: 1000/0.78 k3: 1500/0.84 | 0.84 0.84 0.86 0.77 0.77 0.63 |
| shared subtractor, choice held uniform for 400 steps, handover over 500 steps | 7 | 1 | 0 | k1: 1000/0.86 k2: 1000/0.80 k3: 1500/0.85 | 0.85 0.88 0.90 0.92 0.74 0.73 |
| shared subtractor, choice held uniform for 400 steps, handover over 500 steps | 7 | 2 | 0 | k1: 1000/0.81 k2: 1000/0.76 k3: 1500/0.81 | 0.81 0.72 0.66 0.48 0.11 0.02 |
| shared subtractor, choice held uniform for 400 steps, handover over 500 steps | 7 | 3 | 0 | k1: 1000/0.81 k2: 1000/0.77 k3: 1500/0.83 | 0.83 0.80 0.87 0.86 0.91 0.90 |
| shared subtractor, choice held uniform for 400 steps, handover over 500 steps | 7 | 0 | 4 | k1: 1000/0.77 k2: 1000/0.82 k3: 1500/0.86 | 0.86 0.84 0.74 0.61 0.79 0.80 |
| shared subtractor, choice held uniform for 400 steps, handover over 500 steps | 7 | 1 | 4 | k1: 1000/0.72 k2: 1000/0.80 k3: 1500/0.84 | 0.84 0.80 0.66 0.42 0.15 0.02 |
| shared subtractor, choice held uniform for 400 steps, handover over 500 steps | 7 | 2 | 4 | k1: 1000/0.75 k2: 1000/0.79 k3: 1500/0.85 | 0.85 0.83 0.79 0.85 0.74 0.39 |
| shared subtractor, choice held uniform for 400 steps, handover over 500 steps | 7 | 3 | 4 | k1: 1000/0.78 k2: 1000/0.76 k3: 1500/0.83 | 0.83 0.80 0.79 0.68 0.59 0.58 |
| shared subtractor, choice held uniform for 400 steps, handover over 500 steps | 10 | 0 | 0 | k1: 1000/0.83 k2: 1000/0.69 k3: 1000/0.65 k4: 1500/0.76 | 0.76 0.85 0.96 1.00 1.00 1.00 |
| shared subtractor, choice held uniform for 400 steps, handover over 500 steps | 10 | 1 | 0 | k1: 1000/0.83 k2: 1000/0.68 k3: 1000/0.65 k4: 1500/0.73 | 0.73 0.85 0.95 1.00 1.00 1.00 |
| shared subtractor, choice held uniform for 400 steps, handover over 500 steps | 10 | 2 | 0 | k1: 1000/0.80 k2: 1000/0.71 k3: 1000/0.64 k4: 1500/0.73 | 0.73 0.84 0.94 1.00 1.00 1.00 |
| shared subtractor, choice held uniform for 400 steps, handover over 500 steps | 10 | 3 | 0 | k1: 1000/0.84 k2: 1000/0.69 k3: 1000/0.63 k4: 1500/0.75 | 0.75 0.85 0.94 1.00 1.00 1.00 |
| shared subtractor, choice held uniform for 400 steps, handover over 500 steps | 10 | 0 | 4 | k1: 1000/0.85 k2: 1000/0.70 k3: 1000/0.64 k4: 1500/0.71 | 0.71 0.83 0.95 1.00 1.00 1.00 |
| shared subtractor, choice held uniform for 400 steps, handover over 500 steps | 10 | 1 | 4 | k1: 1000/0.86 k2: 1000/0.69 k3: 1000/0.62 k4: 1500/0.73 | 0.73 0.82 0.94 1.00 1.00 1.00 |
| shared subtractor, choice held uniform for 400 steps, handover over 500 steps | 10 | 2 | 4 | k1: 1000/0.84 k2: 1000/0.70 k3: 1000/0.62 k4: 1500/0.73 | 0.73 0.83 0.94 1.00 1.00 1.00 |
| shared subtractor, choice held uniform for 400 steps, handover over 500 steps | 10 | 3 | 4 | k1: 1000/0.84 k2: 1000/0.69 k3: 1000/0.62 k4: 1500/0.75 | 0.75 0.83 0.94 1.00 1.00 1.00 |
| shared subtractor, choice held uniform for 400 steps, lr 0.0003 after the switch | 7 | 0 | 0 | k1: 1000/0.75 k2: 1000/0.76 k3: 1500/0.82 | 0.10 0.07 0.03 0.03 0.03 0.03 |
| shared subtractor, choice held uniform for 400 steps, lr 0.0003 after the switch | 7 | 1 | 0 | k1: 1000/0.86 k2: 1000/0.80 k3: 1500/0.85 | 0.03 0.04 0.15 0.31 0.67 0.85 |
| shared subtractor, choice held uniform for 400 steps, lr 0.0003 after the switch | 7 | 2 | 0 | k1: 1000/0.81 k2: 1000/0.76 k3: 1500/0.82 | 0.12 0.13 0.20 0.28 0.35 0.37 |
| shared subtractor, choice held uniform for 400 steps, lr 0.0003 after the switch | 7 | 3 | 0 | k1: 1000/0.81 k2: 1000/0.77 k3: 1500/0.83 | 0.27 0.26 0.16 0.19 0.20 0.31 |
| shared subtractor, choice held uniform for 400 steps, lr 0.001 after the switch | 7 | 0 | 0 | k1: 1000/0.74 k2: 1000/0.79 k3: 1500/0.84 | 0.13 0.02 0.03 0.04 0.09 0.13 |
| shared subtractor, choice held uniform for 400 steps, lr 0.001 after the switch | 7 | 1 | 0 | k1: 1000/0.85 k2: 1000/0.79 k3: 1500/0.84 | 0.19 0.15 0.66 0.76 0.88 0.93 |
| shared subtractor, choice held uniform for 400 steps, lr 0.001 after the switch | 7 | 2 | 0 | k1: 1000/0.81 k2: 1000/0.76 k3: 1500/0.82 | 0.12 0.02 0.32 0.48 0.86 0.93 |
| shared subtractor, choice held uniform for 400 steps, lr 0.001 after the switch | 7 | 3 | 0 | k1: 1000/0.81 k2: 1000/0.77 k3: 1500/0.83 | 0.27 0.16 0.04 0.05 0.27 0.82 |
| shared subtractor, choice held uniform for 400 steps, phase B only, from S4 weights, phase B hard_borrow=all, borrow a bit between places | 6 | 0 | 0 |  | 0.23 0.07 0.02 0.03 0.03 0.02 |
| shared subtractor, choice held uniform for 400 steps, phase B only, from S4 weights, phase B hard_borrow=all, borrow a bit between places | 6 | 1 | 0 |  | 0.05 0.03 0.02 0.01 0.04 0.04 |
| shared subtractor, choice held uniform for 400 steps, phase B only, from S4 weights, phase B hard_borrow=all, borrow a bit between places | 6 | 2 | 0 |  | 0.04 0.02 0.03 0.04 0.04 0.04 |
| shared subtractor, choice held uniform for 400 steps, phase B only, from S4 weights, phase B hard_borrow=all, borrow a bit between places | 6 | 3 | 0 |  | 0.22 0.14 0.02 0.02 0.02 0.02 |
| shared subtractor, choice held uniform for 400 steps, phase B only, from S4 weights, phase B hard_borrow=all, borrow a bit between places | 6 | 0 | 4 |  | 0.03 0.03 0.03 0.02 0.01 0.00 |
| shared subtractor, choice held uniform for 400 steps, phase B only, from S4 weights, phase B hard_borrow=all, borrow a bit between places | 6 | 1 | 4 |  | 0.04 0.05 0.01 0.04 0.04 0.04 |
| shared subtractor, choice held uniform for 400 steps, phase B only, from S4 weights, phase B hard_borrow=all, borrow a bit between places | 6 | 2 | 4 |  | 0.05 0.04 0.02 0.02 0.01 0.05 |
| shared subtractor, choice held uniform for 400 steps, phase B only, from S4 weights, phase B hard_borrow=all, borrow a bit between places | 6 | 3 | 4 |  | 0.07 0.03 0.04 0.01 0.04 0.06 |
| shared subtractor, choice held uniform for 400 steps, phase B only, from S4 weights, phase B hard_borrow=all, borrow a bit between places | 7 | 0 | 0 |  | 1.00 1.00 1.00 1.00 1.00 1.00 |
| shared subtractor, choice held uniform for 400 steps, phase B only, from S4 weights, phase B hard_borrow=all, borrow a bit between places | 7 | 1 | 0 |  | 0.93 1.00 1.00 1.00 1.00 1.00 |
| shared subtractor, choice held uniform for 400 steps, phase B only, from S4 weights, phase B hard_borrow=all, borrow a bit between places | 7 | 2 | 0 |  | 0.01 0.01 0.01 0.01 0.01 0.01 |
| shared subtractor, choice held uniform for 400 steps, phase B only, from S4 weights, phase B hard_borrow=all, borrow a bit between places | 7 | 3 | 0 |  | 0.58 0.01 0.01 0.01 0.01 0.01 |
| shared subtractor, choice held uniform for 400 steps, phase B only, from S4 weights, phase B hard_borrow=all, borrow a bit between places | 7 | 0 | 4 |  | 0.90 0.01 0.01 0.01 0.01 0.01 |
| shared subtractor, choice held uniform for 400 steps, phase B only, from S4 weights, phase B hard_borrow=all, borrow a bit between places | 7 | 1 | 4 |  | 0.75 0.17 0.01 0.01 0.01 0.01 |
| shared subtractor, choice held uniform for 400 steps, phase B only, from S4 weights, phase B hard_borrow=all, borrow a bit between places | 7 | 2 | 4 |  | 0.20 0.02 0.01 0.01 0.01 0.01 |
| shared subtractor, choice held uniform for 400 steps, phase B only, from S4 weights, phase B hard_borrow=all, borrow a bit between places | 7 | 3 | 4 |  | 0.36 0.00 0.01 0.01 0.01 0.01 |
| shared subtractor, choice held uniform for 400 steps, phase B only, from S4 weights, phase B hard_borrow=all, borrow a bit between places | 10 | 0 | 0 |  | 0.00 0.00 0.00 0.00 0.00 0.00 |
| shared subtractor, choice held uniform for 400 steps, phase B only, from S4 weights, phase B hard_borrow=all, borrow a bit between places | 10 | 1 | 0 |  | 1.00 1.00 1.00 1.00 1.00 1.00 |
| shared subtractor, choice held uniform for 400 steps, phase B only, from S4 weights, phase B hard_borrow=all, borrow a bit between places | 10 | 2 | 0 |  | 0.88 0.00 0.00 0.00 0.00 0.00 |
| shared subtractor, choice held uniform for 400 steps, phase B only, from S4 weights, phase B hard_borrow=all, borrow a bit between places | 10 | 3 | 0 |  | 0.04 0.00 0.00 0.00 0.00 0.00 |
| shared subtractor, choice held uniform for 400 steps, phase B only, from S4 weights, phase B hard_borrow=all, borrow a bit between places | 10 | 0 | 4 |  | 0.00 0.00 0.00 0.00 0.00 0.00 |
| shared subtractor, choice held uniform for 400 steps, phase B only, from S4 weights, phase B hard_borrow=all, borrow a bit between places | 10 | 1 | 4 |  | 0.38 0.62 0.00 0.00 0.00 0.00 |
| shared subtractor, choice held uniform for 400 steps, phase B only, from S4 weights, phase B hard_borrow=all, borrow a bit between places | 10 | 2 | 4 |  | 0.00 0.00 0.00 0.00 0.00 0.00 |
| shared subtractor, choice held uniform for 400 steps, phase B only, from S4 weights, phase B hard_borrow=all, borrow a bit between places | 10 | 3 | 4 |  | 0.00 0.00 0.00 0.00 0.00 0.00 |
| shared subtractor, not staged | 6 | 0 | 0 | k2: 4500/0.19 | 0.03 0.06 0.06 0.07 0.08 0.08 |
| shared subtractor, not staged | 6 | 1 | 0 | k2: 4500/0.05 | 0.05 0.04 0.04 0.05 0.05 0.06 |
| shared subtractor, not staged | 7 | 0 | 0 | k3: 4500/0.03 | 0.02 0.03 0.03 0.03 0.03 0.03 |
| shared subtractor, not staged | 7 | 1 | 0 | k3: 4500/0.03 | 0.03 0.03 0.02 0.03 0.03 0.03 |
| shared subtractor, not staged | 10 | 0 | 0 | k4: 4500/0.00 | 0.00 0.00 0.00 0.00 0.00 0.00 |
| shared subtractor, not staged | 10 | 1 | 0 | k4: 4500/0.00 | 0.00 0.00 0.00 0.01 0.00 0.00 |
| shared subtractor, own borrow from the start | 6 | 0 | 0 | k1: 700/1.00 k2: 0/1.00 | 1.00 1.00 1.00 1.00 1.00 1.00 |
| shared subtractor, own borrow from the start | 6 | 1 | 0 | k1: 1000/0.73 k2: 800/0.99 | 0.99 1.00 0.99 0.99 0.99 1.00 |
| shared subtractor, own borrow from the start | 7 | 0 | 0 | k1: 1000/0.83 k2: 1000/0.92 k3: 1500/0.98 | 0.98 1.00 0.89 0.02 0.84 0.95 |
| shared subtractor, own borrow from the start | 7 | 1 | 0 | k1: 1000/0.85 k2: 1000/0.98 k3: 1500/0.04 | 0.04 0.04 0.05 0.07 0.22 0.62 |
| shared subtractor, own borrow from the start | 10 | 0 | 0 | k1: 1000/0.27 k2: 1000/0.91 k3: 1000/0.00 k4: 1500/0.00 | 0.00 0.00 0.00 0.00 0.00 0.00 |
| shared subtractor, own borrow from the start | 10 | 1 | 0 | k1: 1000/0.43 k2: 1000/0.38 k3: 1000/0.48 k4: 1500/0.00 | 0.00 0.00 0.00 0.00 0.00 0.00 |
| shared subtractor, window 2/1 | 6 | 0 | 0 | k1: 1000/0.57 k2: 1500/0.80 | 0.03 0.93 1.00 1.00 1.00 1.00 |
| shared subtractor, window 2/1 | 6 | 1 | 0 | k1: 1000/0.56 k2: 1500/0.67 | 0.02 0.90 1.00 1.00 1.00 1.00 |
| shared subtractor, window 2/1 | 6 | 2 | 0 | k1: 1000/0.57 k2: 1500/0.73 | 0.02 0.98 1.00 1.00 1.00 1.00 |
| shared subtractor, window 2/1 | 7 | 0 | 0 | k1: 1000/0.53 k2: 1000/0.42 k3: 1500/0.73 | 0.02 0.02 0.03 0.03 0.05 0.27 |
| shared subtractor, window 2/1 | 7 | 1 | 0 | k1: 1000/0.51 k2: 1000/0.42 k3: 1500/0.73 | 0.02 0.04 0.44 0.55 0.78 0.79 |
| shared subtractor, window 2/1 | 7 | 2 | 0 | k1: 1000/0.51 k2: 1000/0.39 k3: 1500/0.72 | 0.02 0.02 0.02 0.04 0.10 0.34 |
| shared subtractor, window 2/1 | 10 | 0 | 0 | k1: 1000/0.33 k2: 1000/0.19 k3: 1000/0.20 k4: 1500/0.51 | 0.00 0.00 0.00 0.00 0.00 0.00 |
| shared subtractor, window 2/1 | 10 | 1 | 0 | k1: 1000/0.33 k2: 1000/0.17 k3: 1000/0.20 k4: 1500/0.58 | 0.00 0.00 0.00 0.00 0.00 0.00 |
| shared subtractor, window 2/1 | 10 | 2 | 0 | k1: 1000/0.02 k2: 1000/0.01 k3: 1000/0.00 k4: 1500/0.00 | 0.01 0.00 0.00 0.00 0.00 0.00 |

### rows

| variant | bits | seed | holdout | rows right | true-path rows: visited, right | over-subtracting trials' rows: borrow right | wrong rows (r, m, borrow in) |
|---|---|---|---|---|---|---|---|
| shared subtractor | 6 | 0 | 0 | 0.810 | 162, 1.000 | 1.000 | (0,0,1) (0,1,1) (0,2,1) (0,3,1) (1,1,1) (1,2,1) (1,3,1) (1,4,1) (2,2,1) (2,3,1) (2,4,1) (2,5,1) … |
| shared subtractor | 6 | 1 | 0 | 0.045 | 162, 0.056 | 0.515 | (0,0,0) (0,0,1) (0,1,0) (0,1,1) (0,2,0) (0,2,1) (0,3,0) (0,3,1) (0,4,1) (0,5,0) (0,5,1) (0,6,0) … |
| shared subtractor | 6 | 2 | 0 | 0.795 | 162, 0.982 | 1.000 | (0,0,1) (0,1,1) (0,2,1) (0,3,1) (1,1,1) (1,2,1) (1,3,1) (1,4,1) (2,2,1) (2,3,1) (2,4,1) (2,5,1) … |
| shared subtractor | 6 | 3 | 0 | 0.810 | 162, 1.000 | 1.000 | (0,0,1) (0,1,1) (0,2,1) (0,3,1) (1,1,1) (1,2,1) (1,3,1) (1,4,1) (2,2,1) (2,3,1) (2,4,1) (2,5,1) … |
| shared subtractor | 6 | 0 | 4 | 0.810 | 162, 1.000 | 1.000 | (0,0,1) (0,1,1) (0,2,1) (0,3,1) (1,1,1) (1,2,1) (1,3,1) (1,4,1) (2,2,1) (2,3,1) (2,4,1) (2,5,1) … |
| shared subtractor | 6 | 1 | 4 | 0.810 | 162, 1.000 | 0.995 | (0,0,1) (0,1,1) (0,2,1) (0,3,1) (1,1,1) (1,2,1) (1,3,1) (1,4,1) (2,2,1) (2,3,1) (2,4,1) (2,5,1) … |
| shared subtractor | 6 | 2 | 4 | 0.065 | 162, 0.080 | 0.540 | (0,0,0) (0,0,1) (0,1,0) (0,1,1) (0,2,0) (0,2,1) (0,3,0) (0,3,1) (0,4,0) (0,4,1) (0,5,0) (0,5,1) … |
| shared subtractor | 6 | 3 | 4 | 0.800 | 162, 0.988 | 1.000 | (0,0,1) (0,1,1) (0,2,1) (0,3,1) (1,1,1) (1,2,1) (1,3,1) (1,4,1) (2,2,1) (2,3,1) (2,4,1) (2,5,1) … |
| shared subtractor | 7 | 0 | 0 | 0.060 | 200, 0.060 | 0.500 | (0,0,0) (0,0,1) (0,1,0) (0,1,1) (0,2,0) (0,2,1) (0,3,0) (0,3,1) (0,4,0) (0,4,1) (0,5,0) (0,5,1) … |
| shared subtractor | 7 | 1 | 0 | 0.470 | 200, 0.470 | 0.715 | (0,0,1) (0,1,0) (0,1,1) (0,2,0) (0,2,1) (0,3,0) (0,3,1) (0,4,1) (0,5,1) (0,6,1) (0,7,1) (0,8,1) … |
| shared subtractor | 7 | 2 | 0 | 0.385 | 200, 0.385 | 0.640 | (0,0,1) (0,1,0) (0,1,1) (0,2,1) (0,3,0) (0,3,1) (0,4,0) (0,4,1) (0,5,1) (0,6,1) (0,7,1) (0,9,1) … |
| shared subtractor | 7 | 3 | 0 | 0.735 | 200, 0.735 | 0.890 | (0,0,1) (0,1,1) (0,3,1) (0,5,1) (0,6,1) (1,1,1) (1,6,1) (1,7,1) (2,2,1) (2,3,1) (2,4,1) (2,5,1) … |
| shared subtractor | 7 | 0 | 4 | 0.085 | 200, 0.085 | 0.570 | (0,0,1) (0,1,0) (0,1,1) (0,2,0) (0,2,1) (0,3,0) (0,3,1) (0,4,0) (0,4,1) (0,5,1) (0,6,0) (0,6,1) … |
| shared subtractor | 7 | 1 | 4 | 0.100 | 200, 0.100 | 0.610 | (0,0,0) (0,0,1) (0,1,0) (0,1,1) (0,2,0) (0,2,1) (0,3,0) (0,3,1) (0,4,0) (0,4,1) (0,5,0) (0,5,1) … |
| shared subtractor | 7 | 2 | 4 | 0.565 | 200, 0.565 | 0.860 | (0,0,0) (0,0,1) (0,1,1) (0,2,1) (0,3,1) (0,4,1) (0,5,1) (0,6,1) (0,7,1) (1,1,1) (1,2,1) (1,3,1) … |
| shared subtractor | 7 | 3 | 4 | 1.000 | 200, 1.000 | 1.000 | – |
| shared subtractor | 10 | 0 | 0 | 0.090 | 200, 0.090 | 0.605 | (0,0,1) (0,1,0) (0,1,1) (0,2,0) (0,2,1) (0,3,0) (0,3,1) (0,4,0) (0,4,1) (0,5,0) (0,5,1) (0,6,0) … |
| shared subtractor | 10 | 1 | 0 | 0.275 | 200, 0.275 | 0.550 | (0,0,1) (0,1,0) (0,1,1) (0,2,0) (0,2,1) (0,3,0) (0,3,1) (0,4,0) (0,4,1) (0,5,0) (0,5,1) (0,6,0) … |
| shared subtractor | 10 | 2 | 0 | 1.000 | 200, 1.000 | 1.000 | – |
| shared subtractor | 10 | 3 | 0 | 1.000 | 200, 1.000 | 1.000 | – |
| shared subtractor | 10 | 0 | 4 | 0.080 | 200, 0.080 | 0.615 | (0,0,1) (0,1,0) (0,1,1) (0,2,0) (0,2,1) (0,3,0) (0,3,1) (0,4,0) (0,4,1) (0,5,1) (0,6,0) (0,6,1) … |
| shared subtractor | 10 | 1 | 4 | 0.105 | 200, 0.105 | 0.570 | (0,0,0) (0,0,1) (0,1,0) (0,1,1) (0,2,0) (0,2,1) (0,3,0) (0,3,1) (0,4,0) (0,4,1) (0,5,0) (0,5,1) … |
| shared subtractor | 10 | 2 | 4 | 1.000 | 200, 1.000 | 1.000 | – |
| shared subtractor | 10 | 3 | 4 | 0.405 | 200, 0.405 | 0.710 | (0,0,1) (0,1,0) (0,1,1) (0,2,1) (0,3,1) (0,4,1) (0,5,1) (0,6,0) (0,6,1) (0,7,1) (0,8,1) (1,0,1) … |
| shared subtractor, always from the top w digits | 6 | 0 | 0 | 0.215 | 162, 0.265 | 0.485 | (0,0,0) (0,0,1) (0,1,1) (0,2,1) (0,3,1) (0,4,1) (0,6,0) (0,6,1) (0,7,0) (0,7,1) (0,8,1) (0,9,1) … |
| shared subtractor, always from the top w digits | 6 | 1 | 0 | 0.050 | 162, 0.062 | 0.420 | (0,0,0) (0,0,1) (0,1,0) (0,1,1) (0,2,0) (0,2,1) (0,3,0) (0,3,1) (0,4,0) (0,4,1) (0,5,0) (0,5,1) … |
| shared subtractor, always from the top w digits | 7 | 0 | 0 | 0.095 | 200, 0.095 | 0.525 | (0,0,0) (0,0,1) (0,1,0) (0,1,1) (0,2,0) (0,2,1) (0,3,0) (0,3,1) (0,4,0) (0,4,1) (0,5,0) (0,5,1) … |
| shared subtractor, always from the top w digits | 7 | 1 | 0 | 0.445 | 200, 0.445 | 0.715 | (0,0,1) (0,1,0) (0,1,1) (0,2,0) (0,2,1) (0,3,0) (0,3,1) (0,4,1) (0,5,1) (0,6,0) (0,6,1) (0,7,1) … |
| shared subtractor, always from the top w digits | 10 | 0 | 0 | 0.080 | 200, 0.080 | 0.550 | (0,0,1) (0,1,0) (0,1,1) (0,2,0) (0,2,1) (0,3,0) (0,3,1) (0,4,0) (0,4,1) (0,5,0) (0,5,1) (0,6,0) … |
| shared subtractor, always from the top w digits | 10 | 1 | 0 | 0.080 | 200, 0.080 | 0.550 | (0,0,0) (0,0,1) (0,1,0) (0,1,1) (0,2,0) (0,2,1) (0,3,0) (0,3,1) (0,4,0) (0,4,1) (0,5,0) (0,5,1) … |
| shared subtractor, choice held uniform for 400 steps | 6 | 0 | 0 | 0.810 | 162, 1.000 | 1.000 | (0,0,1) (0,1,1) (0,2,1) (0,3,1) (1,1,1) (1,2,1) (1,3,1) (1,4,1) (2,2,1) (2,3,1) (2,4,1) (2,5,1) … |
| shared subtractor, choice held uniform for 400 steps | 6 | 1 | 0 | 0.805 | 162, 0.994 | 1.000 | (0,0,1) (0,1,1) (0,2,1) (0,3,1) (1,1,1) (1,2,1) (1,3,1) (1,4,1) (2,2,1) (2,3,1) (2,4,1) (2,5,1) … |
| shared subtractor, choice held uniform for 400 steps | 6 | 2 | 0 | 0.805 | 162, 0.994 | 1.000 | (0,0,1) (0,1,1) (0,2,1) (0,3,1) (1,1,1) (1,2,1) (1,3,1) (1,4,1) (2,2,1) (2,3,1) (2,4,1) (2,5,1) … |
| shared subtractor, choice held uniform for 400 steps | 6 | 3 | 0 | 0.810 | 162, 1.000 | 1.000 | (0,0,1) (0,1,1) (0,2,1) (0,3,1) (1,1,1) (1,2,1) (1,3,1) (1,4,1) (2,2,1) (2,3,1) (2,4,1) (2,5,1) … |
| shared subtractor, choice held uniform for 400 steps | 6 | 0 | 4 | 0.810 | 162, 0.994 | 1.000 | (0,0,1) (0,1,1) (0,2,1) (0,3,1) (1,1,1) (1,2,1) (1,3,1) (1,4,1) (2,2,1) (2,3,1) (2,4,1) (2,5,1) … |
| shared subtractor, choice held uniform for 400 steps | 6 | 1 | 4 | 0.810 | 162, 0.994 | 1.000 | (0,0,1) (0,1,1) (0,2,1) (0,3,1) (1,1,1) (1,2,1) (1,3,1) (1,4,1) (2,2,1) (2,3,1) (2,4,1) (2,5,1) … |
| shared subtractor, choice held uniform for 400 steps | 6 | 2 | 4 | 0.815 | 162, 1.000 | 1.000 | (0,0,1) (0,1,1) (0,2,1) (0,3,1) (1,1,1) (1,2,1) (1,3,1) (1,4,1) (2,2,1) (2,3,1) (2,4,1) (3,3,1) … |
| shared subtractor, choice held uniform for 400 steps | 6 | 3 | 4 | 0.805 | 162, 0.994 | 1.000 | (0,0,1) (0,1,1) (0,2,1) (0,3,1) (1,1,1) (1,2,1) (1,3,1) (1,4,1) (2,2,1) (2,3,1) (2,4,1) (2,5,1) … |
| shared subtractor, choice held uniform for 400 steps | 7 | 0 | 0 | 0.470 | 200, 0.470 | 0.740 | (0,0,1) (0,1,1) (0,2,1) (0,3,1) (0,4,1) (0,5,1) (0,6,1) (0,7,1) (0,8,1) (1,0,1) (1,1,0) (1,1,1) … |
| shared subtractor, choice held uniform for 400 steps | 7 | 0 | 0 | 0.570 | 200, 0.570 | 0.790 | (0,0,1) (0,1,1) (0,2,1) (0,3,1) (0,4,1) (0,5,1) (0,7,1) (0,8,1) (1,1,1) (1,2,1) (1,3,1) (1,4,1) … |
| shared subtractor, choice held uniform for 400 steps | 7 | 1 | 0 | 0.410 | 200, 0.410 | 0.685 | (0,0,1) (0,1,1) (0,2,0) (0,2,1) (0,3,0) (0,3,1) (0,4,0) (0,4,1) (0,5,1) (0,6,0) (0,6,1) (0,7,0) … |
| shared subtractor, choice held uniform for 400 steps | 7 | 1 | 0 | 0.410 | 200, 0.410 | 0.685 | (0,0,1) (0,1,1) (0,2,0) (0,2,1) (0,3,0) (0,3,1) (0,4,0) (0,4,1) (0,5,1) (0,6,0) (0,6,1) (0,7,0) … |
| shared subtractor, choice held uniform for 400 steps | 7 | 2 | 0 | 0.630 | 200, 0.630 | 0.800 | (0,0,1) (0,1,1) (0,2,1) (0,3,1) (0,4,1) (0,5,1) (0,6,1) (0,9,1) (1,1,1) (1,2,1) (1,3,1) (1,4,1) … |
| shared subtractor, choice held uniform for 400 steps | 7 | 2 | 0 | 0.630 | 200, 0.630 | 0.800 | (0,0,1) (0,1,1) (0,2,1) (0,3,1) (0,4,1) (0,5,1) (0,6,1) (0,9,1) (1,1,1) (1,2,1) (1,3,1) (1,4,1) … |
| shared subtractor, choice held uniform for 400 steps | 7 | 3 | 0 | 0.975 | 200, 0.975 | 0.995 | (2,0,1) (3,0,1) (3,3,1) (4,4,1) (6,6,1) |
| shared subtractor, choice held uniform for 400 steps | 7 | 3 | 0 | 0.975 | 200, 0.975 | 0.995 | (2,0,1) (3,0,1) (3,3,1) (4,4,1) (6,6,1) |
| shared subtractor, choice held uniform for 400 steps | 7 | 0 | 4 | 0.520 | 200, 0.520 | 0.805 | (0,0,1) (0,1,0) (0,1,1) (0,2,1) (0,3,0) (0,3,1) (0,4,1) (0,5,0) (0,5,1) (0,6,0) (0,6,1) (0,7,1) … |
| shared subtractor, choice held uniform for 400 steps | 7 | 1 | 4 | 0.390 | 200, 0.390 | 0.640 | (0,0,0) (0,0,1) (0,1,1) (0,2,1) (0,5,1) (0,6,1) (0,7,1) (0,9,1) (1,0,0) (1,0,1) (1,1,0) (1,1,1) … |
| shared subtractor, choice held uniform for 400 steps | 7 | 2 | 4 | 0.550 | 200, 0.550 | 0.790 | (0,0,1) (0,1,1) (0,2,1) (0,3,1) (0,4,1) (0,5,1) (0,6,1) (0,7,1) (0,8,1) (1,1,1) (1,2,1) (1,3,1) … |
| shared subtractor, choice held uniform for 400 steps | 7 | 3 | 4 | 0.560 | 200, 0.560 | 0.830 | (0,0,1) (0,1,1) (0,2,1) (0,3,1) (0,4,1) (0,5,1) (0,6,1) (0,7,1) (0,8,1) (1,1,1) (1,2,1) (1,3,1) … |
| shared subtractor, choice held uniform for 400 steps | 10 | 0 | 0 | 1.000 | 200, 1.000 | 1.000 | – |
| shared subtractor, choice held uniform for 400 steps | 10 | 1 | 0 | 1.000 | 200, 1.000 | 1.000 | – |
| shared subtractor, choice held uniform for 400 steps | 10 | 2 | 0 | 1.000 | 200, 1.000 | 1.000 | – |
| shared subtractor, choice held uniform for 400 steps | 10 | 3 | 0 | 1.000 | 200, 1.000 | 1.000 | – |
| shared subtractor, choice held uniform for 400 steps | 10 | 0 | 4 | 1.000 | 200, 1.000 | 1.000 | – |
| shared subtractor, choice held uniform for 400 steps | 10 | 1 | 4 | 1.000 | 200, 1.000 | 1.000 | – |
| shared subtractor, choice held uniform for 400 steps | 10 | 2 | 4 | 1.000 | 200, 1.000 | 1.000 | – |
| shared subtractor, choice held uniform for 400 steps | 10 | 3 | 4 | 1.000 | 200, 1.000 | 1.000 | – |
| shared subtractor, choice held uniform for 400 steps, borrow a bit between places | 6 | 0 | 0 | 0.805 | 162, 0.994 | 1.000 | (0,0,1) (0,1,1) (0,2,1) (0,3,1) (1,1,1) (1,2,1) (1,3,1) (1,4,1) (2,2,1) (2,3,1) (2,4,1) (2,5,1) … |
| shared subtractor, choice held uniform for 400 steps, borrow a bit between places | 6 | 1 | 0 | 0.810 | 162, 1.000 | 1.000 | (0,0,1) (0,1,1) (0,2,1) (0,3,1) (1,1,1) (1,2,1) (1,3,1) (1,4,1) (2,2,1) (2,3,1) (2,4,1) (2,5,1) … |
| shared subtractor, choice held uniform for 400 steps, borrow a bit between places | 6 | 2 | 0 | 0.805 | 162, 0.994 | 0.995 | (0,0,1) (0,1,1) (0,2,1) (0,3,1) (1,1,1) (1,2,1) (1,3,1) (1,4,1) (2,2,1) (2,3,1) (2,4,1) (2,5,1) … |
| shared subtractor, choice held uniform for 400 steps, borrow a bit between places | 6 | 3 | 0 | 0.810 | 162, 1.000 | 1.000 | (0,0,1) (0,1,1) (0,2,1) (0,3,1) (1,1,1) (1,2,1) (1,3,1) (1,4,1) (2,2,1) (2,3,1) (2,4,1) (2,5,1) … |
| shared subtractor, choice held uniform for 400 steps, borrow a bit between places | 6 | 0 | 4 | 0.075 | 162, 0.080 | 0.265 | (0,0,0) (0,0,1) (0,1,0) (0,1,1) (0,2,0) (0,3,1) (0,4,1) (0,5,0) (0,5,1) (0,6,0) (0,6,1) (0,7,0) … |
| shared subtractor, choice held uniform for 400 steps, borrow a bit between places | 6 | 1 | 4 | 0.805 | 162, 0.994 | 1.000 | (0,0,1) (0,1,1) (0,2,1) (0,3,1) (1,1,1) (1,2,1) (1,3,1) (1,4,1) (2,2,1) (2,3,1) (2,4,1) (2,5,1) … |
| shared subtractor, choice held uniform for 400 steps, borrow a bit between places | 6 | 2 | 4 | 0.810 | 162, 1.000 | 1.000 | (0,0,1) (0,1,1) (0,2,1) (0,3,1) (1,1,1) (1,2,1) (1,3,1) (1,4,1) (2,2,1) (2,3,1) (2,4,1) (2,5,1) … |
| shared subtractor, choice held uniform for 400 steps, borrow a bit between places | 6 | 3 | 4 | 0.810 | 162, 1.000 | 1.000 | (0,0,1) (0,1,1) (0,2,1) (0,3,1) (1,1,1) (1,2,1) (1,3,1) (1,4,1) (2,2,1) (2,3,1) (2,4,1) (2,5,1) … |
| shared subtractor, choice held uniform for 400 steps, borrow a bit between places | 7 | 0 | 0 | 0.400 | 200, 0.400 | 0.625 | (0,0,0) (0,0,1) (0,1,1) (0,2,1) (0,3,1) (0,4,1) (0,5,1) (0,6,1) (0,7,1) (1,0,0) (1,0,1) (1,1,0) … |
| shared subtractor, choice held uniform for 400 steps, borrow a bit between places | 7 | 1 | 0 | 1.000 | 200, 1.000 | 1.000 | – |
| shared subtractor, choice held uniform for 400 steps, borrow a bit between places | 7 | 2 | 0 | 0.560 | 200, 0.560 | 0.725 | (0,0,0) (0,0,1) (0,1,1) (0,2,1) (0,3,1) (0,4,1) (0,5,1) (0,6,1) (0,7,1) (0,9,1) (1,0,1) (1,1,1) … |
| shared subtractor, choice held uniform for 400 steps, borrow a bit between places | 7 | 3 | 0 | 0.405 | 200, 0.405 | 0.555 | (0,0,0) (0,0,1) (0,1,1) (0,2,0) (0,2,1) (0,3,0) (0,3,1) (0,4,0) (0,4,1) (0,5,1) (0,6,0) (0,6,1) … |
| shared subtractor, choice held uniform for 400 steps, borrow a bit between places | 7 | 0 | 4 | 0.575 | 200, 0.575 | 0.740 | (0,0,0) (0,0,1) (0,1,1) (0,2,1) (0,3,1) (0,4,1) (0,5,1) (0,6,1) (0,7,1) (0,9,0) (0,9,1) (1,1,1) … |
| shared subtractor, choice held uniform for 400 steps, borrow a bit between places | 7 | 1 | 4 | 1.000 | 200, 1.000 | 1.000 | – |
| shared subtractor, choice held uniform for 400 steps, borrow a bit between places | 7 | 2 | 4 | 0.535 | 200, 0.535 | 0.640 | (0,0,0) (0,0,1) (0,2,0) (0,2,1) (0,3,0) (0,3,1) (0,4,0) (0,4,1) (0,5,1) (0,7,1) (0,8,0) (0,8,1) … |
| shared subtractor, choice held uniform for 400 steps, borrow a bit between places | 7 | 3 | 4 | 0.535 | 200, 0.535 | 0.705 | (0,0,0) (0,0,1) (0,1,1) (0,2,1) (0,3,0) (0,3,1) (0,5,1) (0,7,0) (0,7,1) (0,8,0) (0,8,1) (0,9,0) … |
| shared subtractor, choice held uniform for 400 steps, borrow a bit between places | 10 | 0 | 0 | 0.395 | 200, 0.395 | 0.555 | (0,0,0) (0,0,1) (0,1,0) (0,1,1) (0,2,0) (0,2,1) (0,3,0) (0,3,1) (0,4,0) (0,4,1) (0,5,1) (0,6,0) … |
| shared subtractor, choice held uniform for 400 steps, borrow a bit between places | 10 | 1 | 0 | 0.505 | 200, 0.505 | 0.690 | (0,0,0) (0,0,1) (0,1,1) (0,2,0) (0,3,0) (0,3,1) (0,4,0) (0,4,1) (0,5,1) (0,7,1) (0,8,0) (0,8,1) … |
| shared subtractor, choice held uniform for 400 steps, borrow a bit between places | 10 | 2 | 0 | 0.500 | 200, 0.500 | 0.715 | (0,0,0) (0,0,1) (0,1,1) (0,2,1) (0,3,1) (0,4,1) (0,5,1) (0,6,1) (0,7,1) (0,8,1) (1,1,0) (1,1,1) … |
| shared subtractor, choice held uniform for 400 steps, borrow a bit between places | 10 | 3 | 0 | 0.470 | 200, 0.470 | 0.560 | (0,0,0) (0,0,1) (0,1,0) (0,1,1) (0,2,0) (0,2,1) (0,3,0) (0,3,1) (0,4,0) (0,4,1) (0,5,1) (0,6,0) … |
| shared subtractor, choice held uniform for 400 steps, borrow a bit between places | 10 | 0 | 4 | 0.235 | 200, 0.235 | 0.660 | (0,0,0) (0,0,1) (0,1,0) (0,1,1) (0,2,0) (0,2,1) (0,3,0) (0,3,1) (0,4,0) (0,4,1) (0,5,0) (0,5,1) … |
| shared subtractor, choice held uniform for 400 steps, borrow a bit between places | 10 | 1 | 4 | 0.375 | 200, 0.375 | 0.510 | (0,0,0) (0,0,1) (0,1,1) (0,2,1) (0,3,0) (0,3,1) (0,4,0) (0,4,1) (0,5,1) (0,6,0) (0,6,1) (0,7,0) … |
| shared subtractor, choice held uniform for 400 steps, borrow a bit between places | 10 | 2 | 4 | 0.435 | 200, 0.435 | 0.685 | (0,0,0) (0,0,1) (0,1,0) (0,1,1) (0,2,1) (0,3,0) (0,3,1) (0,4,0) (0,4,1) (0,5,1) (0,6,1) (0,7,1) … |
| shared subtractor, choice held uniform for 400 steps, borrow a bit between places | 10 | 3 | 4 | 0.430 | 200, 0.430 | 0.590 | (0,0,0) (0,0,1) (0,1,0) (0,1,1) (0,2,0) (0,2,1) (0,3,0) (0,3,1) (0,4,0) (0,4,1) (0,5,1) (0,6,0) … |
| shared subtractor, choice held uniform for 400 steps, handover over 500 steps | 6 | 0 | 0 | 0.810 | 162, 1.000 | 1.000 | (0,0,1) (0,1,1) (0,2,1) (0,3,1) (1,1,1) (1,2,1) (1,3,1) (1,4,1) (2,2,1) (2,3,1) (2,4,1) (2,5,1) … |
| shared subtractor, choice held uniform for 400 steps, handover over 500 steps | 6 | 1 | 0 | 0.800 | 162, 0.988 | 0.990 | (0,0,0) (0,0,1) (0,1,1) (0,2,1) (0,3,1) (1,1,1) (1,2,1) (1,3,1) (1,4,1) (2,2,1) (2,3,1) (2,4,1) … |
| shared subtractor, choice held uniform for 400 steps, handover over 500 steps | 6 | 2 | 0 | 0.795 | 162, 0.982 | 1.000 | (0,0,1) (0,1,1) (0,2,1) (0,3,1) (1,1,1) (1,2,1) (1,3,1) (1,4,1) (2,2,1) (2,3,1) (2,4,1) (2,5,1) … |
| shared subtractor, choice held uniform for 400 steps, handover over 500 steps | 6 | 3 | 0 | 0.810 | 162, 1.000 | 1.000 | (0,0,1) (0,1,1) (0,2,1) (0,3,1) (1,1,1) (1,2,1) (1,3,1) (1,4,1) (2,2,1) (2,3,1) (2,4,1) (2,5,1) … |
| shared subtractor, choice held uniform for 400 steps, handover over 500 steps | 6 | 0 | 4 | 0.815 | 162, 1.000 | 1.000 | (0,0,1) (0,1,1) (0,2,1) (0,3,1) (1,1,1) (1,2,1) (1,3,1) (1,4,1) (2,2,1) (2,3,1) (2,4,1) (2,5,1) … |
| shared subtractor, choice held uniform for 400 steps, handover over 500 steps | 6 | 1 | 4 | 0.810 | 162, 1.000 | 1.000 | (0,0,1) (0,1,1) (0,2,1) (0,3,1) (1,1,1) (1,2,1) (1,3,1) (1,4,1) (2,2,1) (2,3,1) (2,4,1) (2,5,1) … |
| shared subtractor, choice held uniform for 400 steps, handover over 500 steps | 6 | 2 | 4 | 0.815 | 162, 1.000 | 1.000 | (0,0,1) (0,1,1) (0,2,1) (0,3,1) (1,1,1) (1,2,1) (1,3,1) (1,4,1) (2,2,1) (2,3,1) (2,4,1) (3,3,1) … |
| shared subtractor, choice held uniform for 400 steps, handover over 500 steps | 6 | 3 | 4 | 0.805 | 162, 0.988 | 1.000 | (0,0,1) (0,1,1) (0,2,1) (0,3,1) (1,1,1) (1,2,1) (1,3,1) (1,4,1) (2,2,1) (2,3,1) (2,4,1) (2,5,1) … |
| shared subtractor, choice held uniform for 400 steps, handover over 500 steps | 7 | 0 | 0 | 0.585 | 200, 0.585 | 0.795 | (0,0,1) (0,1,1) (0,2,1) (0,3,1) (0,4,1) (0,6,1) (0,7,1) (1,1,1) (1,2,1) (1,3,1) (1,4,1) (1,5,1) … |
| shared subtractor, choice held uniform for 400 steps, handover over 500 steps | 7 | 1 | 0 | 0.520 | 200, 0.520 | 0.770 | (0,0,1) (0,1,1) (0,2,1) (0,3,1) (0,4,1) (0,5,1) (0,6,1) (0,7,1) (0,9,1) (1,0,0) (1,0,1) (1,1,1) … |
| shared subtractor, choice held uniform for 400 steps, handover over 500 steps | 7 | 2 | 0 | 0.390 | 200, 0.390 | 0.645 | (0,0,1) (0,1,1) (0,2,1) (0,3,1) (0,4,1) (0,5,1) (0,6,1) (0,7,1) (0,8,1) (1,0,1) (1,1,0) (1,1,1) … |
| shared subtractor, choice held uniform for 400 steps, handover over 500 steps | 7 | 3 | 0 | 0.705 | 200, 0.705 | 0.875 | (0,0,1) (0,1,1) (0,3,1) (0,4,1) (0,5,1) (1,1,1) (1,3,1) (1,6,1) (1,8,1) (2,0,1) (2,1,1) (2,2,1) … |
| shared subtractor, choice held uniform for 400 steps, handover over 500 steps | 7 | 0 | 4 | 0.630 | 200, 0.630 | 0.770 | (0,0,1) (0,1,1) (0,3,1) (0,6,1) (0,8,1) (0,9,1) (1,1,1) (1,3,1) (1,4,1) (1,5,1) (1,6,1) (1,7,1) … |
| shared subtractor, choice held uniform for 400 steps, handover over 500 steps | 7 | 1 | 4 | 0.150 | 200, 0.150 | 0.570 | (0,0,0) (0,0,1) (0,1,0) (0,1,1) (0,2,0) (0,2,1) (0,3,0) (0,3,1) (0,4,0) (0,4,1) (0,5,0) (0,5,1) … |
| shared subtractor, choice held uniform for 400 steps, handover over 500 steps | 7 | 2 | 4 | 0.560 | 200, 0.560 | 0.790 | (0,0,1) (0,1,1) (0,2,1) (0,3,1) (0,4,1) (0,5,1) (0,6,1) (0,7,1) (0,8,1) (1,1,1) (1,2,1) (1,3,1) … |
| shared subtractor, choice held uniform for 400 steps, handover over 500 steps | 7 | 3 | 4 | 0.515 | 200, 0.515 | 0.735 | (0,0,1) (0,1,1) (0,2,1) (0,3,1) (0,4,1) (0,5,1) (0,7,1) (0,8,1) (1,1,1) (1,2,1) (1,3,1) (1,4,1) … |
| shared subtractor, choice held uniform for 400 steps, handover over 500 steps | 10 | 0 | 0 | 1.000 | 200, 1.000 | 1.000 | – |
| shared subtractor, choice held uniform for 400 steps, handover over 500 steps | 10 | 1 | 0 | 1.000 | 200, 1.000 | 1.000 | – |
| shared subtractor, choice held uniform for 400 steps, handover over 500 steps | 10 | 2 | 0 | 1.000 | 200, 1.000 | 1.000 | – |
| shared subtractor, choice held uniform for 400 steps, handover over 500 steps | 10 | 3 | 0 | 1.000 | 200, 1.000 | 1.000 | – |
| shared subtractor, choice held uniform for 400 steps, handover over 500 steps | 10 | 0 | 4 | 1.000 | 200, 1.000 | 1.000 | – |
| shared subtractor, choice held uniform for 400 steps, handover over 500 steps | 10 | 1 | 4 | 1.000 | 200, 1.000 | 1.000 | – |
| shared subtractor, choice held uniform for 400 steps, handover over 500 steps | 10 | 2 | 4 | 1.000 | 200, 1.000 | 1.000 | – |
| shared subtractor, choice held uniform for 400 steps, handover over 500 steps | 10 | 3 | 4 | 1.000 | 200, 1.000 | 1.000 | – |
| shared subtractor, choice held uniform for 400 steps, lr 0.0003 after the switch | 7 | 0 | 0 | 0.475 | 200, 0.475 | 0.735 | (0,0,1) (0,1,1) (0,2,1) (0,3,1) (0,4,1) (0,5,1) (0,6,1) (0,7,1) (0,8,1) (1,0,1) (1,1,0) (1,1,1) … |
| shared subtractor, choice held uniform for 400 steps, lr 0.0003 after the switch | 7 | 1 | 0 | 0.300 | 200, 0.300 | 0.570 | (0,0,1) (0,1,1) (0,2,0) (0,2,1) (0,3,0) (0,3,1) (0,4,0) (0,4,1) (0,5,0) (0,5,1) (0,6,0) (0,6,1) … |
| shared subtractor, choice held uniform for 400 steps, lr 0.0003 after the switch | 7 | 2 | 0 | 0.545 | 200, 0.545 | 0.765 | (0,0,1) (0,1,1) (0,2,1) (0,3,1) (0,4,1) (0,5,1) (0,6,1) (0,7,1) (0,8,1) (1,0,1) (1,1,1) (1,2,1) … |
| shared subtractor, choice held uniform for 400 steps, lr 0.0003 after the switch | 7 | 3 | 0 | 0.645 | 200, 0.645 | 0.945 | (0,0,1) (0,1,1) (0,2,1) (0,3,1) (0,4,1) (0,5,1) (0,6,1) (0,7,1) (1,1,1) (1,2,1) (1,3,1) (1,5,1) … |
| shared subtractor, choice held uniform for 400 steps, lr 0.001 after the switch | 7 | 0 | 0 | 0.490 | 200, 0.490 | 0.740 | (0,0,0) (0,0,1) (0,1,1) (0,2,1) (0,3,1) (0,4,1) (0,5,1) (0,6,1) (0,7,1) (0,8,1) (1,0,1) (1,1,1) … |
| shared subtractor, choice held uniform for 400 steps, lr 0.001 after the switch | 7 | 1 | 0 | 0.545 | 200, 0.545 | 0.775 | (0,0,1) (0,1,1) (0,2,1) (0,3,1) (0,4,1) (0,5,1) (0,6,1) (0,7,1) (1,1,1) (1,2,1) (1,3,1) (1,4,1) … |
| shared subtractor, choice held uniform for 400 steps, lr 0.001 after the switch | 7 | 2 | 0 | 0.545 | 200, 0.545 | 0.805 | (0,0,1) (0,1,1) (0,2,1) (0,3,1) (0,4,1) (0,5,1) (0,6,1) (0,7,1) (0,8,1) (0,9,1) (1,1,1) (1,2,1) … |
| shared subtractor, choice held uniform for 400 steps, lr 0.001 after the switch | 7 | 3 | 0 | 0.625 | 200, 0.625 | 0.860 | (0,0,1) (0,1,1) (0,2,1) (0,3,1) (0,4,1) (0,5,1) (0,6,1) (0,7,1) (1,1,1) (1,2,1) (1,3,1) (1,4,1) … |
| shared subtractor, choice held uniform for 400 steps, phase B only, from S4 weights, phase B hard_borrow=all, borrow a bit between places | 6 | 0 | 0 | 0.070 | 162, 0.086 | 0.545 | (0,0,0) (0,0,1) (0,1,0) (0,1,1) (0,2,0) (0,2,1) (0,3,0) (0,3,1) (0,4,0) (0,4,1) (0,5,0) (0,5,1) … |
| shared subtractor, choice held uniform for 400 steps, phase B only, from S4 weights, phase B hard_borrow=all, borrow a bit between places | 6 | 1 | 0 | 0.040 | 162, 0.049 | 0.415 | (0,0,0) (0,0,1) (0,1,0) (0,1,1) (0,2,0) (0,2,1) (0,3,0) (0,3,1) (0,4,0) (0,4,1) (0,5,0) (0,5,1) … |
| shared subtractor, choice held uniform for 400 steps, phase B only, from S4 weights, phase B hard_borrow=all, borrow a bit between places | 6 | 2 | 0 | 0.075 | 162, 0.093 | 0.470 | (0,0,1) (0,1,0) (0,1,1) (0,2,0) (0,2,1) (0,3,0) (0,3,1) (0,4,0) (0,4,1) (0,5,0) (0,5,1) (0,6,0) … |
| shared subtractor, choice held uniform for 400 steps, phase B only, from S4 weights, phase B hard_borrow=all, borrow a bit between places | 6 | 3 | 0 | 0.160 | 162, 0.198 | 0.500 | (0,0,0) (0,0,1) (0,1,0) (0,1,1) (0,2,0) (0,2,1) (0,3,0) (0,3,1) (0,4,0) (0,4,1) (0,6,1) (0,7,0) … |
| shared subtractor, choice held uniform for 400 steps, phase B only, from S4 weights, phase B hard_borrow=all, borrow a bit between places | 6 | 0 | 4 | 0.090 | 162, 0.105 | 0.545 | (0,0,1) (0,1,0) (0,1,1) (0,2,0) (0,3,0) (0,3,1) (0,4,0) (0,4,1) (0,5,0) (0,5,1) (0,6,0) (0,6,1) … |
| shared subtractor, choice held uniform for 400 steps, phase B only, from S4 weights, phase B hard_borrow=all, borrow a bit between places | 6 | 1 | 4 | 0.055 | 162, 0.068 | 0.460 | (0,0,0) (0,0,1) (0,1,0) (0,1,1) (0,2,0) (0,2,1) (0,3,0) (0,3,1) (0,4,0) (0,4,1) (0,5,0) (0,5,1) … |
| shared subtractor, choice held uniform for 400 steps, phase B only, from S4 weights, phase B hard_borrow=all, borrow a bit between places | 6 | 2 | 4 | 0.030 | 162, 0.037 | 0.480 | (0,0,0) (0,0,1) (0,1,0) (0,1,1) (0,2,0) (0,2,1) (0,3,0) (0,3,1) (0,4,0) (0,4,1) (0,5,0) (0,5,1) … |
| shared subtractor, choice held uniform for 400 steps, phase B only, from S4 weights, phase B hard_borrow=all, borrow a bit between places | 6 | 3 | 4 | 0.060 | 162, 0.074 | 0.515 | (0,0,0) (0,0,1) (0,1,0) (0,1,1) (0,2,0) (0,2,1) (0,3,0) (0,3,1) (0,4,0) (0,4,1) (0,5,0) (0,5,1) … |
| shared subtractor, choice held uniform for 400 steps, phase B only, from S4 weights, phase B hard_borrow=all, borrow a bit between places | 7 | 0 | 0 | 1.000 | 200, 1.000 | 1.000 | – |
| shared subtractor, choice held uniform for 400 steps, phase B only, from S4 weights, phase B hard_borrow=all, borrow a bit between places | 7 | 1 | 0 | 1.000 | 200, 1.000 | 1.000 | – |
| shared subtractor, choice held uniform for 400 steps, phase B only, from S4 weights, phase B hard_borrow=all, borrow a bit between places | 7 | 2 | 0 | 0.430 | 200, 0.430 | 0.715 | (0,0,0) (0,0,1) (0,1,0) (0,1,1) (0,2,0) (0,2,1) (0,5,0) (0,8,1) (0,9,1) (1,0,0) (1,0,1) (1,1,0) … |
| shared subtractor, choice held uniform for 400 steps, phase B only, from S4 weights, phase B hard_borrow=all, borrow a bit between places | 7 | 3 | 0 | 0.690 | 200, 0.690 | 0.745 | (0,0,0) (0,0,1) (0,8,1) (1,0,0) (1,0,1) (1,1,1) (2,0,0) (2,0,1) (2,1,0) (2,1,1) (2,2,1) (3,0,1) … |
| shared subtractor, choice held uniform for 400 steps, phase B only, from S4 weights, phase B hard_borrow=all, borrow a bit between places | 7 | 0 | 4 | 0.215 | 200, 0.215 | 0.735 | (0,0,0) (0,0,1) (0,1,0) (0,1,1) (0,2,0) (0,2,1) (0,3,0) (0,3,1) (0,4,0) (0,4,1) (0,5,0) (0,5,1) … |
| shared subtractor, choice held uniform for 400 steps, phase B only, from S4 weights, phase B hard_borrow=all, borrow a bit between places | 7 | 1 | 4 | 0.730 | 200, 0.730 | 0.795 | (0,0,0) (0,0,1) (0,1,0) (1,0,0) (1,0,1) (1,1,0) (1,1,1) (2,0,1) (2,2,1) (3,0,1) (3,1,1) (3,2,1) … |
| shared subtractor, choice held uniform for 400 steps, phase B only, from S4 weights, phase B hard_borrow=all, borrow a bit between places | 7 | 2 | 4 | 0.400 | 200, 0.400 | 0.585 | (0,0,0) (0,0,1) (0,3,0) (0,4,1) (0,6,1) (0,7,1) (0,8,0) (0,9,1) (1,0,0) (1,0,1) (1,1,0) (1,1,1) … |
| shared subtractor, choice held uniform for 400 steps, phase B only, from S4 weights, phase B hard_borrow=all, borrow a bit between places | 7 | 3 | 4 | 0.585 | 200, 0.585 | 0.700 | (0,0,0) (0,0,1) (0,1,0) (0,1,1) (0,2,1) (0,3,0) (0,4,0) (0,5,1) (0,6,0) (0,7,1) (0,8,0) (0,8,1) … |
| shared subtractor, choice held uniform for 400 steps, phase B only, from S4 weights, phase B hard_borrow=all, borrow a bit between places | 10 | 0 | 0 | 0.385 | 200, 0.385 | 0.625 | (0,0,0) (0,1,1) (0,2,0) (0,3,1) (0,6,1) (0,7,1) (0,8,0) (0,8,1) (0,9,1) (1,0,1) (1,1,0) (1,1,1) … |
| shared subtractor, choice held uniform for 400 steps, phase B only, from S4 weights, phase B hard_borrow=all, borrow a bit between places | 10 | 1 | 0 | 1.000 | 200, 1.000 | 1.000 | – |
| shared subtractor, choice held uniform for 400 steps, phase B only, from S4 weights, phase B hard_borrow=all, borrow a bit between places | 10 | 2 | 0 | 0.235 | 200, 0.235 | 0.665 | (0,0,0) (0,0,1) (0,1,0) (0,2,1) (0,3,0) (0,3,1) (0,4,0) (0,4,1) (0,5,0) (0,5,1) (0,6,0) (0,7,0) … |
| shared subtractor, choice held uniform for 400 steps, phase B only, from S4 weights, phase B hard_borrow=all, borrow a bit between places | 10 | 3 | 0 | 0.235 | 200, 0.235 | 0.730 | (0,0,0) (0,0,1) (0,2,0) (0,2,1) (0,3,0) (0,3,1) (0,4,0) (0,4,1) (0,5,0) (0,6,0) (0,6,1) (0,7,0) … |
| shared subtractor, choice held uniform for 400 steps, phase B only, from S4 weights, phase B hard_borrow=all, borrow a bit between places | 10 | 0 | 4 | 0.240 | 200, 0.240 | 0.675 | (0,0,0) (0,0,1) (0,1,0) (0,1,1) (0,2,0) (0,2,1) (0,3,0) (0,4,0) (0,4,1) (0,5,0) (0,5,1) (0,6,0) … |
| shared subtractor, choice held uniform for 400 steps, phase B only, from S4 weights, phase B hard_borrow=all, borrow a bit between places | 10 | 1 | 4 | 0.645 | 200, 0.645 | 0.830 | (0,0,0) (0,0,1) (0,1,1) (0,2,0) (0,4,1) (0,6,1) (0,7,0) (0,8,1) (0,9,0) (1,0,0) (1,0,1) (1,1,1) … |
| shared subtractor, choice held uniform for 400 steps, phase B only, from S4 weights, phase B hard_borrow=all, borrow a bit between places | 10 | 2 | 4 | 0.845 | 200, 0.845 | 0.850 | (0,0,0) (1,0,0) (1,0,1) (2,0,0) (2,0,1) (2,1,1) (3,0,0) (3,0,1) (4,0,0) (4,0,1) (5,0,0) (5,0,1) … |
| shared subtractor, choice held uniform for 400 steps, phase B only, from S4 weights, phase B hard_borrow=all, borrow a bit between places | 10 | 3 | 4 | 0.770 | 200, 0.770 | 0.810 | (0,0,0) (0,0,1) (0,6,1) (1,0,1) (1,1,0) (2,0,0) (2,0,1) (2,1,0) (2,1,1) (2,8,1) (3,0,0) (3,0,1) … |
| shared subtractor, not staged | 6 | 0 | 0 | 0.160 | 162, 0.198 | 0.560 | (0,0,0) (0,0,1) (0,1,1) (0,2,0) (0,2,1) (0,3,0) (0,3,1) (0,4,1) (0,5,1) (0,6,1) (0,7,0) (0,8,0) … |
| shared subtractor, not staged | 6 | 1 | 0 | 0.095 | 162, 0.117 | 0.430 | (0,0,0) (0,0,1) (0,1,0) (0,1,1) (0,2,0) (0,2,1) (0,3,0) (0,3,1) (0,4,1) (0,5,1) (0,6,0) (0,6,1) … |
| shared subtractor, not staged | 7 | 0 | 0 | 0.070 | 200, 0.070 | 0.575 | (0,0,1) (0,1,0) (0,1,1) (0,2,0) (0,2,1) (0,3,0) (0,3,1) (0,4,0) (0,4,1) (0,5,0) (0,5,1) (0,6,0) … |
| shared subtractor, not staged | 7 | 1 | 0 | 0.080 | 200, 0.080 | 0.510 | (0,0,1) (0,1,0) (0,1,1) (0,2,0) (0,2,1) (0,3,0) (0,3,1) (0,4,0) (0,4,1) (0,5,0) (0,5,1) (0,6,0) … |
| shared subtractor, not staged | 10 | 0 | 0 | 0.085 | 200, 0.085 | 0.550 | (0,0,0) (0,0,1) (0,1,0) (0,1,1) (0,2,0) (0,2,1) (0,3,0) (0,3,1) (0,4,0) (0,4,1) (0,5,0) (0,5,1) … |
| shared subtractor, not staged | 10 | 1 | 0 | 0.065 | 200, 0.065 | 0.565 | (0,0,0) (0,0,1) (0,1,0) (0,1,1) (0,2,0) (0,2,1) (0,3,0) (0,3,1) (0,4,0) (0,4,1) (0,5,0) (0,5,1) … |
| shared subtractor, own borrow from the start | 6 | 0 | 0 | 0.810 | 162, 1.000 | 1.000 | (0,0,1) (0,1,1) (0,2,1) (0,3,1) (1,1,1) (1,2,1) (1,3,1) (1,4,1) (2,2,1) (2,3,1) (2,4,1) (2,5,1) … |
| shared subtractor, own borrow from the start | 6 | 1 | 0 | 0.775 | 162, 0.951 | 0.955 | (0,0,1) (0,1,1) (0,2,1) (0,3,1) (0,7,1) (0,8,1) (1,1,1) (1,2,1) (1,3,1) (1,4,1) (1,7,1) (1,8,1) … |
| shared subtractor, own borrow from the start | 7 | 0 | 0 | 0.510 | 200, 0.510 | 0.630 | (0,0,1) (0,1,1) (0,2,1) (0,3,1) (0,4,1) (0,5,1) (0,6,1) (0,7,1) (0,8,1) (0,9,1) (1,0,1) (1,1,1) … |
| shared subtractor, own borrow from the start | 7 | 1 | 0 | 0.515 | 200, 0.515 | 0.745 | (0,0,1) (0,1,1) (0,2,1) (0,3,1) (0,4,1) (0,5,1) (0,6,1) (0,7,1) (0,9,1) (1,0,1) (1,1,1) (1,2,1) … |
| shared subtractor, own borrow from the start | 10 | 0 | 0 | 0.280 | 200, 0.280 | 0.550 | (0,0,1) (0,1,0) (0,1,1) (0,2,0) (0,2,1) (0,3,0) (0,3,1) (0,4,0) (0,4,1) (0,5,0) (0,5,1) (0,6,0) … |
| shared subtractor, own borrow from the start | 10 | 1 | 0 | 0.080 | 200, 0.080 | 0.685 | (0,0,0) (0,0,1) (0,1,0) (0,1,1) (0,2,0) (0,2,1) (0,3,0) (0,3,1) (0,4,0) (0,4,1) (0,5,0) (0,5,1) … |
| shared subtractor, window 2/1 | 6 | 0 | 0 | 0.810 | 162, 1.000 | 1.000 | (0,0,1) (0,1,1) (0,2,1) (0,3,1) (1,1,1) (1,2,1) (1,3,1) (1,4,1) (2,2,1) (2,3,1) (2,4,1) (2,5,1) … |
| shared subtractor, window 2/1 | 6 | 1 | 0 | 0.810 | 162, 1.000 | 1.000 | (0,0,1) (0,1,1) (0,2,1) (0,3,1) (1,1,1) (1,2,1) (1,3,1) (1,4,1) (2,2,1) (2,3,1) (2,4,1) (2,5,1) … |
| shared subtractor, window 2/1 | 6 | 2 | 0 | 0.805 | 162, 0.994 | 1.000 | (0,0,1) (0,1,1) (0,2,1) (0,3,1) (1,1,1) (1,2,1) (1,3,1) (1,4,1) (2,2,1) (2,3,1) (2,4,1) (2,5,1) … |
| shared subtractor, window 2/1 | 7 | 0 | 0 | 0.290 | 200, 0.290 | 0.545 | (0,0,1) (0,1,0) (0,1,1) (0,2,0) (0,2,1) (0,3,0) (0,3,1) (0,4,0) (0,4,1) (0,5,0) (0,5,1) (0,6,0) … |
| shared subtractor, window 2/1 | 7 | 1 | 0 | 0.540 | 200, 0.540 | 0.790 | (0,0,1) (0,1,1) (0,2,1) (0,3,1) (0,4,1) (0,5,1) (0,6,1) (0,7,1) (0,8,1) (0,9,1) (1,1,1) (1,2,1) … |
| shared subtractor, window 2/1 | 7 | 2 | 0 | 0.545 | 200, 0.545 | 0.820 | (0,0,1) (0,1,1) (0,2,1) (0,3,1) (0,4,1) (0,5,1) (0,6,1) (0,7,1) (0,8,1) (1,0,1) (1,1,1) (1,2,1) … |
| shared subtractor, window 2/1 | 10 | 0 | 0 | 0.285 | 200, 0.285 | 0.550 | (0,0,1) (0,1,0) (0,1,1) (0,2,0) (0,2,1) (0,3,0) (0,3,1) (0,4,0) (0,4,1) (0,5,0) (0,5,1) (0,6,0) … |
| shared subtractor, window 2/1 | 10 | 1 | 0 | 0.605 | 200, 0.605 | 0.900 | (0,0,1) (0,1,1) (0,2,1) (0,3,1) (0,4,1) (0,5,1) (0,6,1) (0,7,1) (0,8,1) (1,0,1) (1,1,1) (1,2,1) … |
| shared subtractor, window 2/1 | 10 | 2 | 0 | 0.100 | 200, 0.100 | 0.550 | (0,0,1) (0,1,0) (0,1,1) (0,2,0) (0,2,1) (0,3,0) (0,3,1) (0,4,0) (0,4,1) (0,5,0) (0,5,1) (0,6,0) … |

### maps

| variant | bits | seed | holdout | estimator phase: choice map (all steps right, mapped) | after the switch: choice map (all steps right) |
|---|---|---|---|---|---|
| shared subtractor | 6 | 0 | 0 | [0, 1, 2, 3, 4, 5, 6, 7, 8, 9] (0.85) | [0, 1, 2, 3, 4, 5, 6, 7, 8, 9] (1.00) |
| shared subtractor | 6 | 1 | 0 | [0, 0, 2, 0, 0, 0, 0, 0, 0, 0] (0.02) | [1, 0, 2, 5, 3, 5, 5, 2, 4, 4] (0.02) |
| shared subtractor | 6 | 2 | 0 | [5, 1, 2, 0, 4, 5, 6, 7, 8, 0] (0.33) | [0, 1, 2, 3, 4, 5, 6, 7, 8, 9] (1.00) |
| shared subtractor | 6 | 3 | 0 | [0, 1, 2, 3, 4, 5, 6, 7, 8, 9] (0.84) | [0, 1, 2, 3, 4, 5, 6, 7, 8, 9] (1.00) |
| shared subtractor | 6 | 0 | 4 | [0, 1, 2, 3, 4, 5, 6, 7, 8, 9] (0.81) | [0, 1, 2, 3, 4, 5, 6, 7, 8, 9] (1.00) |
| shared subtractor | 6 | 1 | 4 | [0, 0, 2, 1, 2, 1, 1, 1, 4, 3] (0.09) | [0, 1, 2, 3, 4, 5, 6, 7, 8, 9] (0.98) |
| shared subtractor | 6 | 2 | 4 | [0, 0, 2, 0, 0, 0, 0, 0, 0, 0] (0.02) | [1, 6, 0, 0, 1, 0, 0, 4, 2, 2] (0.00) |
| shared subtractor | 6 | 3 | 4 | [0, 1, 8, 2, 8, 5, 4, 7, 8, 9] (0.32) | [0, 1, 2, 3, 4, 5, 6, 7, 8, 9] (1.00) |
| shared subtractor | 7 | 0 | 0 | [0, 0, 0, 0, 0, 0, 0, 0, 0, 0] (0.01) | [0, 0, 0, 0, 0, 0, 0, 0, 0, 0] (0.00) |
| shared subtractor | 7 | 1 | 0 | [0, 1, 2, 3, 4, 5, 6, 7, 8, 9] (0.80) | [0, 0, 9, 0, 5, 1, 2, 2, 2, 6] (0.01) |
| shared subtractor | 7 | 2 | 0 | [0, 1, 2, 3, 4, 5, 6, 7, 8, 9] (0.80) | [0, 1, 0, 3, 0, 2, 0, 1, 0, 5] (0.01) |
| shared subtractor | 7 | 3 | 0 | [0, 1, 2, 3, 4, 5, 6, 7, 8, 9] (0.80) | [0, 1, 2, 3, 4, 5, 6, 7, 8, 9] (0.99) |
| shared subtractor | 7 | 0 | 4 | [0, 0, 0, 0, 0, 0, 0, 0, 0, 0] (0.01) | [0, 0, 0, 0, 0, 0, 0, 0, 0, 0] (0.00) |
| shared subtractor | 7 | 1 | 4 | [0, 1, 2, 3, 4, 5, 6, 7, 8, 9] (0.81) | [0, 0, 0, 0, 0, 0, 0, 0, 0, 0] (0.00) |
| shared subtractor | 7 | 2 | 4 | [0, 1, 2, 3, 4, 5, 6, 7, 8, 9] (0.80) | [0, 1, 2, 3, 4, 5, 6, 7, 8, 9] (0.99) |
| shared subtractor | 7 | 3 | 4 | [0, 1, 2, 3, 4, 5, 6, 7, 8, 9] (0.79) | [0, 1, 2, 3, 4, 5, 6, 7, 8, 9] (1.00) |
| shared subtractor | 10 | 0 | 0 | [0, 0, 0, 0, 0, 0, 0, 0, 0, 0] (0.00) | [0, 0, 0, 0, 0, 0, 0, 0, 0, 0] (0.00) |
| shared subtractor | 10 | 1 | 0 | [0, 1, 2, 3, 4, 5, 6, 7, 8, 9] (0.78) | [0, 0, 3, 4, 0, 0, 0, 0, 0, 0] (0.00) |
| shared subtractor | 10 | 2 | 0 | [0, 1, 2, 3, 4, 5, 6, 7, 8, 9] (0.77) | [0, 1, 2, 3, 4, 5, 6, 7, 8, 9] (1.00) |
| shared subtractor | 10 | 3 | 0 | [0, 1, 2, 3, 4, 5, 6, 7, 8, 9] (0.76) | [0, 1, 2, 3, 4, 5, 6, 7, 8, 9] (1.00) |
| shared subtractor | 10 | 0 | 4 | [0, 0, 0, 0, 0, 0, 0, 0, 0, 0] (0.00) | [7, 0, 0, 2, 0, 0, 0, 0, 0, 0] (0.00) |
| shared subtractor | 10 | 1 | 4 | [0, 0, 0, 0, 0, 0, 0, 0, 0, 0] (0.00) | [0, 1, 0, 0, 0, 0, 0, 0, 0, 0] (0.00) |
| shared subtractor | 10 | 2 | 4 | [0, 1, 2, 3, 4, 5, 6, 7, 8, 9] (0.76) | [0, 1, 2, 3, 4, 5, 6, 7, 8, 9] (1.00) |
| shared subtractor | 10 | 3 | 4 | [0, 1, 2, 3, 4, 5, 6, 7, 8, 9] (0.74) | [0, 0, 0, 0, 0, 0, 0, 0, 0, 0] (0.00) |
| learned slots (longdiv's cell) | 6 | 0 | 0 | [2, 5, 4, 1, 9, 0, 2, 4, 3, 3] (0.42) | [2, 9, 4, 1, 7, 0, 1, 5, 3, 3] (0.00) |
| learned slots (longdiv's cell) | 6 | 1 | 0 | [3, 0, 1, 2, 1, 3, 2, 3, 6, 2] (0.32) | [3, 0, 1, 9, 1, 4, 2, 3, 5, 2] (0.00) |
| learned slots (longdiv's cell) | 6 | 2 | 0 | [3, 5, 2, 2, 1, 3, 1, 2, 0, 1] (0.27) | [3, 5, 0, 0, 1, 3, 1, 2, 0, 1] (0.00) |
| learned slots (longdiv's cell) | 7 | 0 | 0 | [5, 4, 6, 2, 3, 8, 1, 5, 5, 0] (0.49) | [0, 5, 0, 0, 0, 0, 0, 0, 1, 0] (0.00) |
| learned slots (longdiv's cell) | 7 | 1 | 0 | [5, 0, 6, 6, 4, 0, 5, 2, 1, 8] (0.40) | [1, 2, 0, 0, 0, 0, 0, 0, 1, 0] (0.00) |
| learned slots (longdiv's cell) | 7 | 2 | 0 | [7, 5, 2, 5, 1, 0, 0, 4, 0, 6] (0.31) | [0, 1, 0, 1, 0, 0, 0, 0, 0, 0] (0.00) |
| learned slots (longdiv's cell) | 10 | 0 | 0 | [4, 1, 0, 7, 2, 3, 5, 2, 1, 0] (0.23) | [0, 0, 0, 0, 0, 0, 0, 0, 0, 0] (0.00) |
| learned slots (longdiv's cell) | 10 | 1 | 0 | [2, 0, 4, 5, 9, 9, 1, 3, 7, 0] (0.27) | [0, 0, 0, 0, 0, 0, 0, 0, 0, 0] (0.00) |
| learned slots (longdiv's cell) | 10 | 2 | 0 | [1, 0, 1, 2, 2, 0, 4, 5, 3, 7] (0.19) | [0, 0, 0, 0, 0, 0, 0, 0, 0, 0] (0.00) |
| shared subtractor, always from the top w digits | 6 | 0 | 0 | [0, 1, 2, 2, 4, 5, 6, 7, 8, 9] (0.30) | [0, 1, 2, 3, 4, 0, 1, 2, 3, 3] (0.01) |
| shared subtractor, always from the top w digits | 6 | 1 | 0 | [0, 0, 2, 0, 0, 0, 0, 0, 0, 0] (0.02) | [0, 1, 2, 1, 2, 9, 2, 3, 1, 1] (0.04) |
| shared subtractor, always from the top w digits | 7 | 0 | 0 | [0, 0, 0, 0, 0, 0, 0, 0, 0, 0] (0.01) | [0, 0, 0, 0, 1, 0, 0, 0, 0, 0] (0.00) |
| shared subtractor, always from the top w digits | 7 | 1 | 0 | [0, 1, 2, 3, 4, 5, 6, 7, 8, 9] (0.70) | [0, 0, 0, 1, 4, 6, 0, 0, 0, 0] (0.00) |
| shared subtractor, always from the top w digits | 10 | 0 | 0 | [0, 0, 0, 0, 0, 0, 0, 0, 0, 0] (0.00) | [0, 0, 0, 0, 0, 0, 0, 0, 0, 0] (0.00) |
| shared subtractor, always from the top w digits | 10 | 1 | 0 | [0, 0, 0, 0, 0, 0, 0, 0, 0, 0] (0.00) | [0, 0, 0, 0, 0, 0, 0, 0, 0, 0] (0.00) |
| shared subtractor, choice held uniform for 400 steps | 6 | 0 | 0 | [0, 1, 2, 3, 4, 5, 6, 7, 8, 9] (0.91) | [0, 1, 2, 3, 4, 5, 6, 7, 8, 9] (1.00) |
| shared subtractor, choice held uniform for 400 steps | 6 | 1 | 0 | [0, 1, 2, 3, 4, 5, 6, 7, 8, 9] (0.97) | [0, 1, 2, 3, 4, 5, 6, 7, 8, 9] (1.00) |
| shared subtractor, choice held uniform for 400 steps | 6 | 2 | 0 | [0, 1, 2, 3, 4, 5, 6, 7, 8, 9] (0.92) | [0, 1, 2, 3, 4, 5, 6, 7, 8, 9] (1.00) |
| shared subtractor, choice held uniform for 400 steps | 6 | 3 | 0 | [0, 1, 2, 3, 4, 5, 6, 7, 8, 9] (0.96) | [0, 1, 2, 3, 4, 5, 6, 7, 8, 9] (1.00) |
| shared subtractor, choice held uniform for 400 steps | 6 | 0 | 4 | [0, 1, 2, 3, 4, 5, 6, 7, 8, 9] (0.90) | [0, 1, 2, 3, 4, 5, 6, 7, 8, 9] (1.00) |
| shared subtractor, choice held uniform for 400 steps | 6 | 1 | 4 | [0, 1, 2, 3, 4, 5, 6, 7, 8, 9] (0.88) | [0, 1, 2, 3, 4, 5, 6, 7, 8, 9] (1.00) |
| shared subtractor, choice held uniform for 400 steps | 6 | 2 | 4 | [0, 1, 2, 3, 4, 5, 6, 7, 8, 9] (0.82) | [0, 1, 2, 3, 4, 5, 6, 7, 8, 9] (1.00) |
| shared subtractor, choice held uniform for 400 steps | 6 | 3 | 4 | [0, 1, 2, 3, 4, 5, 6, 7, 8, 9] (0.81) | [0, 1, 2, 3, 4, 5, 6, 7, 8, 9] (1.00) |
| shared subtractor, choice held uniform for 400 steps | 7 | 0 | 0 | [0, 1, 2, 3, 4, 5, 6, 7, 8, 9] (0.78) | [0, 0, 2, 3, 4, 5, 6, 7, 8, 9] (0.33) |
| shared subtractor, choice held uniform for 400 steps | 7 | 0 | 0 | [0, 1, 2, 3, 4, 5, 6, 7, 8, 9] (0.77) | [0, 1, 2, 3, 4, 5, 6, 7, 8, 9] (0.91) |
| shared subtractor, choice held uniform for 400 steps | 7 | 1 | 0 | [0, 1, 2, 3, 4, 5, 6, 7, 8, 9] (0.79) | [0, 1, 2, 3, 4, 5, 6, 7, 8, 9] (0.83) |
| shared subtractor, choice held uniform for 400 steps | 7 | 1 | 0 | [0, 1, 2, 3, 4, 5, 6, 7, 8, 9] (0.79) | [0, 1, 2, 3, 4, 5, 6, 7, 8, 9] (0.83) |
| shared subtractor, choice held uniform for 400 steps | 7 | 2 | 0 | [0, 1, 2, 3, 4, 5, 6, 7, 8, 9] (0.77) | [0, 1, 2, 3, 4, 5, 6, 7, 8, 9] (0.89) |
| shared subtractor, choice held uniform for 400 steps | 7 | 2 | 0 | [0, 1, 2, 3, 4, 5, 6, 7, 8, 9] (0.77) | [0, 1, 2, 3, 4, 5, 6, 7, 8, 9] (0.89) |
| shared subtractor, choice held uniform for 400 steps | 7 | 3 | 0 | [0, 1, 2, 3, 4, 5, 6, 7, 8, 9] (0.78) | [0, 1, 2, 3, 4, 5, 6, 7, 8, 9] (0.99) |
| shared subtractor, choice held uniform for 400 steps | 7 | 3 | 0 | [0, 1, 2, 3, 4, 5, 6, 7, 8, 9] (0.78) | [0, 1, 2, 3, 4, 5, 6, 7, 8, 9] (0.99) |
| shared subtractor, choice held uniform for 400 steps | 7 | 0 | 4 | [0, 1, 2, 3, 4, 5, 6, 7, 8, 9] (0.79) | [0, 1, 2, 3, 4, 5, 6, 7, 8, 9] (0.99) |
| shared subtractor, choice held uniform for 400 steps | 7 | 1 | 4 | [0, 1, 2, 3, 4, 5, 6, 7, 8, 9] (0.80) | [0, 0, 0, 0, 0, 0, 0, 5, 0, 0] (0.01) |
| shared subtractor, choice held uniform for 400 steps | 7 | 2 | 4 | [0, 1, 2, 3, 4, 5, 6, 7, 8, 9] (0.78) | [0, 1, 2, 3, 4, 5, 6, 7, 8, 9] (0.95) |
| shared subtractor, choice held uniform for 400 steps | 7 | 3 | 4 | [0, 1, 2, 3, 4, 5, 6, 7, 8, 9] (0.77) | [0, 1, 2, 3, 4, 5, 6, 7, 8, 9] (0.98) |
| shared subtractor, choice held uniform for 400 steps | 10 | 0 | 0 | [0, 1, 2, 3, 4, 5, 6, 7, 8, 9] (0.75) | [0, 1, 2, 3, 4, 5, 6, 7, 8, 9] (1.00) |
| shared subtractor, choice held uniform for 400 steps | 10 | 1 | 0 | [0, 1, 2, 3, 4, 5, 6, 7, 8, 9] (0.75) | [0, 1, 2, 3, 4, 5, 6, 7, 8, 9] (1.00) |
| shared subtractor, choice held uniform for 400 steps | 10 | 2 | 0 | [0, 1, 2, 3, 4, 5, 6, 7, 8, 9] (0.73) | [0, 1, 2, 3, 4, 5, 6, 7, 8, 9] (1.00) |
| shared subtractor, choice held uniform for 400 steps | 10 | 3 | 0 | [0, 1, 2, 3, 4, 5, 6, 7, 8, 9] (0.76) | [0, 1, 2, 3, 4, 5, 6, 7, 8, 9] (1.00) |
| shared subtractor, choice held uniform for 400 steps | 10 | 0 | 4 | [0, 1, 2, 3, 4, 5, 6, 7, 8, 9] (0.74) | [0, 1, 2, 3, 4, 5, 6, 7, 8, 9] (1.00) |
| shared subtractor, choice held uniform for 400 steps | 10 | 1 | 4 | [0, 1, 2, 3, 4, 5, 6, 7, 8, 9] (0.74) | [0, 1, 2, 3, 4, 5, 6, 7, 8, 9] (1.00) |
| shared subtractor, choice held uniform for 400 steps | 10 | 2 | 4 | [0, 1, 2, 3, 4, 5, 6, 7, 8, 9] (0.75) | [0, 1, 2, 3, 4, 5, 6, 7, 8, 9] (1.00) |
| shared subtractor, choice held uniform for 400 steps | 10 | 3 | 4 | [0, 1, 2, 3, 4, 5, 6, 7, 8, 9] (0.75) | [0, 1, 2, 3, 4, 5, 6, 7, 8, 9] (1.00) |
| shared subtractor, choice held uniform for 400 steps, borrow a bit between places | 6 | 0 | 0 | [0, 1, 2, 3, 4, 5, 6, 7, 8, 9] (0.87) | [0, 1, 2, 3, 4, 5, 6, 7, 8, 9] (1.00) |
| shared subtractor, choice held uniform for 400 steps, borrow a bit between places | 6 | 1 | 0 | [0, 1, 2, 3, 4, 5, 6, 7, 8, 9] (0.95) | [0, 1, 2, 3, 4, 5, 6, 7, 8, 9] (1.00) |
| shared subtractor, choice held uniform for 400 steps, borrow a bit between places | 6 | 2 | 0 | [0, 1, 2, 3, 4, 5, 6, 7, 8, 9] (0.97) | [0, 1, 2, 3, 4, 5, 6, 7, 8, 9] (1.00) |
| shared subtractor, choice held uniform for 400 steps, borrow a bit between places | 6 | 3 | 0 | [0, 1, 2, 3, 4, 5, 6, 7, 8, 9] (0.97) | [0, 1, 2, 3, 4, 5, 6, 7, 8, 9] (1.00) |
| shared subtractor, choice held uniform for 400 steps, borrow a bit between places | 6 | 0 | 4 | [0, 1, 2, 3, 4, 5, 6, 7, 8, 9] (0.81) | [0, 1, 2, 3, 0, 0, 1, 2, 3, 3] (0.00) |
| shared subtractor, choice held uniform for 400 steps, borrow a bit between places | 6 | 1 | 4 | [0, 1, 2, 3, 4, 5, 6, 7, 8, 9] (0.86) | [0, 1, 2, 3, 4, 5, 6, 7, 8, 9] (1.00) |
| shared subtractor, choice held uniform for 400 steps, borrow a bit between places | 6 | 2 | 4 | [0, 1, 2, 3, 4, 5, 6, 7, 8, 9] (0.84) | [0, 1, 2, 3, 4, 5, 6, 7, 8, 9] (1.00) |
| shared subtractor, choice held uniform for 400 steps, borrow a bit between places | 6 | 3 | 4 | [0, 1, 2, 3, 4, 5, 6, 7, 8, 9] (0.87) | [0, 1, 2, 3, 4, 5, 6, 7, 8, 9] (1.00) |
| shared subtractor, choice held uniform for 400 steps, borrow a bit between places | 7 | 0 | 0 | [0, 1, 2, 3, 4, 5, 6, 7, 8, 9] (0.80) | [0, 0, 0, 7, 8, 5, 0, 1, 0, 0] (0.00) |
| shared subtractor, choice held uniform for 400 steps, borrow a bit between places | 7 | 1 | 0 | [0, 1, 2, 3, 4, 5, 6, 7, 8, 9] (0.80) | [0, 1, 2, 3, 4, 5, 6, 7, 8, 9] (1.00) |
| shared subtractor, choice held uniform for 400 steps, borrow a bit between places | 7 | 2 | 0 | [0, 1, 2, 3, 4, 5, 6, 7, 8, 9] (0.80) | [0, 1, 2, 3, 4, 5, 6, 1, 1, 0] (0.00) |
| shared subtractor, choice held uniform for 400 steps, borrow a bit between places | 7 | 3 | 0 | [0, 1, 2, 3, 4, 5, 6, 7, 8, 9] (0.76) | [8, 0, 0, 0, 3, 0, 0, 0, 0, 0] (0.00) |
| shared subtractor, choice held uniform for 400 steps, borrow a bit between places | 7 | 0 | 4 | [0, 1, 2, 3, 4, 5, 6, 7, 8, 9] (0.79) | [1, 2, 2, 3, 4, 5, 6, 1, 0, 0] (0.00) |
| shared subtractor, choice held uniform for 400 steps, borrow a bit between places | 7 | 1 | 4 | [0, 1, 2, 3, 4, 5, 6, 7, 8, 9] (0.78) | [0, 1, 2, 3, 4, 5, 6, 7, 8, 9] (1.00) |
| shared subtractor, choice held uniform for 400 steps, borrow a bit between places | 7 | 2 | 4 | [0, 1, 2, 3, 4, 5, 6, 7, 8, 9] (0.76) | [0, 2, 1, 1, 3, 0, 0, 0, 0, 0] (0.00) |
| shared subtractor, choice held uniform for 400 steps, borrow a bit between places | 7 | 3 | 4 | [0, 1, 2, 3, 4, 5, 6, 7, 8, 9] (0.78) | [0, 1, 2, 3, 0, 0, 0, 5, 0, 0] (0.00) |
| shared subtractor, choice held uniform for 400 steps, borrow a bit between places | 10 | 0 | 0 | [0, 1, 2, 3, 4, 5, 6, 7, 8, 9] (0.69) | [0, 0, 0, 0, 3, 0, 0, 0, 0, 0] (0.00) |
| shared subtractor, choice held uniform for 400 steps, borrow a bit between places | 10 | 1 | 0 | [0, 1, 2, 3, 4, 5, 6, 7, 8, 9] (0.74) | [0, 3, 0, 0, 0, 0, 1, 0, 0, 0] (0.00) |
| shared subtractor, choice held uniform for 400 steps, borrow a bit between places | 10 | 2 | 0 | [0, 1, 2, 3, 4, 5, 6, 7, 8, 9] (0.73) | [0, 7, 0, 5, 5, 1, 6, 4, 4, 0] (0.00) |
| shared subtractor, choice held uniform for 400 steps, borrow a bit between places | 10 | 3 | 0 | [0, 1, 2, 3, 4, 5, 6, 7, 8, 9] (0.71) | [3, 0, 1, 1, 7, 0, 0, 0, 0, 0] (0.00) |
| shared subtractor, choice held uniform for 400 steps, borrow a bit between places | 10 | 0 | 4 | [0, 1, 2, 3, 4, 5, 6, 7, 8, 9] (0.69) | [0, 3, 4, 0, 0, 2, 4, 2, 0, 0] (0.00) |
| shared subtractor, choice held uniform for 400 steps, borrow a bit between places | 10 | 1 | 4 | [0, 1, 2, 3, 4, 5, 6, 7, 8, 9] (0.73) | [0, 5, 0, 0, 0, 0, 0, 0, 0, 0] (0.00) |
| shared subtractor, choice held uniform for 400 steps, borrow a bit between places | 10 | 2 | 4 | [0, 1, 2, 3, 4, 5, 6, 7, 8, 9] (0.68) | [4, 4, 3, 3, 4, 0, 2, 0, 0, 0] (0.00) |
| shared subtractor, choice held uniform for 400 steps, borrow a bit between places | 10 | 3 | 4 | [0, 1, 2, 3, 4, 5, 6, 7, 8, 9] (0.73) | [0, 0, 6, 5, 5, 0, 0, 0, 0, 0] (0.00) |
| shared subtractor, choice held uniform for 400 steps, handover over 500 steps | 6 | 0 | 0 | [0, 1, 2, 3, 4, 5, 6, 7, 8, 9] (0.91) | [0, 1, 2, 3, 4, 5, 6, 7, 8, 9] (1.00) |
| shared subtractor, choice held uniform for 400 steps, handover over 500 steps | 6 | 1 | 0 | [0, 1, 2, 3, 4, 5, 6, 7, 8, 9] (0.94) | [0, 1, 2, 3, 4, 5, 6, 7, 8, 9] (0.98) |
| shared subtractor, choice held uniform for 400 steps, handover over 500 steps | 6 | 2 | 0 | [0, 1, 2, 3, 4, 5, 6, 7, 8, 9] (0.92) | [0, 1, 2, 3, 4, 5, 6, 7, 8, 9] (1.00) |
| shared subtractor, choice held uniform for 400 steps, handover over 500 steps | 6 | 3 | 0 | [0, 1, 2, 3, 4, 5, 6, 7, 8, 9] (0.96) | [0, 1, 2, 3, 4, 5, 6, 7, 8, 9] (1.00) |
| shared subtractor, choice held uniform for 400 steps, handover over 500 steps | 6 | 0 | 4 | [0, 1, 2, 3, 4, 5, 6, 7, 8, 9] (0.90) | [0, 1, 2, 3, 4, 5, 6, 7, 8, 9] (1.00) |
| shared subtractor, choice held uniform for 400 steps, handover over 500 steps | 6 | 1 | 4 | [0, 1, 2, 3, 4, 5, 6, 7, 8, 9] (0.88) | [0, 1, 2, 3, 4, 5, 6, 7, 8, 9] (1.00) |
| shared subtractor, choice held uniform for 400 steps, handover over 500 steps | 6 | 2 | 4 | [0, 1, 2, 3, 4, 5, 6, 7, 8, 9] (0.82) | [0, 1, 2, 3, 4, 5, 6, 7, 8, 9] (1.00) |
| shared subtractor, choice held uniform for 400 steps, handover over 500 steps | 6 | 3 | 4 | [0, 1, 2, 3, 4, 5, 6, 7, 8, 9] (0.83) | [0, 1, 2, 3, 4, 5, 6, 7, 8, 9] (1.00) |
| shared subtractor, choice held uniform for 400 steps, handover over 500 steps | 7 | 0 | 0 | [0, 1, 2, 3, 4, 5, 6, 7, 8, 9] (0.77) | [0, 1, 2, 3, 4, 5, 6, 7, 8, 9] (0.89) |
| shared subtractor, choice held uniform for 400 steps, handover over 500 steps | 7 | 1 | 0 | [0, 1, 2, 3, 4, 5, 6, 7, 8, 9] (0.79) | [0, 1, 2, 3, 4, 5, 6, 7, 8, 9] (0.88) |
| shared subtractor, choice held uniform for 400 steps, handover over 500 steps | 7 | 2 | 0 | [0, 1, 2, 3, 4, 5, 6, 7, 8, 9] (0.77) | [0, 0, 2, 3, 4, 5, 6, 7, 8, 9] (0.25) |
| shared subtractor, choice held uniform for 400 steps, handover over 500 steps | 7 | 3 | 0 | [0, 1, 2, 3, 4, 5, 6, 7, 8, 9] (0.78) | [0, 1, 2, 3, 4, 5, 6, 7, 8, 9] (0.99) |
| shared subtractor, choice held uniform for 400 steps, handover over 500 steps | 7 | 0 | 4 | [0, 1, 2, 3, 4, 5, 6, 7, 8, 9] (0.79) | [0, 1, 2, 3, 4, 5, 6, 7, 8, 9] (0.97) |
| shared subtractor, choice held uniform for 400 steps, handover over 500 steps | 7 | 1 | 4 | [0, 1, 2, 3, 4, 5, 6, 7, 8, 9] (0.80) | [0, 0, 0, 0, 0, 0, 0, 0, 0, 0] (0.01) |
| shared subtractor, choice held uniform for 400 steps, handover over 500 steps | 7 | 2 | 4 | [0, 1, 2, 3, 4, 5, 6, 7, 8, 9] (0.78) | [0, 1, 2, 3, 4, 5, 6, 7, 8, 9] (0.89) |
| shared subtractor, choice held uniform for 400 steps, handover over 500 steps | 7 | 3 | 4 | [0, 1, 2, 3, 4, 5, 6, 7, 8, 9] (0.77) | [0, 1, 2, 3, 4, 5, 6, 7, 8, 9] (0.97) |
| shared subtractor, choice held uniform for 400 steps, handover over 500 steps | 10 | 0 | 0 | [0, 1, 2, 3, 4, 5, 6, 7, 8, 9] (0.74) | [0, 1, 2, 3, 4, 5, 6, 7, 8, 9] (1.00) |
| shared subtractor, choice held uniform for 400 steps, handover over 500 steps | 10 | 1 | 0 | [0, 1, 2, 3, 4, 5, 6, 7, 8, 9] (0.74) | [0, 1, 2, 3, 4, 5, 6, 7, 8, 9] (1.00) |
| shared subtractor, choice held uniform for 400 steps, handover over 500 steps | 10 | 2 | 0 | [0, 1, 2, 3, 4, 5, 6, 7, 8, 9] (0.73) | [0, 1, 2, 3, 4, 5, 6, 7, 8, 9] (1.00) |
| shared subtractor, choice held uniform for 400 steps, handover over 500 steps | 10 | 3 | 0 | [0, 1, 2, 3, 4, 5, 6, 7, 8, 9] (0.75) | [0, 1, 2, 3, 4, 5, 6, 7, 8, 9] (1.00) |
| shared subtractor, choice held uniform for 400 steps, handover over 500 steps | 10 | 0 | 4 | [0, 1, 2, 3, 4, 5, 6, 7, 8, 9] (0.75) | [0, 1, 2, 3, 4, 5, 6, 7, 8, 9] (1.00) |
| shared subtractor, choice held uniform for 400 steps, handover over 500 steps | 10 | 1 | 4 | [0, 1, 2, 3, 4, 5, 6, 7, 8, 9] (0.74) | [0, 1, 2, 3, 4, 5, 6, 7, 8, 9] (1.00) |
| shared subtractor, choice held uniform for 400 steps, handover over 500 steps | 10 | 2 | 4 | [0, 1, 2, 3, 4, 5, 6, 7, 8, 9] (0.75) | [0, 1, 2, 3, 4, 5, 6, 7, 8, 9] (1.00) |
| shared subtractor, choice held uniform for 400 steps, handover over 500 steps | 10 | 3 | 4 | [0, 1, 2, 3, 4, 5, 6, 7, 8, 9] (0.75) | [0, 1, 2, 3, 4, 5, 6, 7, 8, 9] (1.00) |
| shared subtractor, choice held uniform for 400 steps, lr 0.0003 after the switch | 7 | 0 | 0 | [0, 1, 2, 3, 4, 5, 6, 7, 8, 9] (0.78) | [0, 1, 1, 5, 0, 0, 0, 0, 0, 0] (0.02) |
| shared subtractor, choice held uniform for 400 steps, lr 0.0003 after the switch | 7 | 1 | 0 | [0, 1, 2, 3, 4, 5, 6, 7, 8, 9] (0.79) | [0, 1, 2, 3, 4, 5, 6, 7, 8, 9] (0.96) |
| shared subtractor, choice held uniform for 400 steps, lr 0.0003 after the switch | 7 | 2 | 0 | [0, 1, 2, 3, 4, 5, 6, 7, 8, 9] (0.77) | [0, 1, 2, 3, 4, 5, 6, 7, 8, 9] (0.77) |
| shared subtractor, choice held uniform for 400 steps, lr 0.0003 after the switch | 7 | 3 | 0 | [0, 1, 2, 3, 4, 5, 6, 7, 8, 9] (0.78) | [0, 1, 2, 3, 4, 5, 6, 7, 8, 9] (0.94) |
| shared subtractor, choice held uniform for 400 steps, lr 0.001 after the switch | 7 | 0 | 0 | [0, 1, 2, 3, 4, 5, 6, 7, 8, 9] (0.78) | [0, 0, 2, 3, 4, 5, 6, 7, 8, 9] (0.27) |
| shared subtractor, choice held uniform for 400 steps, lr 0.001 after the switch | 7 | 1 | 0 | [0, 1, 2, 3, 4, 5, 6, 7, 8, 9] (0.80) | [0, 1, 2, 3, 4, 5, 6, 7, 8, 9] (0.98) |
| shared subtractor, choice held uniform for 400 steps, lr 0.001 after the switch | 7 | 2 | 0 | [0, 1, 2, 3, 4, 5, 6, 7, 8, 9] (0.77) | [0, 1, 2, 3, 4, 5, 6, 7, 8, 9] (0.86) |
| shared subtractor, choice held uniform for 400 steps, lr 0.001 after the switch | 7 | 3 | 0 | [0, 1, 2, 3, 4, 5, 6, 7, 8, 9] (0.78) | [0, 1, 2, 3, 4, 5, 6, 7, 8, 9] (0.98) |
| shared subtractor, choice held uniform for 400 steps, phase B only, from S4 weights, phase B hard_borrow=all, borrow a bit between places | 6 | 0 | 0 | [0, 1, 2, 3, 4, 5, 6, 7, 8, 9] (0.87) | [0, 0, 0, 0, 0, 0, 0, 2, 0, 2] (0.00) |
| shared subtractor, choice held uniform for 400 steps, phase B only, from S4 weights, phase B hard_borrow=all, borrow a bit between places | 6 | 1 | 0 | [0, 1, 2, 3, 4, 5, 6, 7, 8, 9] (0.95) | [0, 0, 0, 0, 0, 0, 2, 2, 2, 0] (0.00) |
| shared subtractor, choice held uniform for 400 steps, phase B only, from S4 weights, phase B hard_borrow=all, borrow a bit between places | 6 | 2 | 0 | [0, 1, 2, 3, 4, 5, 6, 7, 8, 9] (0.97) | [0, 0, 0, 0, 0, 0, 0, 2, 0, 2] (0.00) |
| shared subtractor, choice held uniform for 400 steps, phase B only, from S4 weights, phase B hard_borrow=all, borrow a bit between places | 6 | 3 | 0 | [0, 1, 2, 3, 4, 5, 6, 7, 8, 9] (0.97) | [2, 0, 0, 0, 0, 0, 0, 0, 0, 0] (0.02) |
| shared subtractor, choice held uniform for 400 steps, phase B only, from S4 weights, phase B hard_borrow=all, borrow a bit between places | 6 | 0 | 4 | [0, 1, 2, 3, 4, 5, 6, 7, 8, 9] (0.81) | [0, 0, 1, 1, 0, 0, 0, 2, 3, 0] (0.00) |
| shared subtractor, choice held uniform for 400 steps, phase B only, from S4 weights, phase B hard_borrow=all, borrow a bit between places | 6 | 1 | 4 | [0, 1, 2, 3, 4, 5, 6, 7, 8, 9] (0.86) | [1, 0, 0, 0, 0, 0, 0, 0, 1, 2] (0.00) |
| shared subtractor, choice held uniform for 400 steps, phase B only, from S4 weights, phase B hard_borrow=all, borrow a bit between places | 6 | 2 | 4 | [0, 1, 2, 3, 4, 5, 6, 7, 8, 9] (0.84) | [0, 0, 0, 0, 0, 0, 0, 0, 0, 2] (0.00) |
| shared subtractor, choice held uniform for 400 steps, phase B only, from S4 weights, phase B hard_borrow=all, borrow a bit between places | 6 | 3 | 4 | [0, 1, 2, 3, 4, 5, 6, 7, 8, 9] (0.87) | [0, 0, 0, 0, 0, 0, 0, 2, 0, 2] (0.00) |
| shared subtractor, choice held uniform for 400 steps, phase B only, from S4 weights, phase B hard_borrow=all, borrow a bit between places | 7 | 0 | 0 | [0, 1, 2, 3, 4, 5, 6, 7, 8, 9] (0.80) | [0, 1, 2, 3, 4, 5, 6, 7, 8, 9] (1.00) |
| shared subtractor, choice held uniform for 400 steps, phase B only, from S4 weights, phase B hard_borrow=all, borrow a bit between places | 7 | 1 | 0 | [0, 1, 2, 3, 4, 5, 6, 7, 8, 9] (0.80) | [0, 1, 2, 3, 4, 5, 6, 7, 8, 9] (1.00) |
| shared subtractor, choice held uniform for 400 steps, phase B only, from S4 weights, phase B hard_borrow=all, borrow a bit between places | 7 | 2 | 0 | [0, 1, 2, 3, 4, 5, 6, 7, 8, 9] (0.80) | [0, 0, 0, 0, 0, 0, 0, 0, 0, 0] (0.01) |
| shared subtractor, choice held uniform for 400 steps, phase B only, from S4 weights, phase B hard_borrow=all, borrow a bit between places | 7 | 3 | 0 | [0, 1, 2, 3, 4, 5, 6, 7, 8, 9] (0.76) | [0, 0, 0, 0, 0, 0, 0, 0, 0, 0] (0.01) |
| shared subtractor, choice held uniform for 400 steps, phase B only, from S4 weights, phase B hard_borrow=all, borrow a bit between places | 7 | 0 | 4 | [0, 1, 2, 3, 4, 5, 6, 7, 8, 9] (0.79) | [0, 0, 0, 0, 0, 0, 0, 0, 0, 0] (0.01) |
| shared subtractor, choice held uniform for 400 steps, phase B only, from S4 weights, phase B hard_borrow=all, borrow a bit between places | 7 | 1 | 4 | [0, 1, 2, 3, 4, 5, 6, 7, 8, 9] (0.78) | [0, 0, 0, 0, 0, 0, 0, 0, 0, 0] (0.01) |
| shared subtractor, choice held uniform for 400 steps, phase B only, from S4 weights, phase B hard_borrow=all, borrow a bit between places | 7 | 2 | 4 | [0, 1, 2, 3, 4, 5, 6, 7, 8, 9] (0.76) | [0, 0, 0, 0, 0, 0, 0, 0, 0, 0] (0.01) |
| shared subtractor, choice held uniform for 400 steps, phase B only, from S4 weights, phase B hard_borrow=all, borrow a bit between places | 7 | 3 | 4 | [0, 1, 2, 3, 4, 5, 6, 7, 8, 9] (0.78) | [0, 0, 0, 0, 0, 0, 0, 0, 0, 0] (0.01) |
| shared subtractor, choice held uniform for 400 steps, phase B only, from S4 weights, phase B hard_borrow=all, borrow a bit between places | 10 | 0 | 0 | [0, 1, 2, 3, 4, 5, 6, 7, 8, 9] (0.69) | [0, 0, 0, 0, 0, 0, 0, 0, 0, 0] (0.00) |
| shared subtractor, choice held uniform for 400 steps, phase B only, from S4 weights, phase B hard_borrow=all, borrow a bit between places | 10 | 1 | 0 | [0, 1, 2, 3, 4, 5, 6, 7, 8, 9] (0.74) | [0, 1, 2, 3, 4, 5, 6, 7, 8, 9] (1.00) |
| shared subtractor, choice held uniform for 400 steps, phase B only, from S4 weights, phase B hard_borrow=all, borrow a bit between places | 10 | 2 | 0 | [0, 1, 2, 3, 4, 5, 6, 7, 8, 9] (0.73) | [0, 0, 0, 0, 0, 0, 0, 0, 0, 0] (0.00) |
| shared subtractor, choice held uniform for 400 steps, phase B only, from S4 weights, phase B hard_borrow=all, borrow a bit between places | 10 | 3 | 0 | [0, 1, 2, 3, 4, 5, 6, 7, 8, 9] (0.71) | [0, 0, 3, 1, 4, 1, 3, 0, 4, 8] (0.00) |
| shared subtractor, choice held uniform for 400 steps, phase B only, from S4 weights, phase B hard_borrow=all, borrow a bit between places | 10 | 0 | 4 | [0, 1, 2, 3, 4, 5, 6, 7, 8, 9] (0.69) | [1, 0, 0, 0, 0, 0, 0, 0, 0, 0] (0.00) |
| shared subtractor, choice held uniform for 400 steps, phase B only, from S4 weights, phase B hard_borrow=all, borrow a bit between places | 10 | 1 | 4 | [0, 1, 2, 3, 4, 5, 6, 7, 8, 9] (0.73) | [0, 0, 0, 0, 0, 0, 0, 0, 0, 0] (0.00) |
| shared subtractor, choice held uniform for 400 steps, phase B only, from S4 weights, phase B hard_borrow=all, borrow a bit between places | 10 | 2 | 4 | [0, 1, 2, 3, 4, 5, 6, 7, 8, 9] (0.68) | [0, 0, 0, 0, 0, 0, 0, 0, 0, 0] (0.00) |
| shared subtractor, choice held uniform for 400 steps, phase B only, from S4 weights, phase B hard_borrow=all, borrow a bit between places | 10 | 3 | 4 | [0, 1, 2, 3, 4, 5, 6, 7, 8, 9] (0.73) | [0, 0, 0, 0, 0, 0, 0, 0, 0, 0] (0.00) |
| shared subtractor, not staged | 6 | 0 | 0 | [0, 0, 1, 3, 3, 6, 2, 1, 2, 3] (0.09) | [0, 1, 0, 4, 4, 5, 2, 2, 2, 0] (0.02) |
| shared subtractor, not staged | 6 | 1 | 0 | [0, 0, 2, 0, 0, 0, 0, 0, 0, 0] (0.02) | [0, 2, 2, 1, 2, 3, 5, 3, 4, 2] (0.05) |
| shared subtractor, not staged | 7 | 0 | 0 | [0, 0, 0, 0, 0, 0, 2, 0, 0, 0] (0.01) | [0, 0, 0, 0, 0, 0, 0, 0, 0, 0] (0.00) |
| shared subtractor, not staged | 7 | 1 | 0 | [0, 0, 0, 0, 0, 1, 0, 0, 0, 0] (0.01) | [0, 0, 0, 0, 0, 0, 0, 0, 0, 0] (0.00) |
| shared subtractor, not staged | 10 | 0 | 0 | [0, 0, 0, 0, 0, 0, 0, 0, 0, 0] (0.00) | [0, 0, 0, 0, 0, 0, 0, 0, 0, 0] (0.00) |
| shared subtractor, not staged | 10 | 1 | 0 | [0, 0, 0, 0, 0, 0, 0, 0, 0, 0] (0.00) | [0, 0, 0, 0, 0, 0, 0, 0, 0, 0] (0.00) |
| shared subtractor, own borrow from the start | 6 | 0 | 0 | [0, 1, 2, 3, 4, 5, 6, 7, 8, 9] (1.00) | [0, 1, 2, 3, 4, 5, 6, 7, 8, 9] (1.00) |
| shared subtractor, own borrow from the start | 6 | 1 | 0 | [0, 1, 2, 3, 4, 5, 6, 7, 8, 9] (1.00) | [0, 1, 2, 3, 4, 5, 6, 7, 8, 9] (1.00) |
| shared subtractor, own borrow from the start | 7 | 0 | 0 | [0, 1, 2, 3, 4, 5, 6, 7, 8, 9] (0.99) | [0, 1, 2, 3, 4, 5, 6, 7, 8, 9] (0.99) |
| shared subtractor, own borrow from the start | 7 | 1 | 0 | [0, 0, 0, 3, 0, 0, 0, 0, 0, 7] (0.01) | [0, 1, 2, 3, 4, 5, 6, 7, 8, 9] (0.96) |
| shared subtractor, own borrow from the start | 10 | 0 | 0 | [0, 0, 0, 0, 0, 0, 0, 0, 0, 0] (0.00) | [0, 0, 0, 0, 0, 0, 0, 0, 0, 0] (0.00) |
| shared subtractor, own borrow from the start | 10 | 1 | 0 | [0, 0, 0, 0, 0, 0, 0, 0, 8, 0] (0.00) | [0, 4, 0, 0, 0, 0, 0, 0, 0, 0] (0.00) |
| shared subtractor, window 2/1 | 6 | 0 | 0 | [0, 1, 2, 3, 4, 5, 6, 7, 8, 9] (0.70) | [0, 1, 2, 3, 4, 5, 6, 7, 8, 9] (1.00) |
| shared subtractor, window 2/1 | 6 | 1 | 0 | [0, 1, 2, 3, 4, 5, 6, 7, 8, 9] (0.60) | [0, 1, 2, 3, 4, 5, 6, 7, 8, 9] (1.00) |
| shared subtractor, window 2/1 | 6 | 2 | 0 | [0, 1, 2, 3, 4, 5, 6, 7, 8, 9] (0.63) | [0, 1, 2, 3, 4, 5, 6, 7, 8, 9] (1.00) |
| shared subtractor, window 2/1 | 7 | 0 | 0 | [0, 1, 2, 3, 4, 5, 6, 7, 8, 9] (0.79) | [0, 1, 2, 3, 4, 5, 6, 7, 8, 9] (0.99) |
| shared subtractor, window 2/1 | 7 | 1 | 0 | [0, 1, 2, 3, 4, 5, 6, 7, 8, 9] (0.77) | [0, 1, 2, 3, 4, 5, 6, 7, 8, 9] (0.94) |
| shared subtractor, window 2/1 | 7 | 2 | 0 | [0, 1, 2, 3, 4, 5, 6, 7, 8, 9] (0.78) | [0, 0, 0, 3, 4, 5, 6, 7, 8, 9] (0.03) |
| shared subtractor, window 2/1 | 10 | 0 | 0 | [0, 1, 2, 3, 4, 5, 6, 7, 8, 9] (0.59) | [0, 0, 0, 0, 0, 0, 0, 0, 0, 3] (0.00) |
| shared subtractor, window 2/1 | 10 | 1 | 0 | [0, 1, 2, 3, 4, 5, 6, 7, 8, 9] (0.69) | [0, 0, 0, 0, 0, 0, 0, 0, 0, 0] (0.00) |
| shared subtractor, window 2/1 | 10 | 2 | 0 | [0, 0, 0, 0, 0, 0, 0, 0, 0, 0] (0.00) | [0, 0, 0, 0, 0, 0, 0, 0, 0, 0] (0.00) |
