# Gazeebo

Gazeebo is a local, process-scoped gaze-navigation tool for Linux desktops. It uses a webcam to move the cursor across every display authorized through the desktop portal. It emits cursor motion only: no eye gesture, pointer-button, keyboard, scrolling, drag, or dwell-click action exists.

## Runtime requirements

Gazeebo needs:

- Linux with a desktop portal that supports RemoteDesktop pointer authorization and returns logical geometry for every selected monitor;
- a Wayland session with layer shell for initial and on-demand target training;
- a locally attached camera supported by OpenCV and V4L2.

The portal selector defines the authorized displays. Gazeebo does not read compositor configuration or invoke a window-manager command. A portal backend that omits multi-display positions fails safely.

## Run

```console
gazeebo
gazeebo --camera /dev/video2
gazeebo --camera 2
gazeebo --camera /dev/video2 --camera-codec MJPG --width 848 --height 480 --fps 30
```

`--camera-codec YUYV` requests uncompressed V4L2 transport, while `--camera-codec MJPG` requests camera-compressed Motion JPEG. OpenSeeFace always analyzes decoded pixels. Explicit codec and negotiated dimensions form part of the opaque camera fingerprint, so calibration collected under one transport cannot silently mix with another. The default leaves codec negotiation to the camera backend.

Startup concurrently loads local training state and vision resources while requesting portal authorization where safe. Cursor motion begins only after pointer access and valid authorized geometry are available.

On the first run, Gazeebo collects five deterministic seed targets across the authorized displays to establish a finite model. It predicts every subsequent target with a model that has not seen that target, incorporates the result immediately, and selects the next target from the updated model and region evidence. It saves every completed target aggregate. Later runs passively select or blend automatic context experts and begin navigation without mandatory target validation. During navigation, Gazeebo refreshes available output geometry and camera context once per second by default. It applies topology first, posture and camera geometry second, and illumination last. Model routing therefore follows mid-session posture and lighting changes instead of remaining fixed at startup.

Predictions can cross authorized displays. Predictions in desktop gaps or outside the selected union project to the nearest authorized display. Motion defaults to snapping to a 15-position rolling median of model coordinates sampled at camera cadence, emitted at most once every 100 milliseconds. This rejects isolated stationary jumps without reintroducing slow cursor transit. A finite 10,000-pixel step bound and noise-adaptive dead zone remain active; smoothing alpha and step size are configurable. `--pointer-update-interval 0` requests continuous updates. A display addition pauses and refreshes portal authorization by default. `--no-allow-display-reauthorization-pause` keeps the existing authorized union after additions, but it can never authorize a new output or retain invalid geometry.

## Add training data

```console
gazeebo train
```

If Gazeebo is active, this command asks that foreground process to enter training through an owner-only Unix socket. Otherwise it starts a foreground training session. No daemon or helper remains afterward.

Training starts immediately whenever it is requested; users do not have to manufacture a posture or camera state different from existing evidence. A visible `3…2…1…` sequence appears on every authorized display. Before each circle, every display shows an arrow toward its next location, including moves to another monitor. The target label shows total progress against the 55-circle limit.

Gazeebo uses current Wayland output geometry, the selected active pixel mode, and physical output dimensions to keep circles close to one configured physical diameter across displays. It does not use unrelated supported modes. If the authorized output cannot be mapped exactly or physical metadata is unavailable, `--training-fallback-diameter` supplies the configurable logical-pixel fallback.

Each click-through, keyboard-inactive circle has a two-second preparation phase by default. The next dot appears inside a sharp, black-backed multicolored square while the prior dot fades for the first second; arrows remain visible on every display. The square then disappears and starts two seconds of active measurement. Gazeebo always uses reliable head pose and normalized face geometry, refining them with eye-local pupil angle when available. Closed, hidden, unstable, or low-confidence pupils neither reset nor fail a target.

If head/face tracking becomes unreliable, measurement time and cursor motion pause while every display shows a transient live camera view with face bounds, pose axes, and corrective guidance. The view remains for at least `--head-diagnostic-minimum` (3 seconds by default), even after quick recovery, so the guidance is readable. Tracking resumes automatically after that minimum once the head is reliable. Failure to recover within `--head-recovery-timeout` (10 seconds by default) shows a terminal diagnosis and exits safely.

### Sensitive false-negative captures

Gazeebo **enables diagnostic frame capture by default** to make false tracking-loss warnings debuggable. A sustained warning retains every lossless raw detector frame from three seconds before through three seconds after the warning, plus per-frame failure reason, confidence, face bounds, pose, and all available landmarks. One event sequence stores the first frame and bytewise XOR deltas for later frames under whole-sequence gzip compression, exploiting inter-frame similarity without losing pixels. Captures are sensitive biometric images. They are stored separately from training data under `$XDG_DATA_HOME/gazeebo/diagnostics-v1` (or `~/.local/share/gazeebo/diagnostics-v1`) with owner-only `0700` directories and `0600` files. They never affect training, navigation, or model selection.

Disable all archive reads and writes for an invocation with `--no-diagnostic-capture`. To disable capture by configuration, create `$XDG_CONFIG_HOME/gazeebo/config.toml` (or `~/.config/gazeebo/config.toml`):

```toml
[diagnostics]
capture = false
```

An explicit `--diagnostic-capture` or `--no-diagnostic-capture` overrides the file. `gazeebo diagnostic-stats` reports archive event count and bytes without opening the camera. `gazeebo reset-diagnostics` permanently removes only a validated diagnostic archive. Captures remain until that reset. A 2 GiB quota preserves old events and stops new capture instead of evicting evidence. Continuous or untriggered camera recording is not supported.

During the active window, Gazeebo predicts an estimated focal position from every raw reliable camera observation, independently of pointer-event cadence. Unseen error and cursor-noise statistics use this high-rate estimate stream. A separate 15-position median determines only the rendered cursor. Gazeebo discards both streams after reducing the raw estimates to target-level dispersion, covariance, median radial spread, and 95th-percentile radial spread. Compatible summaries tune rendering and dead zones within conservative limits and provide bounded, strictly positive fit weights; sparse or incompatible evidence retains fixed safe defaults.

Each target's prediction is recorded before that target enters the fit. Robust per-feature dispersion from the raw active window adds a bounded errors-in-variables penalty, so the Bayesian fit does not amplify noisy face or pupil dimensions. Gazeebo then updates fixed-dimensional sufficient statistics and refits the provisional model before selecting the next target.

Adaptive selection divides every authorized output into a normalized 3×3 grid covering its corners, edges, and center. A new camera seeds the cells deterministically across outputs; if 55 targets cannot cover them all, the next invocation resumes the missing cells. Each observed cell keeps a deterministic fixed-size exponentially decayed error histogram with exact weighted sums inside each bin. It estimates CVaR90 as the interpolated weighted mean of the worst 10% of regional errors. Bayesian predictive uncertainty raises that expected tail cost, while cursor noise widens its one-sided confidence interval; neither gates evidence. Once every cell has error evidence, Gazeebo selects the cell with the highest CVaR90 surprise bound whenever its lower bound exceeds another cell's upper bound. It keeps working on statistically high-surprise cells until all cell intervals overlap, then balances tied cells by observation count and stable output/cell order. A fixed-work low-discrepancy sequence supplies unseen positions inside the selected cell.

Consecutive groups of five unseen predictions report hit rate, median radial error, edge/corner error, response time, cursor spread, selected regions, CVaR90 and its surprise interval, region coverage, and model routing; reports and early stopping remain batch-based. Persistent no-regression acceptance aggregates every unseen target in the invocation and rejects worse worst-region median, CVaR90, or upper-bound quality, so strong global or final-report medians cannot hide a local regression. Deterministic grouped folds select the affine feature set and regularization so each compatible target is both held out once and used by the other folds. Gazeebo finally refits the selected model on every compatible historical and current target. No five-target sequence becomes the model by itself.

For fixed feature, candidate, fold, context-expert, output, 3×3-region, and tail-histogram bounds, loading, grouped selection, scheduler reconstruction, and final fitting take linear work in the target count. Each completed target updates one fixed-size regional tail summary and the provisional model in constant amortized work; target selection scans only the fixed region and histogram tables. Gazeebo does not rescan prior targets per measurement or create a sample-count-sized kernel or pairwise target matrix.

The fixed-dimensional estimator family combines Bayesian weighted affine regression with a bounded nonlinear candidate: a deterministic frozen 48-unit head/face layer feeding the same Bayesian `(x, y)` output. Grouped folds choose whether the hidden layer helps. The all-display model also routes a bounded Bayesian expert for each output with enough evidence. Geometry-based soft weights blend neighboring output experts continuously, while the global model always retains at least 20 percent of the prediction and remains the full fallback for a sparse or added output. Positive target weights update posterior precision and feature-target information directly; the posterior mean produces `(x, y)`, while coefficient covariance, residual variance, and mixture disagreement quantify predictive uncertainty. Repeated compatible evidence contracts that uncertainty. When lower smoothing alpha bounds are configured, Gazeebo uses high uncertainty to slow bounded navigation without rejecting targets or erasing their influence. During active training, each camera observation produces an immediate raw estimate for measurement; only the displayed cursor receives the short position median, with no animated transit consuming the fixed window.

The default precision threshold is 100 logical pixels and one invocation stops early only when a five-target report meets both error gates, every authorized output cell has unseen evidence, no cell remains statistically higher than another, and every cell's CVaR90 surprise upper confidence bound is at or below 100 pixels. Equalizing all regions above the threshold is not success. It otherwise stops after at most 55 targets. The terminal panel reports the actual count and distinguishes accepted precision, a target-limit miss, interruption, and failure. Batch size, threshold, maximum, physical size, pixel fallback, training timings, and internal surprise bounds are finite and validated.

## Automatic training store

Users do not create or select profiles. Gazeebo keeps one automatic store below `$XDG_DATA_HOME/gazeebo`, or `$HOME/.local/share/gazeebo` when `XDG_DATA_HOME` is unset.

Every completed target remains in the store until reset. Each compact record contains median head/face and optional pupil features, per-feature robust dispersion, explicit pupil availability, posture and illumination context, the true output-relative circle center, monitor topology, a finite cursor-noise summary, and optional pre-incorporation radial error and posterior uncertainty. Legacy records remain model evidence but do not gain invented surprise measurements. Repeated camera, schema, output, and topology descriptors are interned; positional records avoid repeated keys; and the payload uses lossless standard compression. A 10,000-target regression fixture stays below 1 KiB per target on disk. The store also contains context statistics, fitted coefficients, and aggregate validation results.

The training store never contains frames, diagnostic images, video, raw frame-level landmarks, preparation observations, or per-frame cursor positions. Warning-triggered frames and detector metadata, when enabled, live only in the separate diagnostic archive described above. The training directory and file are owner-only, updates replace the complete state atomically, and decoding enforces compressed and expanded size limits.

Inspect it without camera or portal access:

```console
gazeebo training-stats
gazeebo dump-training
```

`training-stats` emits compact JSON size information. `dump-training` writes stable, schema-labelled, human-readable JSON to standard output without changing the store or creating a plaintext side file.

Stored targets, per-output experts, and validated model predictions remain bound to their source-output geometry. Gazeebo remaps them after unambiguous monitor resolution, scale, or position changes; excludes removed-output samples; and uses global/context fallback models for added, sparse, or ambiguous outputs. Weak topology matches are reported as inferred and unvalidated. Every result still passes through authorized-union clipping.

Reset all local training data:

```console
gazeebo reset-training
```

Run without reading or writing training state:

```console
gazeebo --ephemeral
```

## Debug HUD

```console
gazeebo --debug-hud
```

The opt-in HUD lists every authorized region, the active region, global cursor coordinates, selected model blend, topology quality, selected training cell, CVaR90, and surprise interval, region coverage, and aggregate noise/smoothing class. It is transparent, read-only, always on top, click-through, and limited to one update per second. It never displays stored feature values or trajectories.

## Privacy and lifecycle

Gazeebo has no telemetry, cloud client, or continuous recording path. Frames and frame-level derived values stay in memory and are discarded after processing except for default-enabled, warning-triggered diagnostic events. Those events remain local, owner-only, quota-bounded, and separate from the training store until `reset-diagnostics`. Only documented finite target-level aggregates enter the local training store. Every historical aggregate is retained until reset, so the compact private store grows with additional training.

One foreground process owns the camera, portal session, models, native surfaces, and runtime socket. Normal exit, errors, `SIGINT`, and `SIGTERM` release them through the same cleanup path.

## Development

The root flake is consumer-clean and exports the package, app, overlay, and NixOS module. Development tooling lives in the `dev` partition.

```console
nix develop
nix fmt
nix flake check --print-build-logs
```

See [`docs/architecture.md`](docs/architecture.md) for runtime boundaries and [`docs/training.md`](docs/training.md) for storage, topology adaptation, clustering, routing, retention, and candidate acceptance.
