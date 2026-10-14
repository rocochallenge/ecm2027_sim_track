# Singapore Embodied AI Challenge for Manufacturing

**SGEAIC — technical repository and participant guide**

**Repository status:** Code, simulation assets, baseline policies, and datasets are pending upload. This README defines the proposed repository organization and documentation requirements for LightWheel collaborators. Paths below are proposed; replace all **TBD** entries and verify the examples against the released implementation before participant use.

## Background

The Singapore Embodied AI Challenge for Manufacturing advances embodied AI for industrial robotic manipulation, with a focus on assembly and semiconductor cleanroom automation. The challenge combines an online simulation benchmark with a subsequent physical robot evaluation to study how policies perform on realistic, sequential manufacturing tasks.

This repository will provide the simulation environment, robot-control interface, baseline policies, and evaluation tools needed to develop and assess participant policies.

## Task

The task is a chamber maintenance and component replacement sequence. A robot must open the chamber, extract an existing large component, pack it into a box, wipe the chamber interior, install a replacement component, and close the chamber.

The six stages are evaluated in order:

| Stage | Required operation | Preliminary pass condition |
| - | - | - |
| 1 | Lift off the chamber lid | The lid is lifted by at least a specified distance. |
| 2 | Extract the large component | The original component is removed and moved a specified distance away from its initial position. |
| 3 | Pack the component into a box | The component contacts the box and its center is below the box's highest point. |
| 4 | Wipe the chamber interior | Contact is registered at all required sampling points: eight on the bottom, excluding the central support region, and three on each of the four side walls, for 20 points in total. |
| 5 | Unbox and place the replacement | The replacement component is within a specified distance of the original component's initial position. |
| 6 | Close and lock the chamber | The chamber lid is within a specified distance of its initial position. |

**Task details to publish:** robot embodiment and joint limits; chamber and component geometry; object masses and friction; box and wiping-tool specifications; initial conditions and randomization ranges; available sensors; and annotated stage illustrations.

The preliminary stage 6 test checks lid position. A separate locking-state check, and any orientation or insertion checks needed for stage 5, remain to be defined before the rules are finalized.

## Policy evaluation and scoring

### Sequential stage evaluation

A stage is evaluated only after the preceding stage has passed. Later-stage completion cannot compensate for an earlier failed stage.

**Proposed score representation, subject to confirmation:** the episode score is the number of consecutive stages passed, from 0 to 6. For example, passing stages 1–3 and failing stage 4 gives a score of 3; stages 5–6 contribute no credit.

The evaluator should report the episode score, pass/fail status and completion time for each stage, elapsed episode time, and termination reason. Across episodes, report mean completed stages, full-task success rate, and the fraction of all episodes reaching each stage. Official aggregation and ranking rules are **TBD**.

### Evaluation specification to finalize

LightWheel collaborators should publish the following alongside the evaluator:

- Numeric thresholds, units, reference frames, and exact comparison operators for all distance tests.

- Definitions of component center, box height, valid box contact, and valid wiping contact; identify which object or tool must make contact.

- Coordinates and tolerances of the 20 wiping points, whether contacts accumulate over time, and any required contact duration.

- Whether stage success is latched or must remain satisfied; handling of dropped objects, collisions, invalid actions, resets, and timeouts.

- Episode time or step limit, number of evaluation episodes, seeds, scene variations, and permitted retries.

- Official score aggregation, tie-break rules, and the relationship between local validation and final evaluation.

- Permitted policy observations and control modes, including whether privileged simulator state is excluded.

Keep the published rules, evaluator configuration, and implementation consistent and versioned together.

## Code usage

### Proposed repository structure

| Path | Purpose |
| - | - |
| `README.md` | Challenge overview, quick start, and links to technical documentation. |
| `pyproject.toml` or equivalent | Package metadata and dependency specification. |
| `src/sgeaic/envs/` | Simulation lifecycle, observations, reset logic, and environment wrappers. |
| `src/sgeaic/robots/` | Robot-control interface, control modes, limits, and sensor adapters. |
| `src/sgeaic/tasks/` | Task definitions, stage logic, and scene randomization. |
| `src/sgeaic/evaluation/` | Stage checks, score aggregation, and result serialization. |
| `configs/` | Versioned environment, robot, policy, and evaluation configurations. |
| `scripts/` | Asset download, rollout, evaluation, and dataset utilities. |
| `examples/` | Minimal robot-control example and a complete policy rollout. |
| `baselines/` | Baseline policy adapters, training and inference instructions. |
| `docs/` | Setup, API, task, evaluation, data, and submission documentation. |
| `tests/` | Installation checks, API checks, and evaluator boundary tests. |
| `assets/` and `data/` | Manifests and local download locations; large files hosted separately. |

Adapt the layout to the actual simulator and package. Keep policy inference separate from environment control and scoring so participants can integrate their own models.

### Installation and quick start

**TBD — provide tested commands after the code upload.**

Document supported operating systems, Python and simulator versions, GPU and driver requirements, minimum memory and storage, dependency installation, asset downloads, and any required licenses or access credentials. Pin a reproducible environment and provide a container if supported.

The quick start must take a participant through installation, launching a scene, running one example policy, and evaluating it locally. Include both interactive rendering and headless execution where supported, with expected outputs and common troubleshooting steps.

### Robot control and simulation API

**TBD — publish the actual API signatures and executable examples.**

The API guide must explain:

- How to construct and configure the environment, reset with a seed, advance the simulation, render, and release resources.

- Observation fields, tensor shapes, data types, units, coordinate frames, camera calibration, timestamps, and availability of tactile or force sensing.

- Action fields, joint ordering, supported control modes, position or velocity conventions, gripper commands, action limits, and control frequency.

- Policy initialization and inference interface, episode-state reset, action batching if supported, and termination or truncation signals.

- Simulator time steps, sensor update rates, action duration, synchronization, and reproducibility limits.

Provide a minimal control example and a complete observation–policy–action rollout using the released API. Distinguish participant-accessible observations from privileged state used internally for evaluation.

### Evaluation script usage

**TBD — add the exact entry point, command-line options, and a tested invocation.**

The evaluation guide must specify how to select a policy or checkpoint, configuration, asset version, seed set, episode count, device, rendering mode, and output directory.

Provide a sample result file with episode-level stage outcomes, scores, timings, termination reasons, and aggregate metrics. Record code revision, simulator and asset versions, evaluation configuration, and seeds with every run. Explain how to reproduce a run and where to find logs, videos, and failure diagnostics.

### Baselines and participant integration

Publish at least one working baseline with checkpoint access, preprocessing and normalization, required observations, training or fine-tuning instructions, inference requirements, and reproducible evaluation results. Candidate baselines include ACT, π0, and π0.5; availability and supported versions are **TBD**.

Define the policy packaging or submission interface, dependency isolation, resource limits, allowed external data, checkpoint loading, and any network-access restrictions. These requirements must be finalized before participants prepare submissions.

## Data release plan

**Recommended hosting:** a dedicated Hugging Face dataset repository for demonstrations and a model repository for baseline checkpoints. Keep large simulation assets outside Git and provide versioned manifests and download utilities.

| Resource | Planned contents | Release trigger | Storage link |
| - | - | - | - |
| Simulation assets | Robot and scene assets, configuration files, licenses, and checksums. | With the first usable simulator release. | TBD |
| Starter demonstrations | Small sample dataset and loader examples for validating the pipeline. | With the quick-start release. | TBD — Hugging Face dataset repository |
| Training demonstrations | Versioned trajectories with observations, actions, episode metadata, and stage annotations. | After schema and quality checks. | TBD — Hugging Face dataset repository |
| Public validation resources | Reproducible validation configurations and any designated public data. | With the evaluator release. | TBD |
| Baseline checkpoints | Weights, model cards, dependencies, and reference results. | With each baseline release. | TBD — Hugging Face model repository |

Release dates, dataset sizes, modalities, formats, and access conditions are **TBD**. Publish dataset cards covering provenance, licenses, schema, splits, normalization, calibration, limitations, and known issues. Provide download instructions, revision identifiers, checksums, and sample-loading code.

**Proposed split policy:** keep training and public validation resources separate; keep final evaluation seeds, scenarios, and any held-out data private until an approved release. Confirm the split policy before publishing data.

## Additional repository requirements

- **Licensing:** include a code license and separate licenses or usage terms for assets, datasets, and model weights.

- **Versioning:** maintain a changelog and compatibility matrix for code, simulator, assets, datasets, checkpoints, and evaluation rules.

- **Validation:** include checks that examples run and that the evaluator handles stage order, threshold boundaries, and termination consistently.

- **Participant support:** provide troubleshooting, known limitations, issue templates, and a technical contribution guide.

- **Benchmark reference:** add task documentation and a citation entry when the benchmark reference is available.

## LightWheel preparation checklist

- [ ] Upload the simulator integration, task assets, robot interface, and pinned dependencies.

- [ ] Replace the proposed paths with the actual repository structure.

- [ ] Add tested installation, robot-control, rollout, and evaluation examples.

- [ ] Finalize the six-stage evaluator, thresholds, aggregation, and termination rules.

- [ ] Publish API and sensor documentation, including observation and action schemas.

- [ ] Release a working baseline and reference evaluation results.

- [ ] Create the data and checkpoint repositories; replace storage-link placeholders.

- [ ] Publish dataset loaders, manifests, checksums, and licensing information.

- [ ] Document the participant policy and submission interface.

- [ ] Verify the complete quick start from a clean environment and remove resolved TBD entries.
