# Singapore Embodied AI Challenge for Manufacturing

**ECM — technical repository and participant guide**

**Repository status:** Code, simulation assets, baseline policies, and datasets are pending upload. This README defines the proposed repository organization and documentation requirements for simulation platform collaborators. Paths below are proposed; replace all **TBD** entries and verify the examples against the released implementation before participant use.

## Background

The Singapore Embodied AI Challenge for Manufacturing advances embodied AI for industrial robotic manipulation, with a focus on assembly and semiconductor cleanroom automation. The challenge combines an online simulation benchmark with a subsequent physical robot evaluation to study how policies perform on realistic, sequential manufacturing tasks.

This repository will provide the simulation environment, robot-control interface, baseline policies, and evaluation tools needed to develop and assess participant policies.

## Task

The task is a chamber maintenance and component replacement sequence. A robot must open the chamber, extract an existing large component, pack it into a box, wipe the chamber interior, install a replacement component, and close the chamber.

The six subtasks are attempted in order. Each subtask has two action-specific scoring milestones: an intermediate achievement and completion of the required operation. Each milestone is worth 1 point, for a maximum of 2 points per subtask and **12 points per episode**. The milestones are described below; their exact detection criteria and numeric thresholds must be published with the evaluator.

| Subtask | Required operation | Preliminary completion condition | Available points |
| - | - | - | - |
| 1 | Lift off the chamber lid | The lid is lifted by at least a specified distance. | 1 for securely grasping the lid; 1 for lifting it clear of the chamber to the required distance. |
| 2 | Extract the large component | The original component is removed and moved a specified distance away from its initial position. | 1 for securely grasping the original component; 1 for extracting it and moving it the required distance from its initial position. |
| 3 | Pack the component into a box | The component is released into the box, contacts the box, and its center is below the box's highest point. | 1 for moving the held component above the box opening; 1 for releasing and successfully placing it inside the box. |
| 4 | Wipe the chamber interior | Contact is registered at all required sampling points: eight on the bottom, excluding the central support region, and three on each of the four side walls, for 20 points in total. | 1 for picking up the wiping tool; 1 for wiping all 20 required sampling points with the tool. |
| 5 | Unbox and place the replacement | The replacement component is within a specified distance of the original component's initial position. | 1 for picking up and lifting the replacement component out of its box; 1 for placing it at the required installation position. |
| 6 | Close and lock the chamber | The chamber lid is within a specified distance of its initial position; a locking-state check remains to be defined. | 1 for picking up the removed lid for reinstallation; 1 for replacing it in the required closed position and satisfying the finalized locking criterion. |

**Task details to publish:** robot embodiment and joint limits; chamber and component geometry; object masses and friction; box and wiping-tool specifications; initial conditions and randomization ranges; available sensors; intermediate and completion milestone criteria for each subtask; and annotated subtask illustrations.

The preliminary subtask 6 completion test checks lid position. A separate locking-state check, and any orientation or insertion checks needed for subtask 5, remain to be defined before the rules are finalized. Intermediate and completion checks must distinguish the two milestones in each subtask. For subtask 3, define the target region above the box opening and a successful release and placement; picking the component again is not a scoring requirement.

## Policy evaluation and scoring

### Subtask scoring and manual intervention

The evaluator records two milestones for each of the six subtasks:

- **Intermediate achievement: 1 point** when the robot achieves the first milestone specified for that subtask in the table, such as grasping the lid, moving the held component above the box, or unboxing the replacement.
- **Subtask completion: 1 point** when the robot achieves the second milestone specified for that subtask, such as lifting the lid clear, releasing the component into the box, or installing the replacement.

Each milestone can earn credit only once per episode. Repeating an action or satisfying the same milestone again does not earn additional points. The episode score is the sum of all credited milestones, from **0 to 12**.

Manual manipulation of objects is permitted under the official intervention procedure to skip a challenging part and allow the robot to continue with the remaining work. A failed or skipped milestone does not prevent later milestones or subtasks from earning credit. Manual manipulation itself earns no points: only milestones achieved by the robot are credited. If an intervention supplies a pick or other prerequisite, the robot may still earn the point for a subsequent operation it completes, provided the evaluator can distinguish that operation from the manually completed portion.

For example, if the component is manually moved above the box and the robot then releases it and successfully places it inside, subtask 3 earns 1 point for placement and 0 points for the manually supplied positioning milestone. Subtasks 4–6 remain eligible for credit. Achieving both milestones in all six subtasks autonomously earns 12 points.

The intervention procedure must define who may manipulate objects, when intervention is allowed, permitted object movements, how execution pauses and resumes, how policy state is handled, and how affected milestones are marked as skipped or still eligible. Interventions must be logged, and manually achieved conditions must not be mistaken for robot achievements.

The evaluator should report the episode score; intermediate and completion milestone outcomes and times for every subtask; elapsed episode time; intervention records; and termination reason. Milestone outcomes should distinguish robot-achieved, manually supplied or skipped, failed, and not attempted states. Across episodes, report mean score out of 12, per-milestone robot success rates, autonomous full-task success rate, and intervention frequency. Official aggregation and ranking rules are **TBD**.

### Policy evaluation

We use containerized policy evaluation to let teams manage their own software dependencies while keeping the evaluation process consistent and reproducible.

Teams submit a **Docker image** containing their policy, dependencies, and model weights. The policy runs as a service through a standardized interface:

- `health()`: report readiness.
- `reset(task_info)`: initialize a new episode and reset policy state.
- `predict(observation)`: return an action or an allowed action sequence.
- `close()`: release resources and shut down.

The official evaluator manages the robot or simulator, sends observations, validates and executes actions, and computes scores. Teams may choose their internal software stack, provided it meets the published hardware compatibility requirements, resource limits, and interface specification. Policy services do not directly control scoring or manual intervention.

All submissions must run **without external network access** and pass the official validation checks before evaluation. Observation formats, action definitions, timing constraints, episode reset rules, and failure handling are standardized across teams. Image digests, evaluation configurations, and execution logs are recorded for reproducibility.

### Evaluation specification to finalize

Simulation platform collaborators should publish the following alongside the evaluator:

- Exact intermediate and completion milestone criteria for every subtask, including valid grasps, lifting and extraction, positioning above the box, release and placement, tool pickup and wiping, unboxing and installation, and lid replacement and locking. Define how objects already held between subtasks are handled without requiring an unnecessary repeated pick.
- Numeric thresholds, units, reference frames, and exact comparison operators for all distance tests.
- Definitions of component center, box height, valid box contact, and valid wiping contact; identify which object or tool must make contact.
- Coordinates and tolerances of the 20 wiping points, whether contacts accumulate over time, and any required contact duration.
- Whether milestone success is latched or must remain satisfied; handling of dropped objects, collisions, invalid actions, resets, and timeouts.
- The manual intervention procedure, milestone eligibility after intervention, intervention logging, and whether intervention time counts toward episode limits.
- Episode time or step limit, number of evaluation episodes, seeds, scene variations, and permitted retries.
- Official score aggregation, tie-break rules, and the relationship between local validation and final evaluation.
- Permitted policy observations and control modes, including whether privileged simulator state is excluded.
- Policy service transport and message schemas for `health()`, `reset(task_info)`, `predict(observation)`, and `close()`, including readiness, error responses, and shutdown behavior.
- Hardware compatibility requirements, container runtime requirements, CPU/GPU and memory limits, and image submission and validation procedures.
- Standard observation and action definitions, allowed action sequences, timing and synchronization constraints, episode reset rules, intervention-resume behavior, and failure handling.

Keep the published rules, evaluator configuration, and implementation consistent and versioned together.

## Code usage

### Proposed repository structure

| Path | Purpose |
| - | - |
| `README.md` | Challenge overview, quick start, and links to technical documentation. |
| `pyproject.toml` or equivalent | Package metadata and dependency specification. |
| `src/ecm/envs/` | Simulation lifecycle, observations, reset logic, and environment wrappers. |
| `src/ecm/robots/` | Robot-control interface, control modes, limits, and sensor adapters. |
| `src/ecm/tasks/` | Task definitions, milestone logic, and scene randomization. |
| `src/ecm/evaluation/` | Milestone checks, intervention tracking, score aggregation, and result serialization. |
| `configs/` | Versioned environment, robot, policy, and evaluation configurations. |
| `scripts/` | Asset download, rollout, evaluation, and dataset utilities. |
| `examples/` | Minimal robot-control example, policy service example, and a complete policy rollout. |
| `baselines/` | Baseline policy adapters, container packaging, training and inference instructions. |
| `docs/` | Setup, API, task, evaluation, data, and submission documentation. |
| `tests/` | Installation checks, policy service validation, API checks, and evaluator boundary tests. |
| `assets/` and `data/` | Manifests and local download locations; large files hosted separately. |

Adapt the layout to the actual simulator and package. Keep policy inference separate from environment control and scoring so participants can integrate their own models through the standardized policy service.

### Installation and quick start

**TBD — provide tested commands after the code upload.**

Document supported operating systems, Python and simulator versions, GPU and driver requirements, minimum memory and storage, dependency installation, asset downloads, and any required licenses or access credentials. Pin a reproducible evaluator environment and document the required Docker runtime and hardware compatibility requirements for policy images.

The quick start must take a participant through installation, launching a scene, building a policy image, validating its service interface, and evaluating it locally without external network access. Include both interactive rendering and headless execution where supported, with expected outputs and common troubleshooting steps.

### Robot control and simulation API

**TBD — publish the actual API signatures and executable examples.**

The API guide must explain:

- How to construct and configure the environment, reset with a seed, advance the simulation, render, and release resources.
- Observation fields, tensor shapes, data types, units, coordinate frames, camera calibration, timestamps, and availability of tactile or force sensing.
- Action fields, joint ordering, supported control modes, position or velocity conventions, gripper commands, action limits, and control frequency.
- The standardized policy service methods `health()`, `reset(task_info)`, `predict(observation)`, and `close()`, including schemas, episode-state reset, allowed action sequences, and termination or truncation signals.
- Simulator time steps, sensor update rates, action duration, synchronization, and reproducibility limits.
- The evaluator's intervention pause and resume process, observation updates after object manipulation, and policy-state handling.

Provide a minimal control example and a complete observation–policy–action rollout using the released API. Distinguish participant-accessible observations from privileged state used internally for evaluation. Official evaluation must route policy actions through the evaluator for validation and execution.

### Evaluation script usage

**TBD — add the exact entry point, command-line options, and a tested invocation.**

The evaluation guide must specify how to select a policy Docker image by digest, configuration, asset version, seed set, episode count, device, rendering mode, and output directory. Model weights and runtime dependencies must be included in the image; evaluation must not depend on external downloads.

Document how to run official validation checks before evaluation, including service readiness and lifecycle checks, schema and action validation, hardware compatibility, resource limits, timing constraints, and operation without external network access. Explain the permitted manual intervention procedure and how to inspect milestone eligibility after resuming.

Provide a sample result file with episode-level intermediate and completion milestone outcomes, scores out of 12, timings, intervention records, termination reasons, and aggregate metrics. Record the policy image digest, code revision, simulator and asset versions, evaluation configuration, seeds, and execution logs with every run. Explain how to reproduce a run and where to find logs, videos, and failure diagnostics.

### Baselines and participant integration

Publish at least one working baseline with checkpoint access, preprocessing and normalization, required observations, training or fine-tuning instructions, inference requirements, and reproducible evaluation results. Candidate baselines include ACT, π0, and π0.5; availability and supported versions are **TBD**.

Provide a baseline Docker image and a reference policy service implementing `health()`, `reset(task_info)`, `predict(observation)`, and `close()`. Include tested build, validation, and local evaluation instructions. Teams may use their preferred internal software stack within the published hardware, resource, and interface requirements.

Finalize image submission procedures, dependency isolation, resource limits, allowed external training data, checkpoint loading, and the detailed service specification before participants prepare submissions. Submitted images must include all runtime dependencies and model weights, operate without external network access, and pass official validation before evaluation.

## Data release plan

**Recommended hosting:** a dedicated Hugging Face dataset repository for demonstrations and a model repository for baseline checkpoints. Keep large simulation assets outside Git and provide versioned manifests and download utilities. Participants must package required model weights and runtime dependencies into their Docker images before evaluation; hosting links are for preparation and distribution.

| Resource | Planned contents | Release trigger | Storage link |
| - | - | - | - |
| Simulation assets | Robot and scene assets, configuration files, licenses, and checksums. | With the first usable simulator release. | TBD |
| Starter demonstrations | Small sample dataset and loader examples for validating the pipeline. | With the quick-start release. | TBD — Hugging Face dataset repository |
| Training demonstrations | Versioned trajectories with observations, actions, episode metadata, intermediate and completion milestone annotations, and intervention records where applicable. | After schema and quality checks. | TBD — Hugging Face dataset repository |
| Public validation resources | Reproducible validation configurations and any designated public data. | With the evaluator release. | TBD |
| Baseline checkpoints | Weights, model cards, dependencies, and reference results. | With each baseline release. | TBD — Hugging Face model repository |

Release dates, dataset sizes, modalities, formats, and access conditions are **TBD**. Publish dataset cards covering provenance, licenses, schema, splits, normalization, calibration, limitations, and known issues. Provide download instructions, revision identifiers, checksums, and sample-loading code.

**Proposed split policy:** keep training and public validation resources separate; keep final evaluation seeds, scenarios, and any held-out data private until an approved release. Confirm the split policy before publishing data.

## Additional repository requirements

- **Licensing:** include a code license and separate licenses or usage terms for assets, datasets, and model weights.
- **Versioning:** maintain a changelog and compatibility matrix for code, simulator, assets, datasets, checkpoints, policy service interface, and evaluation rules.
- **Validation:** include checks that examples run and that the evaluator handles milestone scoring, manual interventions, threshold boundaries, and termination consistently. Verify the 12-point maximum, no duplicate credit, no credit for manually supplied milestones, and continued scoring after skipped milestones. Validate policy containers against the official service, hardware, resource, and network requirements.
- **Participant support:** provide troubleshooting, known limitations, issue templates, and a technical contribution guide.
- **Benchmark reference:** add task documentation and a citation entry when the benchmark reference is available.

## Simulation platform collaborators' preparation checklist

- [ ] Upload the simulator integration, task assets, robot interface, and pinned dependencies.
- [ ] Replace the proposed paths with the actual repository structure.
- [ ] Add tested installation, robot-control, policy container, rollout, and evaluation examples.
- [ ] Implement the six-subtask, 12-point evaluator with the action-specific intermediate and completion milestones listed in the task table.
- [ ] Finalize intermediate milestone criteria, completion thresholds, manual intervention procedures, aggregation, and termination rules.
- [ ] Verify that skipped milestones do not block subsequent robot-earned credit and manual manipulation earns no points.
- [ ] Publish API and sensor documentation, including observation and action schemas.
- [ ] Publish the Docker policy service specification, hardware compatibility requirements, resource limits, and official validation checks.
- [ ] Verify evaluation without external network access and record image digests, evaluation configurations, and execution logs.
- [ ] Release a working baseline image and reference evaluation results.
- [ ] Create the data and checkpoint repositories; replace storage-link placeholders.
- [ ] Publish dataset loaders, manifests, checksums, and licensing information.
- [ ] Document the participant policy and image submission interface.
- [ ] Verify the complete quick start from a clean environment and remove resolved TBD entries.
