# New session - 2026-06-30T17:51:30.609Z

**Session ID:** ses_0e657a8ceffeu35iLeLNqf49rN
**Created:** 6/30/2026, 12:51:30 PM
**Updated:** 6/30/2026, 3:31:00 PM

---

## User

how does it change Pointer Network and Reinforcement Learning agent on reports/03_proposals/fleet_routing/deferred/training.md knowing that now is did reports/03_proposals/fleet_routing/04_method.md

---

## Assistant (Build · DeepSeek V4 Flash Free · 2.2s)

_Thinking:_

The user wants to understand how the Pointer Network and Reinforcement Learning agent content changes in `reports/03_proposals/fleet_routing/deferred/training.md` given the updated `reports/03_proposals/fleet_routing/04_method.md`. Let me first read both files to understand the current state.

**Tool: read**

**Input:**
```json
{
  "filePath": "/home/vicbguti/Projects/vehicles/reports/03_proposals/fleet_routing/deferred/training.md"
}
```

**Output:**
```
<path>/home/vicbguti/Projects/vehicles/reports/03_proposals/fleet_routing/deferred/training.md</path>
<type>file</type>
<content>
1: # NP-Hard Fleet Logistics & Vehicle Distribution
2: 
3: ## How We Train Our Solution (The Study Phase)
4: The machine learning router is not trained on real roads. It learns inside a simulator using our **10-year historical SRI dataset**:
5: 1. **Simulator Setup**: We build a simulator of Ecuador's road network using canton coordinates from the Canton Catalog (`Catálogo_Cantones`).
6: 2. **Realistic Scenarios**: We feed the historical SRI dataset (4.3 million records) into the simulator. The simulator replays real historical sales weeks from 2017 to 2026 so the model faces the actual vehicle mixes and regional demand fluctuations of the Ecuadorian market.
7: 3. **The Learning Loop**: A Reinforcement Learning agent (Pointer Network) tries millions of routing combinations in the simulator:
8:    - When the agent groups heavy trucks randomly, the simulator gives it a **penalty** for overloading.
9:    - When the agent leaves trucks half-empty and uses more carriers than necessary, the simulator gives it a **penalty** for poor capacity utilization (or a **reward** for packing trucks close to 100% maximum capacity, minimizing total trucks rented).
10:    - When the agent routes a truck up and down the Andes unnecessarily, the simulator gives it a **penalty** for wasted fuel.
11:    - Over millions of iterations, the agent adjusts its neural network weights to maximize its efficiency rewards.

(End of file - total 11 lines)
</content>
```

**Tool: read**

**Input:**
```json
{
  "filePath": "/home/vicbguti/Projects/vehicles/reports/03_proposals/fleet_routing/04_method.md"
}
```

**Output:**
```
<path>/home/vicbguti/Projects/vehicles/reports/03_proposals/fleet_routing/04_method.md</path>
<type>file</type>
<content>
1: # Method
2: 
3: ## Problem formulation
4: 
5: **Capacitated fleet loading** — assign each vehicle to Truck 1, Truck 2, or **Defer**, subject to 6.0 CU per truck.
6: 
7: Formal constraints: [deferred/theory/2_generalization/3_partitioning/bin_packing.md](./deferred/theory/2_generalization/3_partitioning/bin_packing.md).
8: 
9: ---
10: 
11: ## 1. Label generation (teacher)
12: 
13: **Exhaustive search** on episodes with N ≤ ~20:
14: 
15: 1. Enumerate feasible assignments.
16: 2. Score: maximize vehicles loaded → CU utilization → minimize leftovers.
17: 3. Store best assignment as **label** per vehicle.
18: 
19: * Scenarios from real SRI weeks ([03_data.md](./03_data.md)).
20: * Labels are **automatic**, not hand-annotated.
21: * Solver is our code — not a black-box library.
22: 
23: ---
24: 
25: ## 2. Trainable model (student)
26: 
27: **Supervised imitation** in **PyTorch** (we implement architecture, loss, and training loop).
28: 
29: | Component | Choice |
30: |-----------|--------|
31: | Input | Variable-size set of vehicles (canton + class/CU) |
32: | Architecture | Per-vehicle classifier or attention over vehicles |
33: | Output | Truck A / Truck B / Defer |
34: | Loss | Cross-entropy vs teacher labels |
35: | Inference | **Capacity masking** — block assignments that exceed 6.0 CU |
36: 
37: Architecture must handle **variable N** (same weights for 12 or 40 vehicles).
38: 
39: ---
40: 
41: ## 3. Baselines
42: 
43: Implemented by us for comparison ([05_evaluation.md](./05_evaluation.md)):
44: 
45: * **Status quo** — regional proximity grouping ([example/problem/3_status_quo/status_quo.md](./example/problem/3_status_quo/status_quo.md))
46: * **Greedy** — first-fit / canton-order ([example/problem/2_greedy/greedy.md](./example/problem/2_greedy/greedy.md)); evaluate **grouping only**, not route km
47: 
48: ---
49: 
50: ## 4. Why learning (not just the teacher)
51: 
52: | Setting | Role of model |
53: |---------|----------------|
54: | Small N, same as training | Approximate teacher; mainly demonstrates learning |
55: | Large N at inference | Fast feasible assignment where exhaustive times out |
56: | Held-out SRI weeks | Generalize without re-running search |
57: 
58: See [02_scope.md](./02_scope.md) for honest limits on optimality claims.
59: 
60: ---
61: 
62: ## Planned modules
63: 
64: | Module | Role |
65: |--------|------|
66: | `src/loading/scenarios.py` | Weekly manifests from CSVs |
67: | `src/loading/labeler.py` | Exhaustive teacher |
68: | `src/loading/assigner.py` | PyTorch model |
69: | `scripts/train_loading.py` | Training |
70: | `scripts/eval_loading.py` | Metrics vs baselines |

(End of file - total 70 lines)
</content>
```

---

## Assistant (Build · DeepSeek V4 Flash Free · 8.5s)

_Thinking:_

Now I understand the situation. The user wants to know how the `training.md` file should change given the updated `04_method.md`. 

The key difference is that `04_method.md` has moved away from a Pointer Network / Reinforcement Learning approach to a **supervised imitation learning** approach with:
1. An **exhaustive search teacher** (labeler) that generates optimal assignments
2. A **trainable student model** (per-vehicle classifier or attention) that learns via cross-entropy loss
3. **Capacity masking** during inference
4. No RL, no Pointer Network, no simulator-based learning loop

The `training.md` file still describes the old approach with RL and Pointer Network. It needs to be updated to match the new method described in `04_method.md`.

Let me think about what the updated `training.md` should look like:

Old content:
- Simulator with 10-year historical SRI dataset
- Reinforcement Learning agent (Pointer Network)
- Penalties for overloading, poor capacity utilization, wasted fuel
- Millions of iterations adjusting neural network weights

New approach (from `04_method.md`):
- Supervised imitation learning
- Exhaustive search teacher generates labels for small N (≤20)
- Student model (classifier/attention) learns via cross-entropy
- Capacity masking during inference
- Works for variable N

Let me update the training.md to reflect the new method.

The `training.md` still describes the old **Pointer Network + RL + simulator** approach, while `04_method.md` now defines a **supervised imitation learning** pipeline (exhaustive search teacher → student classifier via cross-entropy, with capacity masking). The training doc needs to be updated to match.

**Tool: edit**

**Input:**
```json
{
  "filePath": "/home/vicbguti/Projects/vehicles/reports/03_proposals/fleet_routing/deferred/training.md",
  "oldString": "# NP-Hard Fleet Logistics & Vehicle Distribution\n\n## How We Train Our Solution (The Study Phase)\nThe machine learning router is not trained on real roads. It learns inside a simulator using our **10-year historical SRI dataset**:\n1. **Simulator Setup**: We build a simulator of Ecuador's road network using canton coordinates from the Canton Catalog (`Catálogo_Cantones`).\n2. **Realistic Scenarios**: We feed the historical SRI dataset (4.3 million records) into the simulator. The simulator replays real historical sales weeks from 2017 to 2026 so the model faces the actual vehicle mixes and regional demand fluctuations of the Ecuadorian market.\n3. **The Learning Loop**: A Reinforcement Learning agent (Pointer Network) tries millions of routing combinations in the simulator:\n   - When the agent groups heavy trucks randomly, the simulator gives it a **penalty** for overloading.\n   - When the agent leaves trucks half-empty and uses more carriers than necessary, the simulator gives it a **penalty** for poor capacity utilization (or a **reward** for packing trucks close to 100% maximum capacity, minimizing total trucks rented).\n   - When the agent routes a truck up and down the Andes unnecessarily, the simulator gives it a **penalty** for wasted fuel.\n   - Over millions of iterations, the agent adjusts its neural network weights to maximize its efficiency rewards.",
  "newString": "# NP-Hard Fleet Logistics & Vehicle Distribution\n\n## How We Train Our Solution (The Study Phase)\n\nThe model is trained via **supervised imitation learning** — no RL, no simulator loop.\n\n### 1. Teacher: exhaustive search (labeler)\nFor training episodes with **N ≤ ~20 vehicles**:\n1. Enumerate all feasible truck-A / truck-B / defer assignments.\n2. Score each assignment: maximize vehicles loaded → CU utilization → minimize deferred.\n3. Store the best assignment as the **label** for each vehicle.\n\nScenarios are built from real SRI weeks ([04_method.md](../04_method.md)).\n\n### 2. Student: trainable classifier\nThe student is a PyTorch model that predicts Truck A / Truck B / Defer:\n\n| Component | Choice |\n|-----------|--------|\n| Input | Variable-size set of vehicles (canton + class/CU) |\n| Architecture | Per-vehicle classifier or attention over vehicles |\n| Output | Truck A / Truck B / Defer |\n| Loss | Cross-entropy vs teacher labels |\n| Inference | **Capacity masking** — block assignments exceeding 6.0 CU |\n\nThe architecture handles **variable N** (same weights for 12 or 40 vehicles).\n\n### 3. Why this approach\n- **Small N** — the student approximates the teacher; demonstrates learning.\n- **Large N at inference** — produces fast feasible assignments where exhaustive search times out.\n- **Held-out weeks** — generalizes to unseen SRI periods without re-running search."
}
```

**Output:**
```
Edit applied successfully.
```

---

## Assistant (Build · DeepSeek V4 Flash Free · 2.6s)

_Thinking:_

The edit was applied successfully. The training.md now matches the supervised imitation learning approach described in 04_method.md, replacing the old Pointer Network + RL + simulator content.

Updated `training.md` now reflects the **supervised imitation learning** approach:

- **Removed** the RL agent, Pointer Network, simulator loop, and penalty/reward mechanics.
- **Added** the two-stage teacher-student pipeline: exhaustive search labeler (teacher) → PyTorch classifier (student) with cross-entropy loss and capacity masking.
- **Aligned** with the architecture, inputs, and inference strategy from `04_method.md`.

---

## User

what do you suggest to consider when using formal modeling language to describe only the problem, where is not mixed solution design or solution implementation. text description is not formal language of engineering modeling, where if uml is used use-case diagram is not enough

---

## Assistant (Plan · DeepSeek V4 Flash Free · 21.1s)

_Thinking:_

The user is asking a question about formal modeling languages for problem description (not solution design or implementation). They want to know what I suggest to consider when using a formal modeling language to describe only the problem, where the text description isn't formal enough, and UML use-case diagrams alone aren't sufficient.

This seems like a conceptual/architectural question related to their project. They're likely thinking about how to formally describe the fleet routing problem without mixing in solution design details.

Let me think about this carefully:

1. They want to describe ONLY the problem domain - not the solution design, not the implementation
2. Text descriptions are not formal enough for engineering modeling
3. UML use-case diagrams alone are insufficient

This is about **domain modeling** or **problem framing** using formal languages. The key distinction is between problem space and solution space (as in Domain-Driven Design or Problem Frames approach).

Let me consider what formal modeling approaches could work for describing just the problem:

1. **Problem Frames** (Michael Jackson) - This is specifically designed to describe problems, not solutions. It focuses on the real-world phenomena, domains, and requirements.

2. **KAOS (Knowledge Acquisition in autOmated Specification)** - Goal-oriented requirements engineering that can formally describe goals, obstacles, and domain properties without solution bias.

3. **Alloy** - A formal specification language that can model structural constraints of the problem domain. It's good for describing what must hold true.

4. **Z Notation** / **VDM** - Formal specification languages that can describe system state and operations abstractly.

5. **i* (i-Star)** - Agent-oriented modeling for describing strategic dependencies between actors in the problem domain.

6. **System Dynamics / Causal Loop Diagrams** - For describing the behavioral aspects of the problem.

7. **Object-Process Methodology (OPM)** - Can describe both structure and behavior of the problem domain.

8. **BPMN (Business Process Model and Notation)** - If the problem involves business processes.

9. **Formal ontologies (OWL, Description Logics)** - For describing the domain concepts and their relationships.

10. **Constraint Satisfaction Problem (CSP) formulation** - For formally describing the constraints of the problem without prescribing the solving method.

The key points I should emphasize:

- **Problem vs. Solution distinction**: The language must be capable of expressing only domain phenomena, not design constructs
- **Semantic formality**: Text is ambiguous; the modeling language needs formal semantics
- **Use-case limitations**: Use cases describe interactions but not static structure, constraints, or non-functional aspects
- **Multiple views needed**: Structural (what exists), Functional (what must be achieved), Behavioral (how things behave in the domain)

Since I'm in plan mode and can't make edits, I should just provide my analysis and suggestions to the user. Let me also look at what's in the project to understand the context better - they mentioned "the problem" which likely refers to the fleet routing problem described in the deferred directory.

Actually, the user is asking a general conceptual question. Let me just answer directly.

Great question — you're drawing the key **problem-vs-solution** boundary. Here's what I'd consider for a formal problem-only description:

## What to look for in a problem-only formal language

| Requirement | Why it matters |
|---|---|
| **No algorithmic/architectural vocabulary** | The language must not let you say "neural network," "classifier," "labeler," or "search" — those are solution constructs. |
| **Constraints over phenomena** | You state *what must hold* in the domain, not *how to achieve it*. |
| **Multiple views (structural + behavioral + goal)** | Use-cases cover one; you also need static structure (domain concepts, relations) and goals/obstacles. |

## Languages that fit (problem-only)

- **Problem Frames** (Jackson) — designed deliberately to describe problems, not solutions. You identify *domains* (causal, biddable, lexical), their *phenomena*, and the *requirement* constraining them. No design concepts leak in.
- **Alloy** — formal, declarative, constraint-based. You write facts about the domain (`Capacity`, `Vehicle`, `Truck`, `Defer`), then assertions about what must be true. No notion of "solver" or "algorithm" enters the model.
- **KAOS / Goal models** — captures goals, obstacles, domain properties, and agents (in the problem sense). The goal "Maximize CU utilization" is stated as a formal objective; how it's computed is separate.
- **OWL / Description Logic** — formal ontology for domain concepts and their constraints. Good for declarative classification without solution bias.

## What a complete problem model needs (that use-cases miss)

| Aspect | Formalization |
|---|---|
| **Domain structure** | Classes: `Canton`, `VehicleClass`, `Truck`, `Assignment`; relations: `hasCapacity`, `assignedTo`, `deferred` |
| **Invariants** | `sum(CU of assigned vehicles) ≤ 6.0` per truck, `a vehicle is in exactly one state (truck-A / truck-B / defer)` |
| **Goals** | Primary: load as many vehicles as possible. Secondary: minimize wasted CU. Tertiary: minimize deferred. |
| **Domain assumptions** | Cantons have fixed coordinates; each vehicle has a known CU; trucks have identical 6.0 capacity. |
| **Quality constraints** | Assignment must be computable within seconds for N=40 (stated as a requirement on the domain, not on any solver). |

## Structure I'd suggest for a problem-only document

```
01_problem/
├── domain_model.md         # Static structure (classes, relations, invariants) in prose + Alloy/Z snippets
├── goals.md                # Hierarchical goals, obstacles, domain properties
├── constraints.md          # Formal capacity, assignment, routing constraints
└── glossary.md             # Uniform vocabulary to prevent solution-terms from creeping in
```

The litmus test: if the document never mentions "model," "train," "inference," "algorithm," or "search" (except as domain phenomena), it's a pure problem description. The moment you say "exhaustive enumeration" or "classifier," you've crossed into solution space.

---

## User

what if i didn't hear before about how to build any of the languages that you mentioned

---

## Assistant (Plan · DeepSeek V4 Flash Free · 13.7s)

_Thinking:_

The user is saying they haven't heard of or don't know how to build models in the languages I mentioned (Problem Frames, Alloy, KAOS, OWL). They want to understand how to get started with these formal modeling approaches. I'm in plan/read-only mode, so I should provide guidance and resources, not make any changes.

Let me provide practical, actionable advice for learning these languages, tailored to their context (fleet routing problem).

Understood. Here's a practical starting point for each, ordered from least to most investment:

---

## 1. Alloy — best first step (closest to what you already think in)

Alloy is a **declarative constraint language**. You write facts; it finds instances or counterexamples automatically.

**What to read/write to learn it:**

- **Tutorial (30 min):** [Alloy in 5 minutes](https://alloytools.org/alloy/documentation.html) → then do the first 2 exercises from the [Alloy Quick Guide](https://alloytools.org/tutorials/online/).
- **Your fleet problem in Alloy (~50 lines):**

```alloy
sig Vehicle { cu: one Int, canton: one Canton }
sig Canton {}
sig Truck { load: set Vehicle, capacity: Int }

fact noOverload { all t: Truck | sum v: t.load | v.cu <= t.capacity }
fact atMostOneTruck { all v: Vehicle | lone t: Truck | v in t.load }

run {} for 5 but 2 Truck, 10 Vehicle
```

You write the *structure* and *constraints* — Alloy checks if any assignment satisfies them. No algorithm, no solution design.

**Tool:** [Alloy Analyzer](https://alloytools.org/) — desktop GUI with visualizer.

---

## 2. Problem Frames — no code, just structure on paper

Jackson's Problem Frames use **diagrams + tables** — not a programming language. The book is *Problem Frames: Analysing and Structuring Software Development Problems* (Jackson 2001).

**The minimal setup:**

| Element | What you write |
|---|---|
| **Domain diagram** | Box for each real-world thing (Vehicle, Truck, Canton), labeled with domain type (Causal, Biddable, Lexical) |
| **Phenomena table** | List what each domain shares with the machine (e.g., Vehicle!reportsCU, Truck!maxCapacity) |
| **Requirement** | A single sentence: "The assignment shall maximize loaded vehicles subject to ≤6.0 CU per truck" |

**Zero setup cost** — paper, whiteboard, or draw.io.

---

## 3. KAOS / Goal models — formal goals, not algorithms

If you want to write *goals* formally (e.g., "achieve[LoadedVehicles ≥ 95%]"), KAOS uses:

- **Goal diagrams** (AND/OR refinement trees)
- **Formal specification** using Linear Temporal Logic (LTL): `◇ (assignment_complete ∧ utilization ≥ threshold)`

**Learn in 1 hour:** Read [KAOS: An Industrial Requirements Engineering Approach](https://www.researchgate.net/publication/220833820_KAOS_An_Industrial_Requirements_Engineering_Approach) — focus on §3 (Goal Modeling).

**Tool:** [Objectiver](http://www.objectiver.com/) (commercial) or draw.io with a stencil.

---

## Summary: pick one path

| If you want | Start with |
|---|---|
| Code that runs | **Alloy** — install now, write the snippet above in 10 min |
| Just conceptual rigor | **Problem Frames** — no tools needed |
| Goal-oriented thinking | **KAOS** — read the LTL syntax, draw a goal tree |

**My recommendation for your fleet problem:** start with **Alloy**. It takes 30 minutes to learn enough to model the loading constraints, and it gives you instant feedback (the analyzer finds missing constraints). Once you have the Alloy model, the other languages become easier because the *thinking* is the same — only the notation changes.

Want me to walk through how that Alloy snippet maps to your domain?

---

## User

can you give me an example for problem frames?

---

## Assistant (Plan · DeepSeek V4 Flash Free · 12.3s)

_Thinking:_

The user wants an example of Problem Frames applied to their fleet domain. I'm in plan/read-only mode, so I'll just describe/explain — no file edits.

Here's a Problem Frames model for your fleet loading problem — no code, just domain structure and requirements.

---

## Domain diagram

```
┌──────────────────────┐     ┌──────────────────────┐
│    Vehicle Schedule   │     │     Truck Fleet       │
│   (biddable domain)   │     │   (causal domain)     │
│                       │     │                       │
│  phenomena:           │     │  phenomena:           │
│  - Vehicle.cu         │     │  - Truck.capacity     │
│  - Vehicle.canton     │────▶│  - Truck.assignedLoad │
│  - Vehicle.id         │     │  - Truck.route        │
└──────────────────────┘     └──────────────────────┘
         │                            │
         │                            │
         ▼                            ▼
┌──────────────────────────────────────────────┐
│           Loading Assignment Machine          │
│              (designed domain)                │
│                                               │
│  reads: VehicleSchedule                        │
│  controls: Truck.assignedLoad                 │
│  observes: Truck.capacity                     │
│  must satisfy: LoadingRequirement             │
└──────────────────────────────────────────────┘
```

---

## Domain types

| Domain | Type | Why |
|---|---|---|
| Vehicle Schedule | **Biddable** | You don't control which vehicles arrive; they come from SRI data (external). |
| Truck Fleet | **Causal** | Trucks have fixed properties (capacity) but you control their loading. |
| Loading Assignment Machine | **Designed** | The thing you build. |

---

## Phenomena table

| Shared phenomenon | Vehicle Schedule | Truck Fleet | Machine |
|---|---|---|---|
| `Vehicle.cu` | produces | — | reads |
| `Vehicle.canton` | produces | — | reads |
| `Vehicle.id` | produces | — | reads |
| `Truck.capacity` | — | has | observes |
| `Truck.assignedLoad` | — | controlled by | writes |
| `Truck.assignedDefer` | — | controlled by | writes |

---

## Requirement (formal statement)

```
LoadingRequirement:

   For all t: Truck:
       sum(v.cu for v in t.assignedLoad) ≤ t.capacity

   For all v: Vehicle:
       v is in exactly one of:
           assignedTo(Truck1), assignedTo(Truck2), deferred

   Goal: maximize |{v: v in assignedTo(Truck1) ∪ assignedTo(Truck2)}|
   then: minimize sum(t.capacity - used(t) for t in Truck)
   then: minimize |{v: v in deferred}|
```

---

## What makes this "problem-only"

- No mention of **how** the machine computes the assignment — no search, no model, no algorithm.
- The requirement is a constraint over domain phenomena (`Vehicle.cu`, `Truck.capacity`, `assignedLoad`).
- The machine is a black box from the problem perspective: it reads from Vehicle Schedule and writes to Truck Fleet.
- The "designed domain" box exists, but its internals are not specified — that's the solution space.

---

The next step would be decomposing the requirement into subproblems (e.g., capacity constraint vs. optimization goal), which Problem Frames handles via **requirement progression**. Want me to show that too?

---

## User

ok

---

## Assistant (Plan · DeepSeek V4 Flash Free · 13.8s)

_Thinking:_

The user said "ok" to my offer to show requirement progression for the Problem Frames model. I'm in plan mode (read-only), so I can only explain/demonstrate concepts, not modify files. Let me explain requirement progression for the fleet loading problem.

Here's how **requirement progression** decomposes the top-level requirement into subproblems that match real domain chunks.

---

## Top-level requirement (from before)

```
LoadingRequirement:
  assign every vehicle to Truck1, Truck2, or Defer
  subject to capacity constraints
  maximizing loaded vehicles → CU utilization → minimizing deferred
```

---

## Step 1: Break on independent constraints

```
Subproblem A — Capacity constraint
  Reference: VehicleSchedule + TruckFleet
  Requirement: for each truck, sum(v.cu) ≤ 6.0
  Machine reads Vehicle.cu, observes Truck.capacity
  Machine controls Truck.assignedLoad

Subproblem B — Optimization objective
  Reference: VehicleSchedule + TruckFleet
  Requirement: maximize loaded vehicles, then CU fill, then minimize deferred
  This subproblem CONTAINS subproblem A as a constraint
```

Subproblem B *progresses* the requirement into A because optimization is meaningless without feasibility. In Problem Frames notation:

```
LoadingRequirement  ──progresses via──►  CapacityConstraint (Sub A)
                                         OptimizationGoal    (Sub B, contains A)
```

---

## Step 2: Identify machine-to-machine dependencies

Subproblem B needs the *output* of A (feasible assignments) as its *input*. In PF terms, B's machine reads A's phenomena.

```
┌──────────────────────┐      ┌──────────────────────┐
│   Cap. Machine (A)   │────► │   Opt. Machine (B)   │
│                      │      │                      │
│  writes:             │      │  reads:              │
│  - feasibleSets      │      │  - feasibleSets      │
│  - overloadedSets    │      │  - scoringFunction   │
└──────────────────────┘      └──────────────────────┘
```

This is a **required behavior interjection** — B needs to *observe* A's output to fulfill its own requirement.

---

## Step 3: Defer independent subproblems to real-world domains

If some vehicles are **unassignable** by their nature (e.g., special cargo), that's not a machine concern — it's a TruckFleet domain property:

```
Subproblem C — Deferred handling
  Reference: VehicleSchedule + TruckFleet
  Requirement: vehicles marked "special" in TruckFleet.domainProperties
               are always deferred
  Machine only reads; does not decide
```

This is a **domain assumption**, not a machine behavior. In PF, you document it as a *given*.

---

## Resulting problem diagram

```
                          LoadingRequirement
                                │
                    ┌───────────┴───────────┐
                    │                       │
            CapacityConstraint         OptimizationGoal
                    │                       │
                    ▼                       ▼
            ┌──────────────┐       ┌──────────────────┐
            │  Cap Machine │────►  │   Opt Machine    │
            │  (feasible)  │       │  (scoring/rank)  │
            └──────────────┘       └──────────────────┘
                    │                       │
                    ▼                       ▼
            ┌──────────────────────────────────────┐
            │         TruckFleet (causal)           │
            │  - assignedLoad (from Opt Machine)    │
            │  - deferred (from domain assumption)  │
            └──────────────────────────────────────┘
```

---

This decomposition keeps the **problem** description formal (constraints, domain phenomena, requirements) while the **how** (search, model, algorithm) stays outside the frame. The machine boxes have no internal design — they're just placeholders for phenomena transformations.

---

