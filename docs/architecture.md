# Architecture

This document maps the main Python classes and the dynamic assembly workflows.
It is intentionally selective: the diagrams show the classes and methods that
define ownership boundaries and matrix assembly, not every available helper.

## Class Structure

```mermaid
classDiagram
    class Drivetrain {
        <<abstract>>
        +stage
        +main_shaft
        +dynamic_model
        +f_n
        +mode_shape
    }

    class NREL_5MW {
        +gear_set(idx)
        +shaft(idx)
        +bearing(idx)
    }

    class GearSet {
        +configuration
        +N_p
        +bearing
        +output_shaft
        +sub_set(option)
    }

    class Gear {
        +z
        +d
        +d_b
        +mass
        +J_x
    }

    class Carrier {
        +mass
        +J_x
        +a_w
    }

    class Shaft {
        +inertia_matrix(option)
        +stiffness_matrix(option)
        +damping_matrix(option)
    }

    class Bearing {
        +parallel_association()
        +series_association()
        +stiffness_matrix()
        +damping_matrix()
    }

    class DynamicModel {
        +modal_analysis()
        +state_matrix()
        +time_response()
        +frf()
    }

    class Kahraman_94 {
        +stage_inertia_matrix(stage)
        +stage_stiffness_matrix(stage)
        +stage_damping_matrix(stage)
        +fixed_ring_planetary_frequencies(stage)
    }

    class Lin_Parker_99 {
        +raw_stage_inertia_matrix(stage)
        +raw_stage_gyroscopic_matrix(stage)
        +raw_stage_stiffness_matrix(stage)
        +stage_coordinate_change(stage)
        +stage_inertia_matrix(stage)
        +stage_gyroscopic_matrix(stage)
        +stage_stiffness_matrix(stage)
    }

    class torsional_2DOF {
        +M
        +K
        +f_n
    }

    Drivetrain <|-- NREL_5MW
    DynamicModel <|-- Kahraman_94
    DynamicModel <|-- Lin_Parker_99
    DynamicModel <|-- torsional_2DOF
    Gear <|-- GearSet
    Drivetrain "1" o-- "many" GearSet : stages
    Drivetrain "1" o-- "1" Shaft : main_shaft
    GearSet "1" o-- "0..1" Carrier
    GearSet "1" o-- "many" Bearing
    GearSet "1" o-- "1" Shaft : output_shaft
```

## Model Construction Flow

`NREL_5MW` is a concrete drivetrain definition. It builds the component data,
then delegates dynamic matrices and modal results to the selected dynamic model.

```mermaid
sequenceDiagram
    participant User
    participant DT as NREL_5MW
    participant GS as GearSet
    participant Shaft
    participant DM as DynamicModel

    User->>DT: NREL_5MW(dynamic_model=Lin_Parker_99)
    DT->>GS: gear_set(0..N_st-1)
    DT->>Shaft: shaft(-1), shaft(stage)
    DT->>DM: selected dynamic model(DT)
    DM->>DM: assemble M, K, D, G as supported
    DM->>DM: modal_analysis()
    DM-->>DT: f_n, mode_shape, matrices
```

## LP99 Stage Assembly

`Lin_Parker_99` keeps two levels of stage matrix assembly:

- raw helpers mirror the MATLAB/paper isolated-stage matrices;
- public stage helpers map those matrices into the assembled drivetrain
  coordinates and append the output interface.

```mermaid
flowchart TD
    A["stage"] --> B["raw_stage_inertia_matrix(stage)"]
    A --> C["raw_stage_gyroscopic_matrix(stage)"]
    A --> D["raw_stage_stiffness_matrix(stage)"]
    A --> E["stage_coordinate_change(stage)"]

    B --> F["R.T @ M_raw @ R"]
    C --> G["R.T @ G_raw @ R"]
    D --> H["R.T @ K_raw @ R"]
    E --> F
    E --> G
    E --> H

    F --> I["append output interface"]
    G --> I
    H --> I
    I --> J["insert stage block into global matrices"]
```

## Kahraman 1994 Stage Assembly

`Kahraman_94` is a reduced torsional formulation. A planetary stage is ordered
as carrier, planets, sun, output shaft. A parallel stage is ordered as wheel,
pinion, output shaft.

```mermaid
flowchart TD
    A["stage"] --> B["stage_inertia_matrix(stage)"]
    A --> C["stage_stiffness_matrix(stage)"]
    A --> D["stage_damping_matrix(stage)"]
    A --> E["fixed_ring_planetary_frequencies(stage)"]

    B --> F["global M assembly"]
    C --> G["global K assembly"]
    D --> H["global D assembly"]
    E --> I["reference validation tests"]
```

## Matrix Ownership

```mermaid
flowchart LR
    subgraph Components
        Shaft["Shaft matrices"]
        Bearing["Bearing stiffness/damping"]
        GearSet["GearSet geometry and mesh data"]
    end

    subgraph DynamicModels
        Base["DynamicModel utilities"]
        K94["Kahraman_94 torsional model"]
        LP99["Lin_Parker_99 transverse/torsional model"]
    end

    Shaft --> K94
    Shaft --> LP99
    Bearing --> GearSet
    GearSet --> K94
    GearSet --> LP99
    Base --> K94
    Base --> LP99
```

## Notes

- Commercial-tool integrations are optional adapters and should stay outside
  the core component and dynamic model classes.
- Validation tests should prefer small deterministic fixtures, published
  reference values, or independently derived symbolic notebooks where possible.
- New dynamic formulations should document their DOF order before exposing
  stage matrix helpers.
