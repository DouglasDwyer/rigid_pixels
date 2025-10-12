# rigid_pixels

Rigid pixels is a toy 2D physics engine based on ["Iterative Dynamics with Temporal Coherence"](https://box2d.org/files/ErinCatto_IterativeDynamics_GDC2005.pdf), [Box2D](https://box2d.org/), and [Teardown](https://teardowngame.com/). It implements 2D versions of the collision detection techniques described by Dennis Gustafsson in the [Teardown Technical Teardown](https://www.youtube.com/watch?v=0VzE8ROwC58). My goal with this project was to evaluate various collision detection and resolution algorithms, in the hopes of porting it to my [3D voxel engine](https://github.com/DouglasDwyer/octo-release).

### How to run

- Ensure that [`Rust`](https://rust-lang.org/) is installed
- Download this repository
- From a terminal in the repo root, do `cargo run`