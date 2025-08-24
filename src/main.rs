#![allow(unused)]

use egui_macroquad::egui;
use macroquad::miniquad::window::set_window_size;
use macroquad::prelude::*;
use self::detector::*;
use self::force::*;
use self::math::*;
use self::pixel::*;
use self::render::*;
use self::render::Camera;
use self::solver::*;
use slotmap::*;
use std::ops::*;

/// Implements logic for detecting collisions between objects.
mod detector;

/// Defines external force generators.
mod force;

/// Defines various math types used for calculations.
mod math;

/// Defines rigid bodies made of pixels.
mod pixel;

/// Handles drawing the simulation.
mod render;

/// Defines worlds for testing.
mod scene;

/// Provides algorithms for resolving contacts and constraints.
mod solver;

/// Holds all state for the world simulation.
pub struct Simulation {
    /// The camera to use during rendering.
    camera: Camera,
    /// The time of the last physics update.
    last_update: f64,
    /// The selected algorithms for physics.
    physics: PhysicsEngine,
    /// Handles user input and drawing the world to the screen.
    renderer: Renderer,
    /// All objects in the simulation.
    world: PixelWorld
}

impl Simulation {
    /// Initializes a new simulation.
    pub fn new(physics: PhysicsEngine, start_time: f64, world: PixelWorld) -> Self {
        Self {
            camera: Camera::default(),
            last_update: start_time,
            physics,
            renderer: Renderer::default(),
            world
        }
    }

    /// Updates the simulation, advancing to `time` (since simulation start in seconds).
    pub fn update(&mut self, time: f64) {
        self.test_force_generator();
        
        while self.last_update < time {
            self.physics.detector.update(&self.world, self.physics.delta_time);
            self.physics.solver.solve(self.physics.detector.contacts(), &mut self.world, self.physics.delta_time);
            self.last_update += self.physics.delta_time as f64;
        }
        self.clear_force_accumulators();

        self.renderer.draw(&self.physics, &mut self.world);
    }

    /// Clears all external forces to progress to the next frame.
    fn clear_force_accumulators(&mut self) {
        for object in self.world.values_mut() {
            object.forces = ForceAccumulator::default()
        }
    }

    /// Adds a test force to all objects in the scene.
    fn test_force_generator(&mut self) {
        for object in self.world.values_mut() {
            if 0.0 < object.body.inverse_mass() {
                object.forces.force += -G * Vec2::Y / object.body.inverse_mass();
            }
        }
    }
}

const G: f32 = 9.81 * 16.0;

/// Holds the algorithms and data for running the physics engine.
#[derive(Debug)]
pub struct PhysicsEngine {
    /// The detector to use for identifying collisions.
    pub detector: Detector,
    /// The solver to use for constraint resolution.
    pub solver: Solver,
    /// The amount of time (in seconds) between full simulation steps.
    pub delta_time: f32
}

/// Information about a hard contact found by the collision [`Detector`].
#[derive(Copy, Clone, Debug)]
pub struct Contact {
    /// The objects involved in the collision.
    pub objects: [ObjectId; 2],
    /// The position of each pixel involved in the collision. 
    pub pixel_position: [UVec2; 2],
    /// The offset from each object's origin to the contact in local space.
    pub local_position: [Vec2; 2],
    /// The material properties at the contact.
    pub material: PixelMaterial,
    /// The required distance (projected along the normal vector)
    /// between the contact point on B and the contact point on A.
    pub penetration: f32,
    /// The normal (in world space) of object `0`'s surface.
    pub normal: Vec2,
    /// The position of the contact in world space.
    pub position: Vec2
}

impl Contact {
    /// Gets a lightweight ID that can be used to track the contact across frames.
    pub fn id(&self) -> ContactId {
        ContactId { objects: self.objects, pixel_position: self.pixel_position }
    }

    /// Computes the current displacement of the contacts along the normal axis,
    /// relative to the required `penetration` for this contact.
    pub fn separation(&self, world: &PixelWorld) -> f32 {
        self.normal.dot(world[self.objects[1]].transform * self.local_position[1]
            - world[self.objects[0]].transform * self.local_position[0]) - self.penetration        
    }

    /// Swaps the order of the objects involved in the collision.
    pub fn swap_objects(&self) -> Self {
        Self {
            objects: [self.objects[1], self.objects[0]],
            pixel_position: [self.pixel_position[1], self.pixel_position[0]],
            local_position: [self.local_position[1], self.local_position[0]],
            normal: -self.normal,
            ..*self
        }
    }
}

/// Allows for uniquely identifying a constraint across frames.
#[derive(Copy, Clone, Debug, Hash, PartialEq, Eq)]
pub struct ContactId {
    /// The objects involved in the collision.
    pub objects: [ObjectId; 2],
    /// The position of each pixel involved in the collision. 
    pub pixel_position: [UVec2; 2]
}

/// Executes the main loop.
#[macroquad::main("Rigid pixels")]
async fn main() {
    set_window_size(1000, 1000);
    let mut simulation = Simulation::new(PhysicsEngine {
        detector: Detector::new(DetectorKind::Speculative { include_external_forces: true, mode: SpeculativeStepMode::Midpoint }),
        solver: Solver::SequentialImpulse(SequentialImpulse::new(SolverConfig {
            position_baumgarte: 0.2,
            position_iterations: 0,
            velocity_baumgarte: 0.2,
            velocity_iterations: 2,
            relaxation_iterations: 0,
            substeps: 6,
            warm_starting: true
        })),
        delta_time: 0.015
    }, get_time(), scene::single_box());

    loop {
        simulation.update(get_time());
        next_frame().await;
    }
}

/*

Improvements:
- Integrate position AFTER applying constraints to velocity
  > Jitter on boxes is muched reduced
- Use warm starting to increase stability
  > Improvement with big circle lying on ground; previously jittered but now does not
  > Box stack slides without warm starting
  > Tried scaling warm starting by some factor (like 0.8), but this led to sliding. Doesn't seem to be worth it.
- Speculative contacts can work for CCD. External forces NEED to be included in the objects' speculative trajectory.
  > With naive collision detection, falling objects or dragging could cause clipping
  > Without external forces, dragging could clip objects through floor
  > Only speculative contacts with separation greater than zero should count. Separation less implies object clipped through wall.
  > Speculative contacts can be deduplicated since multiple may occur on the same voxel. Earlier contacts should be preferred
  > In the solver, speculative contacts should be handled by removing JUST ENOUGH velocity that separation is zero at the end of the time step
    (this implies restitution must be handled after solving, because otherwise objects will not hit the ground before bouncing)
- Using sequential impulses allows for robust handling of normal/friction simultaneously
  > The impulse vector must be clipped AFTER dynamic friction is calculated
- Linear slop is necessary to ensure the collision detector doesn't miss contacts that are close together
  > Speculative collisions alone did not fix the issue of occasionally missing contacts
- TGS/Baumgarte performs better than pure PGS or NGS on the upside-down pyramid
  > This is using 5 substeps w/ 3 velocity iterations apiece
  > There is some bounciness to the collisions but the stack is otherwise stable
  > The tumbler works as well
  > Apply restitution once on the final impulses output by the solver
  > NGS does not seem to be a signficant improvement; TGS is the big win

Things to think about:
- Relaxation is bugged w.r.t speculative contacts! During relaxation, we try to make the velocity ZERO. But in combination with substepping,
  this means that the velocity goes to zero far too early, and artifacts appear.
  > Turning off relaxation fixes the problem.
  > Is there another way to deal with energetic Baumgarte stabilization?

- Corner normal handling is weird. How does Teardown do it?
  > Just discard any corner-corner normals that conflict? This led to lots of jittering. Not sure why.
  > I also tried using cardinal-only normals. This led to some collisions being missed. I will need to revisit this.
  
- The Tumbler disappeared once. Need to add NaN panics and track that down.

- There were NaNs in collision detection when voxels clipped through the opposite side of an edge. How did the 3D engine handle this?

- Box2D tracks something called totalNormalImpulse to check whether any impulse was EVER generated for a speculative contact.
  It prevents restitution from being applied to contacts without it. How important is this?

*/