#![allow(unused)]

use egui_macroquad::egui;
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
            self.physics.detector.update(&self.world);
            self.physics.solver.update(self.physics.detector.contacts(), &mut self.world, self.physics.delta_time);
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
                object.forces.force += -G * Vec2::Y / object.body.inverse_inertia_tensor();
            }
        }
    }
}

const G: f32 = 8.0;

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
    /// The offset from each object's origin to [`Self::po`]
    pub relative_position: [Vec2; 2],
    /// The coefficient of friction at the contact.
    pub friction: f32,
    /// The coefficient of restitution at the contact.
    pub restitution: f32,
    /// The amount of distance between the objects. Negative when the objects *are* overlapping.
    pub separation: f32,
    /// The normal (in world space) of object `0`'s surface.
    pub normal: Vec2,
    /// The position of the contact in world space.
    pub position: Vec2
}

impl Contact {
    /// Creates constraints to represent this contact.
    pub fn to_constraints(&self) -> [Constraint; 1] {
        [Constraint {
            objects: self.objects,
            c: self.separation,
            j: MotionPair([
                Motion { linear: -self.normal, angular: -self.relative_position[0].perp_dot(self.normal) },
                Motion { linear: self.normal, angular: self.relative_position[1].perp_dot(self.normal) }
            ]),
            range: 0.0..=f32::MAX
        }]
    }
}

/// A single force, generated from a constraint function `C`, that limits objects' motion.
#[derive(Clone, Debug)]
pub struct Constraint {
    /// The objects affected by the constraint.
    pub objects: [ObjectId; 2],
    /// The value of the constraint function `C`.
    pub c: f32,
    /// The Jacobian associated with this constraint.
    pub j: MotionPair,
    /// The range of allowed force values for this constraint.
    pub range: RangeInclusive<f32>
}

/// Executes the main loop.
#[macroquad::main("Rigid pixels")]
async fn main() {
    let mut simulation = Simulation::new(PhysicsEngine {
        detector: Detector::new(DetectorKind::Naive),
        solver: Solver::Pgs(Pgs::new(PgsConfig {
            baumgarte: 0.1,
            iterations: 8
        })),
        delta_time: 0.015
    }, get_time(), scene::circle_rotation_jitter());

    loop {
        simulation.update(get_time());
        next_frame().await;
    }
}