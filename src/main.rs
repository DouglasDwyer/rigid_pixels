#![allow(unused)]

use crate::detector::*;
use crate::pixel::*;
use crate::render::*;
use crate::render::Camera;
use crate::solver::*;
use egui_macroquad::egui;
use macroquad::prelude::*;
use slotmap::*;

/// Implements logic for detecting collisions between objects.
mod detector;

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
            world
        }
    }

    /// Updates the simulation, advancing to `time` (since simulation start in seconds).
    pub fn update(&mut self, time: f64) {
        while self.last_update < time {
            self.physics.detector.update(&self.world);
            self.last_update += self.physics.step_time as f64;
        }
    }
}

/// Holds the algorithms and data for running the physics engine.
#[derive(Debug)]
pub struct PhysicsEngine {
    /// The detector to use for identifying collisions.
    pub detector: Detector,
    /// The solver to use for constraint resolution.
    pub solver: (),
    /// The amount of time (in seconds) between full simulation steps.
    pub step_time: f32
}

/// Information about a hard contact found by the collision [`Detector`].
#[derive(Copy, Clone, Debug)]
pub struct Contact {
    /// The objects involved in the collision.
    pub objects: [ObjectId; 2],
    /// The coefficient of friction at the contact.
    pub friction: f32,
    /// The coefficient of restitution at the contact.
    pub restitution: f32,
    /// The amount of overlap between the objects.
    pub penetration: f32,
    /// The normal (in world space) of object `0`'s surface.
    pub normal: Vec2,
    /// The position of the contact in world space.
    pub position: Vec2
}

/// Executes the main loop.
#[macroquad::main("Rigid pixels")]
async fn main() {
    let mut simulation = Simulation::new(PhysicsEngine {
        detector: Detector::new(DetectorKind::Naive),
        solver: (),
        step_time: 0.025
    }, get_time(), scene::simple());

    loop {
        simulation.update(get_time());
        draw_simulation(&mut simulation);
        next_frame().await;
    }
}