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
    /// The selected algorithms for physics.
    physics: PhysicsConfig,
    /// All objects in the simulation.
    world: PixelWorld
}

impl Simulation {
    /// Initializes a new simulation.
    pub fn new(physics: PhysicsConfig, world: PixelWorld) -> Self {
        Self {
            camera: Camera::default(),
            physics,
            world
        }
    }

    /// Updates the simulation, advancing to `time` (since simulation start in seconds).
    pub fn update(&mut self, time: f64) {

    }
}

/// Specifies which algorithms the physics engine will use.
#[derive(Debug)]
pub struct PhysicsConfig {
    /// The detector to use for identifying collisions.
    pub detector: Detector,
    /// The solver to use for constraint resolution.
    pub solver: (),
    /// The amount of time (in seconds) between full simulation steps.
    pub step_time: f32
}

/// Executes the main loop.
#[macroquad::main("Rigid pixels")]
async fn main() {
    let mut simulation = Simulation::new(PhysicsConfig {
        detector: Detector::Naive,
        solver: (),
        step_time: 0.025
    }, scene::circle_rotation_jitter());

    loop {
        simulation.update(get_time());
        draw_simulation(&mut simulation);
        next_frame().await;
    }
}