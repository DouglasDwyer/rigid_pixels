use crate::*;
use std::collections::*;

pub use self::si::*;

/// Implements the sequential impulse solver.
mod si;

/// Determines how the physics solver will behave.
#[derive(Copy, Clone, Debug)]
pub struct SolverConfig {
    /// The Baumgarte factor for velocity to apply over the entire tick.
    pub baumgarte: f32,
    /// The number of extra velocity iterations to perform *without*
    /// any Baumgarte stabilization. This removes extra energy from the system.
    pub relaxation_iterations: u32,
    /// The number of solver (TGS) iterations to perform.
    pub substeps: u32,
    /// The number of velocity-correcting iterations to perform.
    pub velocity_iterations: u32,
    /// Whether to cache constraint forces and use them as the initial guess next frame.
    pub warm_starting: bool
}

/// An algorithm for resolving collisions between objects.
#[derive(Debug)]
pub enum Solver {
    /// Catto-style iterative solver.
    SequentialImpulse(SequentialImpulse)
}

impl Solver {
    /// Solves all contacts and joints, then updates the position/velocity of every object in `world`.
    pub fn solve(&mut self, contacts: &[Contact], world: &mut PixelWorld, delta_time: f32) {
        match self {
            Solver::SequentialImpulse(x) => x.solve(contacts, world, delta_time)
        }
    }
}

/// Describes a force applied by the solver.
#[derive(Copy, Clone, Debug)]
pub struct Force {
    /// The object experiencing the force.
    pub object: ObjectId,
    /// The position (in world space) where the force is applied.
    pub position: Vec2,
    /// The components of the force being applied.
    pub value: Vec2
}


/// Integrates the forces on all objects in the world, updating their velocities.
fn integrate_external_forces(world: &mut PixelWorld, delta_time: f32) {
    for object in world.values_mut() {
        object.velocity = object.velocity.integrate_force(delta_time, object.forces, &object.body);
    }
}

/// Integrates the velocities of all objects in the world, updating their positions.
fn integrate_velocities(world: &mut PixelWorld, delta_time: f32) {
    for object in world.values_mut() {
        object.transform = object.transform.integrate_velocity(delta_time, object.velocity);
    }
}