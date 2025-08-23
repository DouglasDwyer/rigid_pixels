use crate::*;
use std::collections::*;

pub use self::sequential_impulse::*;

/// Implements the sequential impulse solver.
mod sequential_impulse;

/// Determines how the physics solver will behave.
#[derive(Copy, Clone, Debug)]
pub struct SolverConfig {
    /// The Baumgarte factor to apply.
    pub baumgarte: f32,
    /// The number of solver iterations to perform.
    pub iterations: u32,
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
    /// Returns a list of all constraint forces applied.
    pub fn solve(&mut self, contacts: &[Contact], world: &mut PixelWorld, delta_time: f32) -> Vec<Force> {
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
        let old_velocity = object.velocity;
        object.velocity = object.velocity.integrate_force(delta_time, object.forces, &object.body);
        //println!("INTEGRATE {old_velocity:?} => {:?} ({delta_time} {:?})", object.velocity, object.forces);
    }
}

/// Integrates the velocities of all objects in the world, updating their positions.
fn integrate_velocities(world: &mut PixelWorld, delta_time: f32) {
    for object in world.values_mut() {
        object.transform = object.transform.integrate_velocity(delta_time, object.velocity);
    }
}

/*
/// Integrates the `force` on `body` into `velocity` over `delta_time`.
pub fn integrate_force(mut velocity: Motion, force: Motion, body: &PixelBody, delta_time: f32) -> Motion {
    velocity += delta_time * Motion {
        linear: body.inverse_mass() * force.linear,
        angular: body.inverse_inertia_tensor() * force.angular
    };
    velocity
}

/// Integrates `velocity` over `delta_time` into `transform`.
pub fn integrate_velocity(mut transform: Transform, velocity: Motion, delta_time: f32) -> Transform {
    transform.position += delta_time * velocity.linear;
    transform.rotation += delta_time * velocity.angular;
    transform
}
 */