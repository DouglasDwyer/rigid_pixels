use crate::*;
use std::collections::*;

pub use self::pgs::*;

/// Implements the PGS solver.
mod pgs;

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
    /// Basic Projected Gauss-Seidel with Baumgarte stabilization. 
    Pgs(Pgs)
}

impl Solver {
    /// Solves all contacts and joints, then integrates object velocities and positions.
    pub fn update(&mut self, contacts: &[Contact], world: &mut PixelWorld, delta_time: f32) {
        match self {
            Solver::Pgs(pgs) => {
                let constraints  = contacts.iter().flat_map(Contact::to_constraints).collect::<Vec<_>>();
                pgs.solve(&constraints, world, delta_time);
            },
        }
    }
}

/// Integrates the forces on all objects in the world, updating their velocities.
pub fn integrate_external_forces(world: &mut PixelWorld, delta_time: f32) {
    for object in world.values_mut() {
        object.velocity += delta_time * Motion {
            linear: object.body.inverse_mass() * object.forces.force,
            angular: object.body.inverse_inertia_tensor() * object.forces.torque
        };
    }
}

/// Integrates `velocity` over `delta_time` into `transform`.
pub fn integrate_velocity(mut transform: Transform, velocity: Motion, delta_time: f32) -> Transform {
    transform.position += delta_time * velocity.linear;
    transform.rotation += delta_time * velocity.angular;
    transform
}

/// Integrates the velocities of all objects in the world, updating their positions.
pub fn integrate_velocities(world: &mut PixelWorld, delta_time: f32) {
    for object in world.values_mut() {
        object.transform = integrate_velocity(object.transform, object.velocity, delta_time);
    }
}