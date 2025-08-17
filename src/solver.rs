use crate::*;

pub use self::pgs::*;

/// Implements the PGS solver.
mod pgs;

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
                integrate_velocities(world, delta_time);
            },
        }
    }
}

/// Integrates the velocities of all objects in the world, updating their positions.
fn integrate_velocities(world: &mut PixelWorld, delta_time: f32) {
    for object in world.values_mut() {
        object.transform.position += delta_time * object.velocity.linear;
        object.transform.rotation += delta_time * object.velocity.angular;
    }
}