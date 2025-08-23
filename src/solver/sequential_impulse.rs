use crate::solver::*;
use std::collections::*;

/// Iteratively applies impulses to resolve velocity and position constraints.
/// In the style of Catto's Box2d.
#[derive(Debug)]
pub struct SequentialImpulse {
    /// The configuration to use.
    config: SolverConfig,
    /// A cache containing forces from the previous frame, to use with warm starting.
    force_cache: HashMap<ContactId, f32>,
}

impl SequentialImpulse {
    /// Creates a new sequential impulse solver.
    pub fn new(config: SolverConfig) -> Self {
        let force_cache = HashMap::new();

        Self {
            config,
            force_cache
        }
    }

    /// Solves all contacts and joints, then updates the position/velocity of every object in `world`.
    /// Returns a list of all constraint forces applied.
    pub fn solve(&mut self, contacts: &[Contact], world: &mut PixelWorld, delta_time: f32) -> Vec<Force> {
        self.solve_velocities(contacts, world, delta_time)
    }

    /// Solves velocity constraints, then updates the velocity of every object in `world`.
    fn solve_velocities(&mut self, contacts: &[Contact], world: &mut PixelWorld, delta_time: f32) -> Vec<Force> {
        //todo: warm starting
        integrate_external_forces(world, delta_time);

        let mut constraints = contacts.iter().copied().map(Constraint::new).collect::<Vec<_>>();
        
        for _ in 0..self.config.iterations {
            for constraint in &mut constraints {
                self.solve_velocity(constraint, world, delta_time);
            }
        }

        integrate_velocities(world, delta_time);
        Vec::new()
    }

    /// Computes a linear map from impulse to the associated velocity.
    fn velocity_per_impulse(contact: &Contact, world: &PixelWorld) -> Mat2 {
        let mut result = Mat2::ZERO;

        for (index, id) in contact.objects.into_iter().enumerate() {
            let object = &world[id];

            let scaled_tangent = contact.relative_position[index].rotate(Vec2::Y);
            result += Mat2::from_diagonal(Vec2::splat(object.body.inverse_mass()));
            result += object.body.inverse_inertia_tensor() * vec2_outer_product(scaled_tangent, scaled_tangent);
        }

        result
    }

    /// Gets the relative velocity of two objects at the `contact` point.
    fn relative_velocity(contact: &Contact, world: &PixelWorld) -> Vec2 {
        let mut velocity = Vec2::ZERO;
        for (index, object) in contact.objects.into_iter().enumerate() {
            let body = &world[object];
            let relative_position = contact.relative_position[index];
            velocity += [-1.0, 1.0][index] * (body.velocity.linear + body.velocity.angular * relative_position.rotate(Vec2::Y));
        }
        velocity
    }

    fn solve_velocity(&self, constraint: &mut Constraint, world: &mut PixelWorld, delta_time: f32) {
        let contact = &constraint.contact;
        let baumgarte_velocity = -self.config.baumgarte * contact.separation.min(0.0) / delta_time;
        let relative_velocity = Self::relative_velocity(contact, world) - baumgarte_velocity * contact.normal;
        let velocity_per_impulse = Self::velocity_per_impulse(contact, world);
        let impulse_per_velocity = velocity_per_impulse.inverse();

        let static_friction_impulse = constraint.impulse - impulse_per_velocity * relative_velocity;
        
        let normal_impulse = static_friction_impulse.dot(contact.normal);
        let planar_impulse = static_friction_impulse.reject_from_normalized(contact.normal);
        let planar_impulse_length = planar_impulse.length();

        let total_impulse_unclamped = if contact.friction * normal_impulse < planar_impulse_length {
            let relative_velocity_without_impulse = relative_velocity - velocity_per_impulse * constraint.impulse;
            let impulse_direction = (contact.normal + contact.friction * planar_impulse.normalize()).normalize();
            let velocity_per_directed_impulse = velocity_per_impulse * impulse_direction;
            let impulse_magnitude = -relative_velocity_without_impulse.dot(contact.normal) / velocity_per_directed_impulse.dot(contact.normal);
            impulse_magnitude * impulse_direction
        }
        else {
            static_friction_impulse
        };

        // Important: total impulse may only be clamped AFTER static/dynamic friction is resolved
        let total_impulse = if total_impulse_unclamped.dot(contact.normal) < 0.0 {
            Vec2::ZERO
        }
        else {
            total_impulse_unclamped
        };

        let delta_impulse = total_impulse - constraint.impulse;
        constraint.impulse = total_impulse;

        for (index, id) in contact.objects.into_iter().enumerate() {
            let object = &mut world[id];
            let impulsive_torque = contact.relative_position[index].perp_dot(delta_impulse);
            let sign = [-1.0, 1.0][index];
            object.velocity += Velocity {
                linear: sign * object.body.inverse_mass() * delta_impulse,
                angular: sign * object.body.inverse_inertia_tensor() * impulsive_torque
            };
        }
    }

    fn solve_position() {
        /*
        
        let contact = &constraint.contact;
        let relative_velocity = Self::relative_velocity(contact, world);
        let velocity_per_impulse = Self::velocity_per_impulse(contact, world);

        let baumgarte_velocity = -self.config.baumgarte * contact.separation.min(0.0) / delta_time;
        let normal_velocity_per_impulse = velocity_per_impulse * contact.normal;
        let required_impulse = (baumgarte_velocity - relative_velocity.dot(contact.normal)) / normal_velocity_per_impulse.dot(contact.normal);

        let total_impulse = (constraint.normal_impulse + required_impulse).max(0.0);
        let delta_impulse = total_impulse - constraint.normal_impulse;
        constraint.normal_impulse = total_impulse;

        let impulse = delta_impulse * contact.normal;

        for (index, id) in contact.objects.into_iter().enumerate() {
            let object = &mut world[id];
            let impulsive_torque = contact.relative_position[index].perp_dot(impulse);
            let sign = [-1.0, 1.0][index];
            object.velocity += Velocity {
                linear: sign * object.body.inverse_mass() * impulse,
                angular: sign * object.body.inverse_inertia_tensor() * impulsive_torque
            };
        }
        
        
         */
    }
}

/// Holds information about the forces generated by a [`Contact`]. Used
/// by the solver for computation.
#[derive(Copy, Clone, Debug)]
struct Constraint {
    /// The contact dictating this constraint.
    pub contact: Contact,
    /// The amount of impulse applied (both normal force and friction).
    pub impulse: Vec2,
}

impl Constraint {
    /// Creates a new constraint to model `contact`.
    pub fn new(contact: Contact) -> Self {
        Self {
            contact,
            impulse: Vec2::ZERO,
        }
    }
}
