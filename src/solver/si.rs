use crate::solver::*;
use std::collections::*;

/// Iteratively applies impulses to resolve velocity and position constraints.
/// In the style of Catto's Box2d.
#[derive(Debug)]
pub struct SequentialImpulse {
    /// The configuration to use.
    config: SolverConfig,
    /// A cache containing forces from the previous frame, to use with warm starting.
    force_cache: HashMap<ContactId, Vec2>,
}

impl SequentialImpulse {
    /// A small amount of error to maintain when solving contact constraints.
    /// This ensures that the collision detector consistently picks up on contacts
    /// even after they are initially solved.
    const LINEAR_SLOP: f32 = 0.01;

    /// Creates a new sequential impulse solver.
    pub fn new(config: SolverConfig) -> Self {
        let force_cache = HashMap::new();

        Self {
            config,
            force_cache
        }
    }

    /// Solves all contacts and joints, then updates the position/velocity of every object in `world`.
    pub fn solve(&mut self, contacts: &[Contact], world: &mut PixelWorld, delta_time: f32) {
        let mut constraints = contacts.iter().map(|x| VelocityConstraint::new(x, world, &self.force_cache))
            .collect::<Vec<_>>();

        let substep_time = delta_time / self.config.substeps as f32;
        for _ in 0..self.config.substeps {
            self.apply_impulses(&mut constraints, world);
            integrate_external_forces(world, substep_time);
            
            for _ in 0..self.config.velocity_iterations {
                for constraint in &mut constraints {
                    self.solve_velocity(constraint, world, substep_time, true);
                }
            }

            integrate_velocities(world, substep_time);

            let mut positions = contacts.iter().map(PositionConstraint::new)
                .collect::<Vec<_>>();

            for _ in 0..self.config.position_iterations {
                for constraint in &mut positions {
                    self.solve_position(constraint, world);
                }
            }
                
            for _ in 0..self.config.relaxation_iterations {
                for constraint in &mut constraints {
                    self.solve_velocity(constraint, world, substep_time, false);
                }
            }
        }
        
        self.cache_constraint_forces(&constraints);

        for constraint in &mut constraints {
            self.solve_restitution(constraint, world);
        }
    }

    /// Applies restitution to a constraint. Solves for the normal impulse that makes
    /// the relative velocity equal to the opposite of the *original* relative velocity,
    /// scaled by the restitution.
    fn solve_restitution(&mut self, constraint: &mut VelocityConstraint, world: &mut PixelWorld) {
        let contact = &constraint.contact;
        let relative_velocity = calculate_relative_velocity(contact, world);
        let velocity_per_impulse = Self::velocity_per_impulse(contact, world);

        let normal_velocity_per_impulse = velocity_per_impulse * contact.normal;
        let required_impulse = -(relative_velocity + contact.material.restitution * constraint.original_relative_velocity).dot(contact.normal)
            / normal_velocity_per_impulse.dot(contact.normal);

        let prior_normal_impulse = constraint.impulse.dot(contact.normal);
        let total_impulse = (prior_normal_impulse + required_impulse).max(0.0);
        let delta_impulse = total_impulse - prior_normal_impulse;

        Self::apply_impulse(constraint, world, delta_impulse * contact.normal);
    }

    /// Solves a single position constraint, updating the total applied integral impulse and
    /// the positions of the bodies in the `world`.
    fn solve_position(&self, constraint: &mut PositionConstraint, world: &mut PixelWorld) {
        let contact = &constraint.contact;
        let displacement_per_integral_impulse = Self::velocity_per_impulse(contact, world);

        let normal_displacement_per_integral_impulse = displacement_per_integral_impulse * contact.normal;
        let required_displacement = contact.separation(world) + Self::LINEAR_SLOP;
        let required_integral_impulse = -(self.config.position_baumgarte / self.config.substeps as f32) * required_displacement
            / normal_displacement_per_integral_impulse.dot(contact.normal);

        let total_integral_impulse = (constraint.integral_impulse + required_integral_impulse).max(0.0);
        let delta_integral_impulse = (total_integral_impulse - constraint.integral_impulse) * contact.normal;
        constraint.integral_impulse = total_integral_impulse;

        for (index, id) in contact.objects.into_iter().enumerate() {
            let object = &mut world[id];
            let relative_position = contact.local_position[index].rotate(Vec2::from_angle(object.transform.rotation));
            let integral_impulsive_torque = relative_position.perp_dot(delta_integral_impulse);
            let sign = [-1.0, 1.0][index];
            object.transform.position += sign * object.body.inverse_mass() * delta_integral_impulse;
            object.transform.rotation += sign * object.body.inverse_inertia_tensor() * integral_impulsive_torque;
        }
        
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
        }     */
    }

    /// Solves a single velocity constraint, updating the total applied impulse and the velocity
    /// of the bodies in the `world`.
    fn solve_velocity(&self, constraint: &mut VelocityConstraint, world: &mut PixelWorld, delta_time: f32, apply_bias: bool) {
        let contact = &constraint.contact;
        let relative_velocity = calculate_relative_velocity(contact, world) + if apply_bias { self.bias_velocity(contact, world, delta_time) } else { Vec2::ZERO };
        let velocity_per_impulse = Self::velocity_per_impulse(contact, world);
        let impulse_per_velocity = velocity_per_impulse.inverse();

        let static_friction_impulse = constraint.impulse - impulse_per_velocity * relative_velocity;
        
        let normal_impulse = static_friction_impulse.dot(contact.normal);
        let planar_impulse = static_friction_impulse.reject_from_normalized(contact.normal);
        let planar_impulse_length = planar_impulse.length();

        let total_impulse_unclamped = if contact.material.friction * normal_impulse < planar_impulse_length {
            let relative_velocity_without_impulse = relative_velocity - velocity_per_impulse * constraint.impulse;
            let impulse_direction = (contact.normal + contact.material.friction * planar_impulse.normalize()).normalize();
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
        Self::apply_impulse(constraint, world, delta_impulse);
    }

    /// Caches the forces from this tick. The forces will be used in warm-starting
    /// the constraints for the next tick.
    fn cache_constraint_forces(&mut self, constraints: &[VelocityConstraint]) {
        self.force_cache.clear();
        if self.config.warm_starting {
            for constraint in constraints {
                self.force_cache.insert(constraint.contact.id(), constraint.impulse);
            }
        }
    }

    /// Applies the total impulse stored in each constraint to the objects in the world.
    fn apply_impulses(&self, constraints: &mut [VelocityConstraint], world: &mut PixelWorld) {
        if self.config.warm_starting {
            for constraint in constraints {
                let impulse = constraint.impulse;
                constraint.impulse = Vec2::ZERO;
                Self::apply_impulse(constraint, world, impulse);
            }
        }
    }

    /// Applies an impulse from a contact. Updates the contact's total impulse and the velocity of
    /// the associated bodies in the `world`.
    fn apply_impulse(constraint: &mut VelocityConstraint, world: &mut PixelWorld, impulse: Vec2) {
        let contact = &constraint.contact;
        constraint.impulse += impulse;
        for (index, id) in constraint.contact.objects.into_iter().enumerate() {
            let object = &mut world[id];
            let relative_position = contact.local_position[index].rotate(Vec2::from_angle(object.transform.rotation));
            let impulsive_torque = relative_position.perp_dot(impulse);
            let sign = [-1.0, 1.0][index];
            object.velocity += Velocity {
                linear: sign * object.body.inverse_mass() * impulse,
                angular: sign * object.body.inverse_inertia_tensor() * impulsive_torque
            };
        }
    }

    /// Computes the bias velocity to use when solving a contact constraint.
    /// 
    /// For speculative contacts, the bias removes exactly enough velocity
    /// to eliminate any separation between the bodies.
    /// 
    /// For non-speculative contacts, the bias includes the Baumgarte coefficient and slop.
    fn bias_velocity(&self, contact: &Contact, world: &PixelWorld, delta_time: f32) -> Vec2 {
        let separation = contact.separation(world);
        let magnitude = if separation < 0.0 {
            (self.config.velocity_baumgarte / self.config.substeps as f32) * (separation + Self::LINEAR_SLOP).min(0.0) / delta_time
        }
        else {
            separation / delta_time
        };

        magnitude * contact.normal
    }

    /// Computes a linear map from impulse to the associated velocity.
    fn velocity_per_impulse(contact: &Contact, world: &PixelWorld) -> Mat2 {
        let mut result = Mat2::ZERO;

        for (index, id) in contact.objects.into_iter().enumerate() {
            let object = &world[id];

            let scaled_tangent = contact.local_position[index].rotate(Vec2::from_angle(object.transform.rotation)).rotate(Vec2::Y);
            result += Mat2::from_diagonal(Vec2::splat(object.body.inverse_mass()));
            result += object.body.inverse_inertia_tensor() * vec2_outer_product(scaled_tangent, scaled_tangent);
        }

        result
    }
}

/// Holds information about the displacements generated by a [`Contact`].
/// Used by the solver for computation.
#[derive(Copy, Clone, Debug)]
struct PositionConstraint {
    /// The contact dictating this constraint.
    pub contact: Contact,
    /// The total amount of impulse applied, integrated over the time step.
    /// This has units `mass * length`.
    pub integral_impulse: f32
}

impl PositionConstraint {
    /// Creates a new constraint to model `contact`.
    pub fn new(contact: &Contact) -> Self {
        Self {
            contact: *contact,
            integral_impulse: 0.0
        }
    }
}

/// Holds information about the forces generated by a [`Contact`].
/// Used by the solver for computation.
#[derive(Copy, Clone, Debug)]
struct VelocityConstraint {
    /// The contact dictating this constraint.
    pub contact: Contact,
    /// The amount of impulse applied (both normal force and friction).
    pub impulse: Vec2,
    /// The velocity of the contact at the beginning of the tick, before
    /// applying *any* forces.
    pub original_relative_velocity: Vec2
}

impl VelocityConstraint {
    /// Creates a new constraint to model `contact`.
    pub fn new(contact: &Contact, world: &PixelWorld, impulse_cache: &HashMap<ContactId, Vec2>) -> Self {
        Self {
            contact: *contact,
            impulse: impulse_cache.get(&contact.id()).copied().unwrap_or_default(),
            original_relative_velocity: calculate_relative_velocity(&contact, world),
        }
    }
}

/// Gets the relative velocity of two objects at the `contact` point.
fn calculate_relative_velocity(contact: &Contact, world: &PixelWorld) -> Vec2 {
    let mut velocity = Vec2::ZERO;
    for (index, object) in contact.objects.into_iter().enumerate() {
        let body = &world[object];
        let relative_position = contact.local_position[index].rotate(Vec2::from_angle(body.transform.rotation));
        velocity += [-1.0, 1.0][index] * (body.velocity.linear + body.velocity.angular * relative_position.rotate(Vec2::Y));
    }
    velocity
}