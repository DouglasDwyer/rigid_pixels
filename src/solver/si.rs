use crate::solver::*;
use std::collections::*;

/// Iteratively applies impulses to resolve velocity and position constraints.
/// In the style of Catto's Box2d.
#[derive(Debug)]
pub struct SequentialImpulse {
    /// The configuration to use.
    config: SolverConfig,
    /// A cache containing forces from the previous frame, to use with warm starting.
    impulse_cache: HashMap<ConstraintId, CachedImpulse>,
}

impl SequentialImpulse {
    /// A small amount of error to maintain when solving contact constraints.
    /// This ensures that the collision detector consistently picks up on contacts
    /// even after they are initially solved.
    const LINEAR_SLOP: f32 = 0.01;

    /// Creates a new sequential impulse solver.
    pub fn new(config: SolverConfig) -> Self {
        let impulse_cache = HashMap::new();

        Self {
            config,
            impulse_cache
        }
    }

    /// Solves all contacts and joints, then updates the position/velocity of every object in `world`.
    pub fn solve(&mut self, contacts: &[Contact], world: &mut PixelWorld, delta_time: f32) {
        let mut constraints = world.joints.iter().map(|x| Constraint::new(x.clone(), world, &self.impulse_cache))
            .chain(contacts.iter().map(|x| Constraint::new(x.clone(), world, &self.impulse_cache)))
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
    fn solve_restitution(&mut self, constraint: &mut Constraint, world: &mut PixelWorld) {
        if let ConstraintSource::Contact(contact) = &constraint.source {
            let relative_velocity = constraint.relative_velocity(world);
            let velocity_per_impulse = Self::velocity_per_impulse(constraint, world);

            let normal_velocity_per_impulse = velocity_per_impulse * contact.normal;
            let required_impulse = -(relative_velocity + contact.material.restitution * constraint.original_relative_velocity).dot(contact.normal)
                / normal_velocity_per_impulse.dot(contact.normal);

            let prior_normal_impulse = constraint.impulse.dot(contact.normal);
            let total_impulse = (prior_normal_impulse + required_impulse).max(0.0);
            let delta_impulse = total_impulse - prior_normal_impulse;

            Self::apply_impulse(constraint, world, delta_impulse * contact.normal, 0.0);
        }
    }
    
    /// Solves a single velocity constraint, updating the total applied impulse and the velocity
    /// of the bodies in the `world`.
    fn solve_velocity(&self, constraint: &mut Constraint, world: &mut PixelWorld, delta_time: f32, apply_baumgarte: bool) {
        match &constraint.source {
            ConstraintSource::Contact(_) => self.solve_contact_velocity(constraint, world, delta_time, apply_baumgarte),
            ConstraintSource::Joint(_) => self.solve_joint_velocity(constraint, world, delta_time, apply_baumgarte),
        }
    }

    /// Solves a contact-based velocity constraint.
    fn solve_contact_velocity(&self, constraint: &mut Constraint, world: &mut PixelWorld, delta_time: f32, apply_baumgarte: bool) {
        let ConstraintSource::Contact(contact) = &constraint.source else { panic!("Called solve_contact_velocity on non-contact constraint") };
        let relative_velocity = constraint.relative_velocity(world) + self.contact_bias_velocity(contact, world, delta_time, apply_baumgarte);
        let velocity_per_impulse = Self::velocity_per_impulse(constraint, world);
        let impulse_per_velocity = velocity_per_impulse.inverse();

        let static_friction_impulse = constraint.impulse - impulse_per_velocity * relative_velocity;
        
        let normal_impulse = static_friction_impulse.dot(contact.normal);
        let planar_impulse = static_friction_impulse.reject_from_normalized(contact.normal);
        let planar_impulse_length = planar_impulse.length();

        let total_impulse_unclamped = if contact.material.friction * normal_impulse < planar_impulse_length {
            let relative_velocity_without_impulse = relative_velocity - velocity_per_impulse * constraint.impulse;
            let impulse_direction = (contact.normal + contact.material.friction * planar_impulse.normalize_or_zero()).normalize();
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
        Self::apply_impulse(constraint, world, delta_impulse, 0.0);
    }

    /// Solves a joint-based velocity constraint.
    fn solve_joint_velocity(&self, constraint: &mut Constraint, world: &mut PixelWorld, delta_time: f32, apply_baumgarte: bool) {
        let delta_impulse = self.solve_joint_linear_velocity(constraint, world, delta_time, apply_baumgarte);
        let delta_impulsive_torque = 0.0;// self.solve_joint_angular_velocity(constraint, world, delta_time, apply_baumgarte);
        Self::apply_impulse(constraint, world, delta_impulse, delta_impulsive_torque);
    }

    /// Solves the translational half of a joint constraint.
    fn solve_joint_linear_velocity(&self, constraint: &Constraint, world: &PixelWorld, delta_time: f32, apply_baumgarte: bool) -> Vec2 {
        let ConstraintSource::Joint(joint) = &constraint.source else { panic!("Called solve_joint_velocity on non-joint constraint") };
        
        // Todo: could this be bad because it does not account for rotational motion far from the origin?
        // The relative velocity only takes into account how the POINTS are moving, but the basis where we clamp is moving too.
        // Would this be more stable if instead of relative velocity we computed velocity of point A in reference frame B (or vice versa)?
        // What's more, the velocity per impulse would be WRONG in this case because it assumes the points overlap... can we calculate the correct one?
        let velocity_per_impulse = Self::velocity_per_impulse(constraint, world);
        let relative_displacement = constraint.relative_displacement(world);
        let relative_velocity = constraint.relative_velocity(world) - velocity_per_impulse * constraint.impulse;

        let world_to_joint_rotation = Mat2::from_angle(-world.objects[joint.objects[1]].transform.rotation - joint.local_transform[1].rotation);
        let joint_displacement = world_to_joint_rotation * relative_displacement;
        let joint_velocity = world_to_joint_rotation * relative_velocity;

        let substep_baumgarte = if apply_baumgarte { self.config.baumgarte / self.config.substeps as f32 } else { 0.0 };
        let velocity_lower_bound = Vec2::select(joint_displacement.cmplt(joint.translation_min), Vec2::splat(substep_baumgarte), Vec2::ONE)
            * (joint.translation_min - joint_displacement) / delta_time;
        let velocity_upper_bound = Vec2::select(joint_displacement.cmpgt(joint.translation_max), Vec2::splat(substep_baumgarte), Vec2::ONE)
            * (joint.translation_max - joint_displacement) / delta_time;

        //let clamped_velocity = joint_velocity.clamp(velocity_lower_bound, velocity_upper_bound);
        //println!("CV mm {} for lims {} {}", clamped_velocity - joint_velocity, joint.translation_min, joint.translation_max);

        let mut clamped_velocity = Vec2::ZERO;

        for i in 0..2 {
            if joint.translation_min[i] != 0.0 || joint.translation_max[i] != 0.0 {
                clamped_velocity[i] = joint_velocity[i];
            }
            else {
                clamped_velocity[i] = -substep_baumgarte * joint_displacement[i] / delta_time;
            }
        }
        println!("CV {clamped_velocity} {joint_velocity} {relative_velocity}");

        let desired_velocity = world_to_joint_rotation.inverse() * clamped_velocity;
        let velocity_delta = desired_velocity - relative_velocity;

        let total_impulse = (velocity_per_impulse.inverse() * velocity_delta).clamp_length_max(joint.max_force * delta_time);
        total_impulse - constraint.impulse
    }

    /// Solves the rotational half of a joint constraint.
    fn solve_joint_angular_velocity(&self, constraint: &Constraint, world: &PixelWorld, delta_time: f32, apply_baumgarte: bool) -> f32 {
        let ConstraintSource::Joint(joint) = &constraint.source else { panic!("Called solve_joint_velocity on non-joint constraint") };
        
        let angular_velocity_per_impulsive_torque = Self::angular_velocity_per_impulsive_torque(constraint, world);
        let relative_angular_velocity = constraint.relative_angular_velocity(world);
        
        let rotation_a = world.objects[constraint.objects()[0]].transform.rotation;
        let rotation_b = world.objects[constraint.objects()[1]].transform.rotation;
        let angle_difference = Vec2::from_angle(rotation_b).angle_between(Vec2::from_angle(rotation_a));

        // todo here: figure out how to clamp the angles (and ensure rotational freedom when desired)...
        let substep_baumgarte = if apply_baumgarte { self.config.baumgarte / self.config.substeps as f32 } else { 0.0 };
        let angular_velocity_to_kill = relative_angular_velocity - substep_baumgarte * angle_difference / delta_time;
        let torque = -angular_velocity_per_impulsive_torque.recip() * angular_velocity_to_kill;

        let total_impulsive_torque = (torque + constraint.impulsive_torque).clamp(-joint.max_torque, joint.max_torque);
        total_impulsive_torque - constraint.impulsive_torque
    }

    /// Caches the forces from this tick. The forces will be used in warm-starting
    /// the constraints for the next tick.
    fn cache_constraint_forces(&mut self, constraints: &[Constraint]) {
        self.impulse_cache.clear();
        if self.config.warm_starting {
            for constraint in constraints {
                self.impulse_cache.insert(constraint.id(), CachedImpulse {
                    impulse: constraint.impulse,
                    impulsive_torque: constraint.impulsive_torque
                });
            }
        }
    }

    /// Applies the total impulse stored in each constraint to the objects in the world.
    fn apply_impulses(&self, constraints: &mut [Constraint], world: &mut PixelWorld) {
        if self.config.warm_starting {
            for constraint in constraints {
                let impulse = constraint.impulse;
                let impulsive_torque = constraint.impulsive_torque;
                constraint.impulse = Vec2::ZERO;
                constraint.impulsive_torque = 0.0;
                Self::apply_impulse(constraint, world, impulse, impulsive_torque);
            }
        }
    }

    /// Applies an impulse from a contact. Updates the contact's total impulse and the velocity of
    /// the associated bodies in the `world`.
    fn apply_impulse(constraint: &mut Constraint, world: &mut PixelWorld, impulse: Vec2, impulsive_torque: f32) {
        constraint.impulse += impulse;
        constraint.impulsive_torque += impulsive_torque;

        for (index, id) in constraint.objects().into_iter().enumerate() {
            let object = &mut world.objects[id];
            let relative_position = constraint.local_position()[index].rotate(Vec2::from_angle(object.transform.rotation));
            let body_impulsive_torque = relative_position.perp_dot(impulse) + impulsive_torque;
            let sign = [-1.0, 1.0][index];
            object.velocity += Velocity {
                linear: sign * object.body.inverse_mass() * impulse,
                angular: sign * object.body.inverse_inertia_tensor() * body_impulsive_torque
            };
        }
    }

    /// Computes the bias velocity to use when solving a contact constraint.
    /// 
    /// For speculative contacts, the bias removes exactly enough velocity
    /// to eliminate any separation between the bodies.
    /// 
    /// For non-speculative contacts, the bias includes the Baumgarte coefficient and slop.
    fn contact_bias_velocity(&self, contact: &Contact, world: &PixelWorld, delta_time: f32, apply_baumgarte: bool) -> Vec2 {
        let separation = contact.separation(world);
        let magnitude = if separation < 0.0 {
            if apply_baumgarte {
                (self.config.baumgarte / self.config.substeps as f32) * (separation + Self::LINEAR_SLOP).min(0.0) / delta_time
            }
            else {
                0.0
            }            
        }
        else {
            separation / delta_time
        };

        magnitude * contact.normal
    }

    /// Computes the amount of angular velocity added per unit of impulsive torque.
    fn angular_velocity_per_impulsive_torque(constraint: &Constraint, world: &PixelWorld) -> f32 {
        let mut result = 0.0;

        for (index, id) in constraint.objects().into_iter().enumerate() {
            let object = &world.objects[id];
            result += object.body.inverse_inertia_tensor();
        }

        result
    }

    /// Computes a linear map from impulse to the associated velocity.
    fn velocity_per_impulse(constraint: &Constraint, world: &PixelWorld) -> Mat2 {
        let mut result = Mat2::ZERO;

        for (index, id) in constraint.objects().into_iter().enumerate() {
            let object = &world.objects[id];

            let scaled_tangent = constraint.local_position()[index].rotate(Vec2::from_angle(object.transform.rotation)).rotate(Vec2::Y);
            result += Mat2::from_diagonal(Vec2::splat(object.body.inverse_mass()));
            result += object.body.inverse_inertia_tensor() * vec2_outer_product(scaled_tangent, scaled_tangent);
        }

        result
    }
}

/// Holds information about the force generated by a [`Contact`].
/// Used by the solver for computation.
#[derive(Debug)]
struct Constraint {
    /// The amount of impulse applied (both normal force and friction).
    pub impulse: Vec2,
    /// The amount of torque applied for rotation constraints.
    pub impulsive_torque: f32,
    /// The velocity of the contact at the beginning of the tick, before
    /// applying *any* forces.
    pub original_relative_velocity: Vec2,
    /// The source of the constraint.
    pub source: ConstraintSource,
}

impl Constraint {
    /// Creates a new constraint from the provided source. Attempts to load
    /// the impulse from the cache, if possible.
    pub fn new(source: impl Into<ConstraintSource>, world: &PixelWorld, impulse_cache: &HashMap<ConstraintId, CachedImpulse>) -> Self {
        let mut result = Self {
            impulse: Vec2::ZERO,
            impulsive_torque: 0.0,
            original_relative_velocity: Vec2::ZERO,
            source: source.into(),
        };

        result.initialize(world, impulse_cache);

        result
    }

    /// Gets an identifier for tracking this constraint across frames.
    pub fn id(&self) -> ConstraintId {
        match &self.source {
            ConstraintSource::Contact(contact) => ConstraintId::ContactId(contact.id()),
            ConstraintSource::Joint(joint) => ConstraintId::JointId(joint.id),
        }
    }

    /// Gets the position of the constrained point on each object.
    pub fn local_position(&self) -> [Vec2; 2] {
        match &self.source {
            ConstraintSource::Contact(contact) => contact.local_position,
            ConstraintSource::Joint(joint) => joint.local_transform.map(|x| x.position),
        }
    }

    /// Gets the objects affected by the constraint.
    pub fn objects(&self) -> [ObjectId; 2] {
        match &self.source {
            ConstraintSource::Contact(contact) => contact.objects,
            ConstraintSource::Joint(joint) => joint.objects,
        }
    }

    /// Gets the displacement from the contact point on the first object to the second.
    pub fn relative_displacement(&self, world: &PixelWorld) -> Vec2 {
        let mut displacement = Vec2::ZERO;
        for (index, object) in self.objects().into_iter().enumerate() {
            let body = &world.objects[object];
            displacement += [-1.0, 1.0][index] * (body.transform * self.local_position()[index]);
        }
        displacement
    }

    /// Gets the relative angular velocity of the two objects.
    pub fn relative_angular_velocity(&self, world: &PixelWorld) -> f32 {
        let mut angular_velocity = 0.0;
        for (index, object) in self.objects().into_iter().enumerate() {
            let body = &world.objects[object];
            angular_velocity += [-1.0, 1.0][index] * body.velocity.angular;
        }
        angular_velocity
    }

    /// Gets the relative velocity of the two objects at the constrained point.
    pub fn relative_velocity(&self, world: &PixelWorld) -> Vec2 {
        let mut velocity = Vec2::ZERO;
        for (index, object) in self.objects().into_iter().enumerate() {
            let body = &world.objects[object];
            let relative_position = self.local_position()[index].rotate(Vec2::from_angle(body.transform.rotation));
            velocity += [-1.0, 1.0][index] * (body.velocity.linear + body.velocity.angular * relative_position.rotate(Vec2::Y));
        }
        velocity
    }

    /// Initializes the impulse and relative velocity when the contact is first created.
    fn initialize(&mut self, world: &PixelWorld, impulse_cache: &HashMap<ConstraintId, CachedImpulse>) {
        if let Some(cached) = impulse_cache.get(&self.id()) {
            self.impulse = cached.impulse;
            self.impulsive_torque = cached.impulsive_torque;
        }

        self.original_relative_velocity = self.relative_velocity(world);
    }
}

/// What generates a particular [`Constraint`].
#[derive(Debug)]
enum ConstraintSource {
    /// The constraint comes from a contact.
    Contact(Contact),
    /// The constraint comes from a joint.
    Joint(Joint)
}

/// Allows for uniquely identifying a constraint across frames.
#[derive(Copy, Clone, Debug, Hash, PartialEq, Eq)]
enum ConstraintId {
    /// The constraint comes from a contact.
    ContactId(ContactId),
    /// The constraint comes from a joint.
    JointId(JointId)
}

/// Stores the impulses from a [`Constraint`] across frames.
#[derive(Copy, Clone, Debug)]
struct CachedImpulse {
    /// The total amount of impulse applied.
    pub impulse: Vec2,
    /// The amount of torque applied for rotation constraints.
    pub impulsive_torque: f32,
}

impl From<Contact> for ConstraintSource {
    fn from(value: Contact) -> Self {
        Self::Contact(value)
    }
}

impl From<Joint> for ConstraintSource {
    fn from(value: Joint) -> Self {
        Self::Joint(value)
    }
}