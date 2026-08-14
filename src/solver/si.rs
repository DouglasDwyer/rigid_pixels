use arrayvec::ArrayVec;

use crate::solver::*;
use std::collections::*;

/// Iteratively applies impulses to resolve velocity and position constraints.
/// In the style of Catto's Box2d.
#[derive(Debug)]
pub struct SequentialImpulse {
    /// The configuration to use.
    config: SolverConfig,
    /// Events generated during the last solver iteration.
    events: Vec<SolverEvent>,
    /// A cache containing forces from the previous frame, to use with warm starting.
    force_cache: HashMap<ConstraintId, CachedImpulse>,
}

impl SequentialImpulse {
    /// Percentage by which to reduce velocity per second.
    const DAMPING: f32 = 0.95;

    /// A small amount of error to maintain when solving contact constraints.
    /// This ensures that the collision detector consistently picks up on contacts
    /// even after they are initially solved.
    const LINEAR_SLOP: f32 = 0.01;

    /// Creates a new sequential impulse solver.
    pub fn new(config: SolverConfig) -> Self {
        let impulse_cache = HashMap::new();
        let events = Vec::new();

        Self {
            config,
            events,
            force_cache: impulse_cache
        }
    }

    /// Gets the events generated during the last solver iteration.
    pub fn events(&self) -> &[SolverEvent] {
        &self.events
    }

    /// Solves all contacts and joints, then updates the position/velocity of every object in `world`.
    pub fn solve(&mut self, contacts: &[Contact], world: &mut PixelWorld, delta_time: f32) {
        self.events.clear();
        
        let substep_time = delta_time / self.config.substeps as f32;

        let mut constraints = world.joints.iter().map(|x| Constraint::new(x.clone(), world, &self.force_cache, substep_time))
            .chain(contacts.iter().map(|x| Constraint::new(x.clone(), world, &self.force_cache, substep_time)))
            .collect::<Vec<_>>();

        for _ in 0..self.config.substeps {
            self.apply_impulses(&mut constraints, world);
            integrate_external_forces(world, substep_time);

            for object in world.objects.values_mut() {
                object.velocity.angular *= Self::DAMPING.powf(substep_time);
                object.velocity.linear *= Self::DAMPING.powf(substep_time);
            }
            
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

            self.sum_total_impulse(&mut constraints);
        }

        self.cache_constraint_forces(&constraints, substep_time);

        for constraint in &mut constraints {
            self.solve_restitution(constraint, world);
        }

        for constraint in &mut constraints {
            self.solve_fracture(constraint, world, delta_time);
        }
    }

    /// Finds contacts where the impulse exceeded [`PixelMaterial::breaking_impulse`].
    /// Removes the excess impulse and generates a fracture event.
    fn solve_fracture(&mut self, constraint: &mut Constraint, world: &mut PixelWorld, delta_time: f32) {
        /// The surface area over which force "spreads out". Used to make small objects fracture quicker.
        const FRACTURE_SURFACE_LENGTH: f32 = 7.0;

        if let ConstraintSource::Contact(contact) = &constraint.source {
            let total_normal_impulse = contact.normal.dot(constraint.total_impulse);

            let scaled_impulses = [
                world.objects[contact.objects[0]].body.area().min(FRACTURE_SURFACE_LENGTH).sqrt() * contact.material.breaking_impulses[0],
                world.objects[contact.objects[1]].body.area().min(FRACTURE_SURFACE_LENGTH).sqrt() * contact.material.breaking_impulses[1]
            ];

            let max_immediate_impulse = scaled_impulses[0].min(scaled_impulses[1]);

            if max_immediate_impulse < total_normal_impulse {
                for (i, max_impulse) in scaled_impulses.into_iter().enumerate() {
                    if max_impulse < total_normal_impulse {
                        self.events.push(SolverEvent::Fracture(Fracture {
                            impulse: [-1.0, 1.0][i] * total_normal_impulse * contact.normal,
                            object: contact.objects[i],
                            pattern: contact.material.fracture_patterns[i],
                            pixel_position: contact.pixel_position[i],
                            strength_ratio: total_normal_impulse / max_impulse
                        }));
                    }
                }

                let excess_impulse = total_normal_impulse - max_immediate_impulse;
                Self::apply_impulse(constraint, world, -excess_impulse * contact.normal, 0.0);
            }
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

            constraint.total_impulse += delta_impulse * contact.normal;
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

        if !Self::approx_zero(delta_impulse) {
            Self::apply_impulse(constraint, world, delta_impulse, 0.0);
        }
    }

    /// Solves a joint-based velocity constraint.
    fn solve_joint_velocity(&self, constraint: &mut Constraint, world: &mut PixelWorld, delta_time: f32, apply_baumgarte: bool) {
        //println!("preiter ({:?} {:?})", world.objects[constraint.objects()[0]].velocity, world.objects[constraint.objects()[1]].velocity);
        /*let mut delta_impulse = self.solve_joint_linear_velocity(constraint, world, delta_time, apply_baumgarte);
        let delta_impulsive_torque = self.solve_joint_angular_velocity(constraint, world, delta_time, apply_baumgarte);
        Self::apply_impulse(constraint, world, delta_impulse, delta_impulsive_torque);*/

        let ConstraintSource::Joint(joint) = &constraint.source else { panic!() };

        let mut solver = BlockSolver::default();

        /*
        let x_0 = Vec2::ZERO;
        let x_1 = Vec2::ZERO;

        let r_0 = Vec2::ZERO;
        let r_1 = Vec2::ZERO;

        let theta_0 = 0.0;
        let theta_1 = 0.0;

        let c_trans = x_1 + r_1 - x_0 - r_0;
        let c_rot = theta_1 - theta_0;

        solver.add_constraint(
            c_trans.x,
            [
                Screw { linear: Vec2::X, angular: -r_0.y },
                Screw { linear: Vec2::NEG_X, angular: r_1.y }
            ],
            0.0
        );
        
        solver.add_constraint(
            c_trans.y,
            [
                Screw { linear: Vec2::Y, angular: r_0.x },
                Screw { linear: Vec2::NEG_Y, angular: -r_1.x }
            ],
            0.0
        ); */
    }

    /// Draws an arrow between `start` and `end`.
    fn draw_arrow(start: Vec2, end: Vec2, color: Color) {
        // Draw main shaft of the arrow
        draw_line(start.x, start.y, end.x, end.y, 1.0, color);

        // Calculate direction and angle
        let direction = end - start;
        let angle = direction.to_angle();

        // Arrowhead parameters
        let head_length = 3.0;
        let head_angle = 0.5;

        let left = vec2(
            end.x - head_length * (angle - head_angle).cos(),
            end.y - head_length * (angle - head_angle).sin(),
        );
        let right = vec2(
            end.x - head_length * (angle + head_angle).cos(),
            end.y - head_length * (angle + head_angle).sin(),
        );

        // Draw arrowhead
        draw_line(end.x, end.y, left.x, left.y, 1.0, color);
        draw_line(end.x, end.y, right.x, right.y, 1.0, color);
    }

    /// Solves the translational half of a joint constraint.
    fn solve_joint_linear_velocity(&self, constraint: &Constraint, world: &PixelWorld, delta_time: f32, apply_baumgarte: bool) -> Vec2 {
        let ConstraintSource::Joint(joint) = &constraint.source else { panic!("Called solve_joint_velocity on non-joint constraint") };
        
        /*
        
        Key thing that's not working:
         - The joint velocity should NOT CHANGE the velocity along unconstrained axes
         - Right now, the joint's existence appears to be ADDING velocity horizontally.
         - This shouldn't happen since `relative_velocity` is set to the velocity WITHOUT the constraint being there
         - So the solution should attempt to preserve the ORIGINAL horizontal velocity rather than adding more.
        
         */

        let velocity_per_impulse = Self::velocity_per_impulse(constraint, world);
        let relative_displacement = Self::relative_displacement(joint, world);
        let relative_velocity = constraint.relative_velocity(world) - velocity_per_impulse * constraint.impulse;

        //let world_to_joint_rotation = Mat2::from_angle(-world.objects[joint.objects[0]].transform.rotation - joint.local_transform[0].rotation);
        //let joint_displacement = world_to_joint_rotation * relative_displacement;
        //let joint_velocity = world_to_joint_rotation * relative_velocity;
        
        let substep_baumgarte = if apply_baumgarte { self.config.baumgarte / self.config.substeps as f32 } else { 0.0 };
        
        /*
        let velocity_lower_bound = Vec2::select(joint_displacement.cmplt(joint.translation_min), Vec2::splat(substep_baumgarte), Vec2::ONE)
            * (joint.translation_min - joint_displacement) / delta_time;
        let velocity_upper_bound = Vec2::select(joint_displacement.cmpgt(joint.translation_max), Vec2::splat(substep_baumgarte), Vec2::ONE)
            * (joint.translation_max - joint_displacement) / delta_time;

        let clamped_velocity = joint_velocity.clamp(velocity_lower_bound, velocity_upper_bound);
        //println!("Dvel {clamped_velocity} vs {relative_velocity} {}", constraint.relative_velocity(world)); */

        //let substep_baumgarte = if apply_baumgarte { self.config.baumgarte / self.config.substeps as f32 } else { 0.0 };
        //
        //let normal_direction = world_to_joint_rotation.inverse() * Vec2::Y;
        //let velocity_per_normal_impulse = velocity_per_impulse * normal_direction;
        //let required_impulse = -(substep_baumgarte * relative_displacement.dot(normal_direction) / delta_time + relative_velocity.dot(normal_direction)) / velocity_per_normal_impulse.dot(normal_direction);

        
        //let desired_velocity = world_to_joint_rotation.inverse() * joint_velocity;
        let velocity_delta = -substep_baumgarte * relative_displacement / delta_time - relative_velocity;


        let total_impulse = (velocity_per_impulse.inverse() * velocity_delta).clamp_length_max(joint.max_force * delta_time);
        //println!("vpi {velocity_per_impulse} * {total_impulse} = {velocity_delta} (for relv {relative_velocity}");
        total_impulse - constraint.impulse
        //required_impulse * normal_direction - constraint.impulse
    }

    /// Solves the rotational half of a joint constraint.
    fn solve_joint_angular_velocity(&self, constraint: &Constraint, world: &PixelWorld, delta_time: f32, apply_baumgarte: bool) -> f32 {
        let ConstraintSource::Joint(joint) = &constraint.source else { panic!("Called solve_joint_velocity on non-joint constraint") };
        
        let angular_velocity_per_impulsive_torque = Self::angular_velocity_per_impulsive_torque(constraint, world);
        let relative_angular_velocity = constraint.relative_angular_velocity(world);
        
        let rotation_a = world.objects[constraint.objects()[0]].transform.rotation + joint.local_transform[0].rotation;
        let rotation_b = world.objects[constraint.objects()[1]].transform.rotation + joint.local_transform[1].rotation;
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
    fn cache_constraint_forces(&mut self, constraints: &[Constraint], substep_time: f32) {
        self.force_cache.clear();
        if self.config.warm_starting {
            for constraint in constraints {
                self.force_cache.insert(constraint.id(), CachedImpulse {
                    force: constraint.impulse / substep_time,
                    torque: constraint.impulsive_torque / substep_time
                });
            }
        }
    }
    
    /// Stores the total impulse from the substep.
    fn sum_total_impulse(&mut self, constraints: &mut [Constraint]) {
        for constraint in constraints {
            constraint.total_impulse += constraint.impulse;
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
            let relative_position = constraint.local_position(world)[index].rotate(Vec2::from_angle(world.objects[id].transform.rotation));
            let object = &mut world.objects[id];
            let body_impulsive_torque = relative_position.perp_dot(impulse) + impulsive_torque;
            let sign = [-1.0, 1.0][index];
            object.velocity += Screw {
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

    /// Determines whether all components of the vector `v` are approximately equal to zero.
    fn approx_zero(v: Vec2) -> bool {
        v.abs().max_element() < 1e-6
    }
    
    /// Gets the displacement in world space between the two joint anchors.
    fn relative_displacement(joint: &Joint, world: &PixelWorld) -> Vec2 {
        let mut displacement = Vec2::ZERO;
        for (index, object) in joint.objects.into_iter().enumerate() {
            let body = &world.objects[object];
            displacement += [-1.0, 1.0][index] * (body.transform * joint.local_transform[index].position);
        }
        displacement
    }

    /// Computes a linear map from impulse to the associated velocity.
    fn velocity_per_impulse(constraint: &Constraint, world: &PixelWorld) -> Mat2 {
        let mut result = Mat2::ZERO;

        for (index, id) in constraint.objects().into_iter().enumerate() {
            let object = &world.objects[id];

            let scaled_tangent = constraint.local_position(world)[index].rotate(Vec2::from_angle(object.transform.rotation)).rotate(Vec2::Y);
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
    /// The total impulse applied along the normal over the entire step.
    pub total_impulse: Vec2
}

impl Constraint {
    /// Creates a new constraint from the provided source. Attempts to load
    /// the impulse from the cache, if possible.
    pub fn new(source: impl Into<ConstraintSource>, world: &PixelWorld, force_cache: &HashMap<ConstraintId, CachedImpulse>, substep_time: f32) -> Self {
        let mut result = Self {
            impulse: Vec2::ZERO,
            impulsive_torque: 0.0,
            original_relative_velocity: Vec2::ZERO,
            source: source.into(),
            total_impulse: Vec2::ZERO
        };

        result.initialize(world, force_cache, substep_time);

        result
    }

    /// Gets an identifier for tracking this constraint across frames.
    pub fn id(&self) -> ConstraintId {
        match &self.source {
            ConstraintSource::Contact(contact) => ConstraintId::ContactId(contact.id()),
            ConstraintSource::Joint(joint) => ConstraintId::JointId(joint.id),
        }
    }

    /// Gets the position on each object (in local space) where the force should be applied.
    pub fn local_position(&self, world: &PixelWorld) -> [Vec2; 2] {
        match &self.source {
            ConstraintSource::Contact(contact) => contact.local_position,
            ConstraintSource::Joint(joint) => [
                joint.local_transform[0].position,
                joint.local_transform[1].position,
            ]
        }
    }

    /// Gets the objects affected by the constraint.
    pub fn objects(&self) -> [ObjectId; 2] {
        match &self.source {
            ConstraintSource::Contact(contact) => contact.objects,
            ConstraintSource::Joint(joint) => joint.objects,
        }
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
            let relative_position = self.local_position(world)[index].rotate(Vec2::from_angle(body.transform.rotation));
            velocity += [-1.0, 1.0][index] * (body.velocity.linear + body.velocity.angular * relative_position.rotate(Vec2::Y));
        }
        velocity
    }

    /// Initializes the impulse and relative velocity when the contact is first created.
    fn initialize(&mut self, world: &PixelWorld, force_cache: &HashMap<ConstraintId, CachedImpulse>, substep_time: f32) {
        if let Some(cached) = force_cache.get(&self.id()) {
            self.impulse = substep_time * cached.force;
            self.impulsive_torque = substep_time * cached.torque;
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
    pub force: Vec2,
    /// The amount of torque applied for rotation constraints.
    pub torque: f32,
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

#[derive(Copy, Clone, Debug)]
struct MassProperties {
    inverse_inertia: f32,
    inverse_mass: f32,
}

#[derive(Debug, Default)]
struct BlockSolver {
    j_rows: BlockSolverArray<[Screw; 2]>,
    zetas: BlockSolverArray<f32>
}

impl BlockSolver {
    pub fn add_constraint(&mut self, j: [Screw; 2], zeta: f32) {
        self.j_rows.push(j);
        self.zetas.push(zeta);
    }

    pub fn solve(&self, v: [Screw; 2], masses: [MassProperties; 2]) -> BlockSolverArray<f32> {
        // todo: have this instead return a Screw to apply to both bodies
        // todo: refactor to use nalgebra to write out the matrices all nicely

        let b = self.calculate_j_v_plus_zeta(v);
        let mut result = BlockSolverArray::new();

        match self.j_rows.len() {
            0 => {},
            1 => {
                result.push(b[0] / self.k_entry(0, 0, masses));
            },
            2 => {
                let k = Mat2::from_cols(
                    vec2(self.k_entry(0, 0, masses), self.k_entry(1, 0, masses)),
                    vec2(self.k_entry(0, 1, masses), self.k_entry(1, 1, masses))
                );

                let lambda = k.inverse() * vec2(b[0], b[1]);
                result.push(lambda.x);
                result.push(lambda.y);
            },
            3 => {
                let k = Mat3::from_cols(
                    vec3(self.k_entry(0, 0, masses), self.k_entry(1, 0, masses), self.k_entry(2, 0, masses)),
                    vec3(self.k_entry(0, 1, masses), self.k_entry(1, 1, masses), self.k_entry(2, 1, masses)),
                    vec3(self.k_entry(0, 2, masses), self.k_entry(1, 2, masses), self.k_entry(2, 2, masses))
                );

                let lambda = k.inverse() * vec3(b[0], b[1], b[2]);
                result.push(lambda.x);
                result.push(lambda.y);
                result.push(lambda.z);
            },
            _ => unreachable!()
        }

        result
    }

    /// Computes entry `(i, j)` of the effective mass matrix `J * M^-1 * J^T`.
    fn k_entry(&self, i: usize, j: usize, masses: [MassProperties; 2]) -> f32 {
        let row_i = self.j_rows[i];
        let row_j = self.j_rows[j];

        (0..2).map(|body| {
            masses[body].inverse_mass * row_i[body].linear.dot(row_j[body].linear)
                + masses[body].inverse_inertia * row_i[body].angular * row_j[body].angular
        }).sum()
    }

    fn calculate_j_v_plus_zeta(&self, v: [Screw; 2]) -> BlockSolverArray<f32> {
        let mut result = BlockSolverArray::new();

        for i in 0..self.j_rows.len() {
            let j = self.j_rows[i];
            let zeta = self.zetas[i];
            result.push(j[0].dot(v[0]) + j[1].dot(v[1]) + zeta);
        }

        result
    }
}

type BlockSolverArray<T> = ArrayVec<T, 3>;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_fixed_joint() {
    }
}