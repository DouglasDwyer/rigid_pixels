use arrayvec::ArrayVec;

use crate::solver::*;
use std::collections::*;
use nalgebra::*;

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
            //self.solve_fracture(constraint, world, delta_time);
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
                Self::apply_body_impulse(constraint, world, -excess_impulse * contact.normal);
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

            let prior_normal_impulse = constraint.impulses[1].linear.dot(contact.normal);
            let total_impulse = (prior_normal_impulse + required_impulse).max(0.0);
            let delta_impulse = total_impulse - prior_normal_impulse;

            constraint.total_impulse += delta_impulse * contact.normal;
            Self::apply_body_impulse(constraint, world, delta_impulse * contact.normal);
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

        let static_friction_impulse = constraint.impulses[1].linear - impulse_per_velocity * relative_velocity;
        
        let normal_impulse = static_friction_impulse.dot(contact.normal);
        let planar_impulse = static_friction_impulse.reject_from_normalized(contact.normal);
        let planar_impulse_length = planar_impulse.length();

        let total_impulse_unclamped = if contact.material.friction * normal_impulse < planar_impulse_length {
            let relative_velocity_without_impulse = relative_velocity - velocity_per_impulse * constraint.impulses[1].linear;
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

        let delta_impulse = total_impulse - constraint.impulses[1].linear;

        if !Self::approx_zero(delta_impulse) {
            Self::apply_body_impulse(constraint, world, delta_impulse);
        }
    }

    /// Solves a joint-based velocity constraint.
    fn solve_joint_velocity(&self, constraint: &mut Constraint, world: &mut PixelWorld, delta_time: f32, apply_baumgarte: bool) {
        // For now, just implement a translational fixed joint.

        // Undo prior guess
        Self::apply_impulse(constraint, world, constraint.impulses.map(|x| -x));
        let ConstraintSource::Joint(joint) = &constraint.source else { panic!() };

        let objects = joint.objects.map(|id| &world.objects[id]);

        let mut solver = BlockSolver::new(
            [objects[0].velocity, objects[1].velocity],
            [objects[0].body.mass_properties(), objects[1].body.mass_properties()]
        );

        let baumgarte_factor = if apply_baumgarte { (self.config.baumgarte / self.config.substeps as f32) / delta_time } else { 0.0 };

        let relative_displacement = Self::relative_displacement(joint, world);
        let relative_angle = Vec2::angle_between(Vec2::from_angle(objects[0].transform.rotation), Vec2::from_angle(objects[1].transform.rotation));

        let relative_offsets = [
            objects[0].transform * joint.local_transform[0].position - objects[0].transform.position,
            objects[1].transform * joint.local_transform[1].position - objects[1].transform.position
        ];

        solver.add_constraint(BlockConstraint {
            j: [
                Screw { linear: Vec2::NEG_X, angular: relative_offsets[0].y },
                Screw { linear: Vec2::X, angular: -relative_offsets[1].y }
            ],
            lambda_limits: -f32::INFINITY..=f32::INFINITY,
            zeta: -baumgarte_factor * relative_displacement.x
        });
        
        solver.add_constraint(BlockConstraint {
            j: [
                Screw { linear: Vec2::NEG_Y, angular: -relative_offsets[0].x },
                Screw { linear: Vec2::Y, angular: relative_offsets[1].x }
            ],
            lambda_limits: -f32::INFINITY..=f32::INFINITY,
            zeta: -baumgarte_factor * relative_displacement.y
        });

        if false {
            solver.add_constraint(BlockConstraint {
                j: [
                    Screw { linear: Vec2::ZERO, angular: -1.0 },
                    Screw { linear: Vec2::ZERO, angular: 1.0 }
                ],
                lambda_limits: -0.0..=0.0,
                zeta: -baumgarte_factor * relative_angle
            });
        }

        let impulses = solver.solve();
        
        Self::apply_impulse(constraint, world, impulses);
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

    /// Caches the forces from this tick. The forces will be used in warm-starting
    /// the constraints for the next tick.
    fn cache_constraint_forces(&mut self, constraints: &[Constraint], substep_time: f32) {
        self.force_cache.clear();
        if self.config.warm_starting {
            for constraint in constraints {
                self.force_cache.insert(constraint.id(),
                    CachedImpulse(constraint.impulses.map(|impulse| impulse * substep_time.recip())));
            }
        }
    }
    
    /// Stores the total impulse from the substep.
    fn sum_total_impulse(&mut self, constraints: &mut [Constraint]) {
        for constraint in constraints {
            constraint.total_impulse += constraint.impulses[1].linear;
        }
    }

    /// Applies the total impulse stored in each constraint to the objects in the world.
    fn apply_impulses(&self, constraints: &mut [Constraint], world: &mut PixelWorld) {
        if self.config.warm_starting {
            for constraint in constraints {
                let impulses = constraint.impulses;
                constraint.impulses = [Screw::default(); 2];
                Self::apply_impulse(constraint, world, impulses);
            }
        }
    }

    /// Applies an impulse from a contact. Updates the contact's total impulse and the velocity of
    /// the associated bodies in the `world`.
    fn apply_body_impulse(constraint: &mut Constraint, world: &mut PixelWorld, impulse: Vec2) {
        for (index, id) in constraint.objects().into_iter().enumerate() {
            let relative_position = constraint.local_position(world)[index].rotate(Vec2::from_angle(world.objects[id].transform.rotation));
            let object = &mut world.objects[id];
            let body_impulsive_torque = relative_position.perp_dot(impulse);
            let sign = [-1.0, 1.0][index];
            object.velocity += Screw {
                linear: sign * object.body.inverse_mass() * impulse,
                angular: sign * object.body.inverse_inertia_tensor() * body_impulsive_torque
            };

            constraint.impulses[index] += Screw {
                linear: sign * impulse,
                angular: sign * body_impulsive_torque
            };
        }
    }

    /// Applies an impulse from a contact. Updates the contact's total impulse and the velocity of
    /// the associated bodies in the `world`.
    fn apply_impulse(constraint: &mut Constraint, world: &mut PixelWorld, impulses: [Screw; 2]) {
        for (index, id) in constraint.objects().into_iter().enumerate() {
            let object = &mut world.objects[id];
            object.velocity += Screw {
                linear: object.body.inverse_mass() * impulses[index].linear,
                angular: object.body.inverse_inertia_tensor() * impulses[index].angular
            };
                
            constraint.impulses[index] += impulses[index];
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
    /// The amount of impulse applied _at each object's center of mass_.
    pub impulses: [Screw; 2],
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
            impulses: [Screw::default(); 2],
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
            self.impulses = cached.0.map(|force| substep_time * force);
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
struct CachedImpulse([Screw; 2]);

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


#[derive(Clone, Debug)]
struct BlockConstraint {
    j: [Screw; 2],
    lambda_limits: RangeInclusive<f32>,
    zeta: f32
}

struct BlockSolver {
    lambda_max: Vector3<f32>,
    lambda_min: Vector3<f32>,
    len: usize,
    j_t: Matrix6<f32>,
    m: [MassProperties; 2],
    v: Vector6<f32>,
    zeta: Vector3<f32>
}

impl BlockSolver {
    pub fn new(v: [Screw; 2], m: [MassProperties; 2]) -> Self {
        Self {
            lambda_max: Vector3::zeros(),
            lambda_min: Vector3::zeros(),
            len: 0,
            j_t: Matrix6::zeros(),
            m,
            v: Vector6::new(
                v[0].linear.x,
                v[0].linear.y,
                v[0].angular,
                v[1].linear.x,
                v[1].linear.y,
                v[1].angular,
            ),
            zeta: Vector3::zeros()
        }
    }

    pub fn add_constraint(&mut self, constraint: BlockConstraint) {
        assert!(self.len < 3);

        self.lambda_max[self.len] = *constraint.lambda_limits.end();
        self.lambda_min[self.len] = *constraint.lambda_limits.start();
        self.j_t.set_column(self.len, &Vector6::new(
            constraint.j[0].linear.x,
            constraint.j[0].linear.y,
            constraint.j[0].angular,
            constraint.j[1].linear.x,
            constraint.j[1].linear.y,
            constraint.j[1].angular,
        ));
        self.zeta[self.len] = constraint.zeta;

        self.len += 1;
    }

    pub fn solve(&self) -> [Screw; 2] {
        let m_inv = Matrix6::from_diagonal(&Vector6::new(
            self.m[0].inverse_mass,
            self.m[0].inverse_mass,
            self.m[0].inverse_inertia_tensor,
            self.m[1].inverse_mass,
            self.m[1].inverse_mass,
            self.m[1].inverse_inertia_tensor
        ));

        match self.len {
            0 => [Screw::default(); _],
            1 => self.solve_cols::<1>(m_inv),
            2 => self.solve_cols::<2>(m_inv),
            3 => self.solve_cols::<3>(m_inv),
            _ => unreachable!()
        }
    }

    fn solve_cols<const C: usize>(&self, m_inv: Matrix6<f32>) -> [Screw; 2] {
        // Solve for λ:
        // JM⁻¹Jᵀλ = ζ - JV

        let j = self.j_t.fixed_columns::<C>(0).into_owned().transpose();
        let v = self.v;
        let zeta = self.zeta.fixed_rows::<C>(0).into_owned();

        let lambda_max = self.lambda_max.fixed_rows::<C>(0).into_owned();
        let lambda_min = self.lambda_min.fixed_rows::<C>(0).into_owned();

        let k = j * m_inv * j.transpose();
        let rhs = zeta - j * v;

        let cholesky = Cholesky::new(k).expect("cholesky decomp failed");
        let mut lambda = cholesky.solve(&rhs);

        for i in 0..C {
            lambda[i] = lambda[i].clamp(self.lambda_min[i], self.lambda_max[i]);
        }
        
        let impulses = j.transpose() * lambda;

        [
            Screw { linear: vec2(impulses[0], impulses[1]), angular: impulses[2] },
            Screw { linear: vec2(impulses[3], impulses[4]), angular: impulses[5] },
        ]
    }
}

type BlockSolverArray<T> = ArrayVec<T, 3>;