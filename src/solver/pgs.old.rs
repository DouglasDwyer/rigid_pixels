use std::collections::HashMap;

use crate::*;


/*


/// A single force, generated from a constraint function `C`, that limits objects' motion.
#[derive(Clone, Debug)]
pub struct Constraint {
    /// The objects affected by the constraint.
    pub objects: [ObjectId; 2],
    /// Uniquely identifies this constraint.
    pub id: ConstraintId,
    /// The value of the constraint function `C`.
    pub c: f32,
    /// The Jacobian associated with this constraint.
    pub j: MotionPair,
    /// The range of allowed force values for this constraint.
    pub range: RangeInclusive<f32>
}

    /// Creates constraints to represent this contact.
    pub fn to_constraints(&self) -> [Constraint; 1] {
        [Constraint {
            objects: self.objects,
            id: ConstraintId { objects: self.objects, pixel_position: self.pixel_position },
            c: self.separation,
            j: MotionPair([
                Motion { linear: -self.normal, angular: -self.relative_position[0].perp_dot(self.normal) },
                Motion { linear: self.normal, angular: self.relative_position[1].perp_dot(self.normal) }
            ]),
            range: 0.0..=f32::MAX
        }]
    }

 */
#[derive(Debug)]
pub struct Pgs {
    /// The configuration to use.
    config: SolverConfig,
    /// A cache containing forces to use 
    force_cache: HashMap<ConstraintId, f32>,
    pub cached_constraints: Vec<Constraint>,
    pub cached_lambdas: Vec<f32>
}

impl Pgs {
    /// Creates a new PGS instance with the specified settings.
    pub fn new(config: SolverConfig) -> Self {
        let force_cache = HashMap::new();

        Self {
            config,
            force_cache,
            cached_constraints: Vec::new(),
            cached_lambdas: Vec::new()
        }
    }
    
    /// Solves the system of constraints and updates the velocities of bodies in the world.
    pub fn solve(&mut self, constraints: &[Constraint], world: &mut PixelWorld, delta_time: f32) {
        integrate_external_forces(world, delta_time);

        let mut lambdas = self.calculate_initial_lambdas(constraints);

        let eta = self.calculate_eta(constraints, world, delta_time);
        let b = self.calculate_b(constraints, world);
        let d = self.calculate_d(&b, constraints);
        let mut a = self.calculate_a(&lambdas, &b, constraints, world);

        for _ in 0..self.config.iterations {
            for (i, constraint) in constraints.iter().enumerate() {
                let delta_lambda = (eta[i] - constraint.j.dot(MotionPair(constraint.objects.map(|id| a[id])))) / d[i];
                let original_lambda = lambdas[i];
                lambdas[i] = (original_lambda + delta_lambda).clamp(*constraint.range.start(), *constraint.range.end());
                let clamped_delta_lambda = lambdas[i] - original_lambda;
                for (id, b_entry) in constraint.objects.into_iter().zip(*b[i]) {
                    a[id] += clamped_delta_lambda * b_entry;
                }
            }
        }

        self.cache_constraint_forces(constraints, &lambdas);
        self.apply_constraint_forces(constraints, &lambdas, world, delta_time);
        integrate_velocities(world, delta_time);

        self.cached_constraints = constraints.to_vec();
        self.cached_lambdas = lambdas.to_vec();
    }

    /// Integrates constraint forces into the velocities of all objects.
    fn apply_constraint_forces(&mut self, constraints: &[Constraint], lambdas: &[f32], world: &mut PixelWorld, delta_time: f32) {
        for (constraint, lambda) in constraints.into_iter().zip(lambdas.iter().copied()) {
            for ((object_id, motion), j) in constraint.objects.into_iter().zip(*constraint.j).zip(*constraint.j) {
                let object = &mut world.objects[object_id];
                let constraint_force = j * lambda;
                object.velocity += delta_time * Motion {
                    linear: object.body.inverse_mass() * constraint_force.linear,
                    angular: object.body.inverse_inertia_tensor() * constraint_force.angular
                };
            }
        }
    }

    /// Stores this frame's constraint forces for warm-starting next frame.
    fn cache_constraint_forces(&mut self, constraints: &[Constraint], lambdas: &[f32]) {
        if self.config.warm_starting {
            self.force_cache.clear();
            for (constraint, lambda) in constraints.into_iter().zip(lambdas.iter().copied()) {
                self.force_cache.insert(constraint.id, lambda);
            }
        }
    }

    /// Calculates `a`, the product of `B` and lambdas.
    fn calculate_a(&self, lambdas: &[f32], b: &[MotionPair], constraints: &[Constraint], world: &PixelWorld) -> SecondaryMap<ObjectId, Motion> {
        let mut result = SecondaryMap::with_capacity(world.capacity());
        for (i, (constraint, lambda)) in constraints.iter().zip(lambdas.iter().copied()).enumerate() {
            for (id, b_entry) in constraint.objects.into_iter().zip(*b[i]) {
                result.insert(id, lambda * b_entry + result.get(id).copied().unwrap_or_default());
            }
        }
        result
    }

    /// Calculates `B`, the product of inverse mass and J's transpose.
    fn calculate_b(&self, constraints: &[Constraint], world: &PixelWorld) -> Vec<MotionPair> {
        let mut result = Vec::with_capacity(constraints.len());

        for constraint in constraints {
            result.push(MotionPair([Motion {
                linear: world.objects[constraint.objects[0]].body.inverse_mass() * constraint.j[0].linear,
                angular: world.objects[constraint.objects[0]].body.inverse_inertia_tensor() * constraint.j[0].angular
            },
            Motion {
                linear: world.objects[constraint.objects[1]].body.inverse_mass() * constraint.j[1].linear,
                angular: world.objects[constraint.objects[1]].body.inverse_inertia_tensor() * constraint.j[1].angular
            }]));
        }

        result
    }

    /// Calculates the diagonal elements of `JB`.
    fn calculate_d(&self, b: &[MotionPair], constraints: &[Constraint]) -> Vec<f32> {
        let mut result = Vec::with_capacity(constraints.len());
        for (b, constraint) in b.iter().zip(constraints) {
            result.push(b.dot(constraint.j));
        }
        result
    }

    /// Calculates `eta`, the right-hand side of the PGS equation. This remains
    /// constant during iteration.
    fn calculate_eta(&self, constraints: &[Constraint], world: &PixelWorld, delta_time: f32) -> Vec<f32> {
        let mut result = Vec::with_capacity(constraints.len());
        for constraint in constraints {
            let zeta_term = if constraint.c < 0.0 { -self.config.baumgarte * constraint.c / delta_time } else { -constraint.c / delta_time };
            let j_term = -constraint.j.dot(MotionPair([world.objects[constraint.objects[0]].velocity, world.objects[constraint.objects[1]].velocity]));
            result.push((zeta_term + j_term) / delta_time);
        }
        result
    }

    /// Computes an initial guess for each constraint value.
    fn calculate_initial_lambdas(&self, constraints: &[Constraint]) -> Vec<f32> {
        if self.config.warm_starting {
            constraints.iter().map(|x| self.force_cache.get(&x.id).copied().unwrap_or_default())
                .collect()
        }
        else {
            vec![0.0; constraints.len()]
        }
    }
}