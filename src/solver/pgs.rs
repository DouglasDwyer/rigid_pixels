use crate::*;

/// Determines how PGS will behave.
#[derive(Copy, Clone, Debug)]
pub struct PgsConfig {
    /// The Baumgarte factor to apply.
    pub baumgarte: f32,
    /// The number of solver iterations to perform.
    pub iterations: u32
}

#[derive(Debug)]
pub struct Pgs {
    /// The configuration to use.
    config: PgsConfig
}

impl Pgs {
    /// Creates a new PGS instance with the specified settings.
    pub fn new(config: PgsConfig) -> Self {
        Self {
            config
        }
    }
    
    /// Solves the system of constraints and updates the velocities of bodies in the world.
    pub fn solve(&self, constraints: &[Constraint], world: &mut PixelWorld, delta_time: f32) {
        for object in world.values_mut() {
            object.velocity += delta_time * Motion {
                linear: object.body.inverse_mass() * object.forces.force,
                angular: object.body.inverse_inertia_tensor() * object.forces.torque
            };
        }

        let mut lambdas = vec![0.0; constraints.len()];

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

        for (constraint, lambda) in constraints.into_iter().zip(lambdas.iter().copied()) {
            for ((object_id, motion), j) in constraint.objects.into_iter().zip(*constraint.j).zip(*constraint.j) {
                let object = &mut world[object_id];
                let constraint_force = j * lambda;
                object.velocity += delta_time * Motion {
                    linear: object.body.inverse_mass() * constraint_force.linear,
                    angular: object.body.inverse_inertia_tensor() * constraint_force.angular
                };
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
                linear: world[constraint.objects[0]].body.inverse_mass() * constraint.j[0].linear,
                angular: world[constraint.objects[0]].body.inverse_inertia_tensor() * constraint.j[0].angular
            },
            Motion {
                linear: world[constraint.objects[1]].body.inverse_mass() * constraint.j[1].linear,
                angular: world[constraint.objects[1]].body.inverse_inertia_tensor() * constraint.j[1].angular
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
            let zeta_term = -self.config.baumgarte * constraint.c / delta_time;
            let j_term = -constraint.j.dot(MotionPair([world[constraint.objects[0]].velocity, world[constraint.objects[1]].velocity]));
            result.push((zeta_term + j_term) / delta_time);
        }
        result
    } 
}