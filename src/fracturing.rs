use crate::*;

/*
pub fn apply_fracture(fracture: Fracture, world: &mut PixelWorld) {
    if let Some(object) = world.objects.get(fracture.object) {
        let mut copied_grid = object.body.grid().clone();
        let mut new_bodies = HashMap::new();
        
        for y in -destroy_radius..=destroy_radius {
            for x in -destroy_radius..=destroy_radius {
                let within_bounds = ivec2(x, y).length_squared() < destroy_radius.pow(2);
                let cell = (clicked_cell + ivec2(x, y)).as_uvec2();
                if within_bounds && object.body.grid().get_or_empty(cell) {
                    let seed = voronoi.evaluate(cell.as_vec2() + Vec2::splat(0.5));
                    if seed.distance_squared(clicked_position) < destroy_radius.pow(2) as f32 {
                        new_bodies.entry((seed.x.to_bits(), seed.y.to_bits())).or_insert_with(|| PixelGrid::new(object.body.grid().resolution()))
                            .set(cell, true);
                        copied_grid.set(cell, false);
                    }
                }
            }
        }

        let min_corner_transform = object.world_grid_matrix().transform_point2(Vec2::ZERO);

        let mut to_insert = Vec::new();
        for (new_grid, was_prev) in new_bodies.into_values().zip(std::iter::repeat(false)).chain(std::iter::once((copied_grid, true))) {
            for neighborhood in new_grid.neighborhoods() {
                let can_stay_immobile = was_prev && neighborhood.count_ones() > 50;
                let new_body = PixelBody::new(
                    neighborhood,
                    object.body.material(),
                    can_stay_immobile && object.body.inverse_mass() == 0.0,
                    can_stay_immobile && object.body.inverse_inertia_tensor() == 0.0);
                let new_transform = Transform::new(
                    object.world_grid_matrix().transform_vector2(new_body.local_center_of_mass()) + min_corner_transform,
                    object.transform.rotation);

                to_insert.push(PixelObject::new(new_body, random_color(), new_transform));
            }
        }

        world.objects.remove(id);
        
        for object in to_insert {
            world.objects.insert(object);
        }
    }
} */

pub struct VoronoiNoise2d {
    /// Converts from coordinates in input space
    /// to coordinates in the Voronoi diagram.
    transform: Mat3,
    transform_inverse: Mat3
}

impl VoronoiNoise2d {
    pub fn new(seed: u32, coordinate_transform: Mat3) -> Self {
        let transform = Mat3::from_translation(Self::noise_2d(Vec2::splat(seed as f32))) * coordinate_transform;
        let transform_inverse = transform.inverse();

        Self {
            transform,
            transform_inverse
        }
    }

    /// Determines the closest seed to `position`.
    /// Returns the seed's location in Voronoi space.
    pub fn evaluate(&self, position: Vec2) -> Vec2 {
        let uv = self.transform.transform_point2(position);
        let base_cell = (uv - Vec2::splat(0.5)).floor();

        let mut best_seed = Vec2::NAN;
        let mut best_distance_sq = f32::MAX;

        for y in 0..2 {
            for x in 0..2 {
                let cell = base_cell + ivec2(x, y).as_vec2();
                let seed = cell + (0.5 * Self::noise_2d(cell) + Vec2::splat(0.25));

                let distance_sq = seed.distance_squared(uv);
                if distance_sq < best_distance_sq {
                    best_seed = seed;
                    best_distance_sq = distance_sq;
                }
            }
        }

        self.transform_inverse.transform_point2(best_seed)
    }

    /// Generates a pseudorandom value. Returns a different
    /// result for different values of `p`.
    fn noise_2d(p: Vec2) -> Vec2 {
        vec2(vec2(591.32, 154.077).dot(p).sin(),
            vec2(391.32, 49.077).dot(p).cos()).fract_gl()
    }
}