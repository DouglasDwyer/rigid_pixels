use crate::*;
use std::collections::*;

/// Stores a set of pre-generated fragments that determine how an object is split.
pub struct FracturePattern {
    /// A grid of separate "pieces" into which the object should be broken.
    /// These define the shape and location of each fragment.
    /// When splitting an object, the algorithm begins at [`IVec2::ZERO`] and copies
    /// out any pixels from that piece. If some pixels were found, then proceeds
    /// in a breadth-first manner to adjacent cells.
    cells: HashMap<IVec2, FractureCell>,
    /// Pixels to destroy immediately.
    destroy_pixels: HashSet<IVec2>
}

impl FracturePattern {
    /// Generates a pattern that will split an object up into rectangles.
    /// Each rectangle will have `cell_size`. Everything within `radius_pixels`
    /// will be affected.
    pub fn grid(radius_pixels: u32, cell_size: UVec2) -> Self {
        let cell_size_ivec2 = cell_size.as_ivec2();
        let radius_pixels_i32 = radius_pixels as i32;
        let radius_cells = (IVec2::splat(radius_pixels_i32) + cell_size_ivec2 - IVec2::ONE) / cell_size_ivec2;
        
        let mut cells = HashMap::default();
        for cell in iter_grid_inclusive_i32(-radius_cells, radius_cells) {
            let offset = cell_size_ivec2 * cell;

            if offset.length_squared() <= radius_pixels_i32.pow(2) {
                let mut grid = PixelGrid::new(cell_size);
                grid.fill(true);
                cells.insert(cell, FractureCell {
                    grid,
                    offset
                });
            }
        }

        Self {
            cells,
            destroy_pixels: [IVec2::ZERO].into()
        }
    }

    pub fn voronoi(seed: u32, radius_pixels: u32, angle: f32, scale: Vec2) -> Self {
        let radius_pixels_i32 = radius_pixels as i32;
        let grid_resolution = UVec2::splat(2 * radius_pixels + 1);
        let offset = IVec2::splat(-radius_pixels_i32);

        let transform = Mat3::from_scale(scale) * Mat3::from_angle(angle);
        let radius_voronoi = (radius_pixels as f32 * scale.min_element() - 1.0).max(1.0);
        
        let mut cells = HashMap::default();
        for pixel in iter_grid_inclusive_i32(IVec2::splat(-radius_pixels_i32), IVec2::splat(radius_pixels_i32)) {
            let (cell, point) = voronoi_2d(seed, transform.transform_point2(pixel.as_vec2() + Vec2::splat(0.2)));
            if point.length_squared() < radius_voronoi.powi(2) {
                cells.entry(cell).or_insert_with(|| FractureCell {
                    grid: PixelGrid::new(grid_resolution),
                    offset
                }).grid.set((pixel - offset).as_uvec2(), true);
            }
        }

        Self {
            cells,
            destroy_pixels: [IVec2::ZERO].into()
        }
    }

    /// Causes everything listed in `pixels` to be destroyed immediately upon impact.
    pub fn destroy_pixels(mut self, pixels: impl IntoIterator<Item = IVec2>) -> Self {
        self.destroy_pixels = pixels.into_iter().collect();
        self
    }

    /// Causes everything within `radius_pixels` to be destroyed immediately upon impact.
    pub fn destroy_radius(self, radius_pixels: u32) -> Self {
        let radius_i32 = radius_pixels as i32;
        self.destroy_pixels(iter_grid_inclusive_i32(IVec2::splat(-radius_i32), IVec2::splat(radius_i32))
            .filter(|x| x.length_squared() <= radius_i32.pow(2)))
    }

    /// Gets the maximum number of pieces that this fragment pattern will generate.
    pub fn piece_len(&self) -> usize {
        self.cells.len()
    }

    /// Breaks an object into pieces based upon this fracture pattern.
    /// Deletes the body at `id` and replaces it with smaller pieces.
    pub fn apply(&self, fracture: Fracture, world: &mut PixelWorld) {
        if let Some(object) = world.objects.get(fracture.object) {
            let mut copied_grid = object.body.grid().clone();
            
            for destroy_offset in self.destroy_pixels.iter().copied() {
                let object_pixel = (fracture.pixel_position.as_ivec2() + destroy_offset).as_uvec2();
                if object_pixel.cmplt(copied_grid.resolution()).all() {
                    copied_grid.set(object_pixel, false);
                }
            }

            let mut new_bodies = Vec::new();
            let mut seen_cells = HashSet::new();
            let mut cells_to_check = vec![IVec2::ZERO];

            while let Some(next_position) = cells_to_check.pop() {
                if let Some(cell) = self.cells.get(&next_position)
                    && seen_cells.insert(next_position) {
                    let mut new_body = PixelGrid::new(copied_grid.resolution());

                    let mut check_neighbors = false;
                    let mut found_pixel = false;

                    for pattern_pixel in iter_grid_u32(UVec2::ZERO, cell.grid.resolution()) {
                        if cell.grid.get(pattern_pixel) {
                            let fracture_offset = pattern_pixel.as_ivec2() + cell.offset;
                            let object_pixel = (fracture.pixel_position.as_ivec2() + fracture_offset).as_uvec2();
                            
                            check_neighbors |= self.destroy_pixels.contains(&fracture_offset);

                            if copied_grid.get_or_empty(object_pixel) {
                                copied_grid.set(object_pixel, false);
                                new_body.set(object_pixel, true);
                                found_pixel = true;
                                check_neighbors = true;
                            }
                        }
                    }

                    if check_neighbors {
                        cells_to_check.extend([IVec2::NEG_X, IVec2::NEG_Y, IVec2::X, IVec2::Y].map(|v| next_position + v));
                    }

                    if found_pixel {
                        new_bodies.push(new_body);
                    }
                }
            }

            world.split_object(fracture.object, new_bodies.into_iter().chain(std::iter::once(copied_grid))
                .flat_map(|x| x.neighborhoods()));
        }
    }
}

/// A single piece within a [`FracturePattern`].
#[derive(Clone, Debug)]
struct FractureCell {
    /// The grid defining which pixels are included in this fragment.
    pub grid: PixelGrid,
    /// The offset of the minimum cell in [`Self::grid`] from the origin (the fracture point).
    pub offset: IVec2
}

/// Returns the Voronoi seed closest to `p`.
/// The noise is sampled from a pseudorandom grid
/// with a cell spacing of `1.0`.
fn voronoi_2d(seed: u32, p: Vec2) -> (IVec2, Vec2) {
    let base_cell = (p - Vec2::splat(0.5)).floor().as_ivec2();
    let seed_translation = 1000.0 * noise_2d(Vec2::splat(seed as f32));

    let mut best_point = (IVec2::ZERO, Vec2::NAN);
    let mut best_distance_sq = f32::MAX;

    for y in 0..2 {
        for x in 0..2 {
            let cell_i32 = base_cell + ivec2(x, y);
            let cell = cell_i32.as_vec2();
            let seed = cell + (0.5 * noise_2d(cell + seed_translation) + Vec2::splat(0.25));

            let distance_sq = seed.distance_squared(p);
            if distance_sq < best_distance_sq {
                best_point = (cell_i32, seed);
                best_distance_sq = distance_sq;
            }
        }
    }

    best_point
}

/// Generates a pseudorandom value. Returns a different
/// result for different values of `p`.
fn noise_2d(p: Vec2) -> Vec2 {
    vec2(vec2(591.32, 154.077).dot(p).sin(),
        vec2(391.32, 49.077).dot(p).cos()).fract_gl()
}

/// Returns an iterator over all points in the box between `start` and `end`, inclusive.
fn iter_grid_inclusive_i32(start: IVec2, end: IVec2) -> impl Iterator<Item = IVec2> {
    (start.y..=end.y).flat_map(move |y| (start.x..=end.x).map(move |x| ivec2(x, y)))
}

/// Returns an iterator over all points in the box between `start` and `end`, exclusive.
fn iter_grid_u32(start: UVec2, end: UVec2) -> impl Iterator<Item = UVec2> {
    (start.y..end.y).flat_map(move |y| (start.x..end.x).map(move |x| uvec2(x, y)))
}