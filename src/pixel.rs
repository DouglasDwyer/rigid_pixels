use bitflags::*;
use bitvec::prelude::*;
use macroquad::prelude::*;
use slotmap::*;

new_key_type! {
    /// Uniquely identifies an object in a [`PixelWorld`].
    pub struct ObjectId;
}

/// Stores a collection of objects in a world.
pub type PixelWorld = SlotMap<ObjectId, PixelObject>;

/// Describes the current location and orientation of an object.
#[derive(Copy, Clone, Debug, Default)]
pub struct Transform {
    /// The position of the object's center of mass.
    pub position: Vec2,
    /// The rotation from the origin.
    pub rotation: f32
}

/// Describes how quickly an object is moving, and in what direction.
#[derive(Copy, Clone, Debug, Default)]
pub struct Velocity {
    /// The linear velocity in units per second.
    pub linear: Vec2,
    /// The rotational velocity in radians per second.
    pub angular: f32
}

/// Efficiently gathers all external forces over the course of a frame.
#[derive(Copy, Clone, Debug, Default)]
pub struct ForceAccumulator {
    /// The total amount of force added (with respect to the center of mass).
    pub force: Vec2,
    /// The total amount of torque added (with respect to the center of mass).
    pub torque: f32
}

/// An object in the game world.
#[derive(Debug)]
pub struct PixelObject {
    /// The rigidbody associated with this object.
    pub body: PixelBody,
    /// The color of the object.
    pub color: Color,
    /// Any external forces added to the object.
    pub forces: ForceAccumulator,
    /// The location of the object.
    pub transform: Transform,
    /// The motion of the object.
    pub velocity: Velocity
}

impl PixelObject {
    /// Creates a new object.
    pub fn new(body: PixelBody, color: Color, transform: Transform) -> Self {
        Self {
            body,
            color,
            forces: ForceAccumulator::default(),
            transform,
            velocity: Velocity::default()
        }
    }

    /// Gets a matrix that converts from grid space to world space.
    pub fn world_grid_matrix(&self) -> Mat3 {
        self.world_model_matrix() * Mat3::from_translation(-self.body.local_center_of_mass())
    }

    /// Gets a matrix that converts from grid space to world space.
    pub fn world_model_matrix(&self) -> Mat3 {
        Mat3::from_scale_angle_translation(Vec2::ONE, self.transform.rotation, self.transform.position)
    }
}

/// Holds intrinsic data about a rigid object.
#[derive(Clone, Debug, Default)]
pub struct PixelBody {
    /// The center of mass in the grid frame.
    local_center_of_mass: Vec2,
    /// The invrese of `m`.
    inverse_mass: f32,
    /// The inverse of `I`.
    inverse_inertia_tensor: f32,
    /// A 2D grid with neighbor information for each pixel. Same size as [`Self::grid`].
    neighbors: Vec<PixelNeighbors>,
    /// A list of all positions in `grid` containing corner pixels.
    corners: Vec<UVec2>,
    /// Which pixels are part of the object?
    grid: PixelGrid,
    /// A Euclidean radius that bounds the object.
    radius: f32
}

impl PixelBody {
    /// Creates a new body for the given geometry.
    /// If `fixed` is true, then the body is presumed not to move.
    pub fn new(grid: PixelGrid, fixed: bool) -> Self {
        let mut result = Self {
            local_center_of_mass: Vec2::ZERO,
            neighbors: Vec::new(),
            corners: Vec::new(),
            grid,
            inverse_mass: 0.0,
            inverse_inertia_tensor: 0.0,
            radius: 0.0
        };

        result.rebuild(fixed);

        result
    }

    /// The grid of pixels associated with the body.
    pub fn grid(&self) -> &PixelGrid {
        &self.grid
    }

    /// Gets the center of mass in grid space.
    pub fn local_center_of_mass(&self) -> Vec2 {
        self.local_center_of_mass
    }

    /// Which adjacent pixels are set?
    pub fn neighbors(&self, position: UVec2) -> PixelNeighbors {
        self.neighbors[(position.x + position.y * self.grid.size().x) as usize]
    }

    /// Recalculates all acceleration structure data for this body.
    fn rebuild(&mut self, fixed: bool) {
        self.neighbors.clear();
        self.corners.clear();

        let mut min_val = UVec2::splat(u32::MAX);
        let mut max_val = UVec2::ZERO;
        for y in 0..self.grid.size().y {
            for x in 0..self.grid.size().x {
                let neighbors = Self::calculate_neighbors(&self.grid, uvec2(x, y));
                let classification = neighbors.kind();

                if classification == PixelKind::Corner {
                    self.corners.push(uvec2(x, y));
                }

                if classification != PixelKind::Empty {
                    min_val = min_val.min(uvec2(x, y));
                    max_val = max_val.max(uvec2(x, y) + UVec2::ONE);
                }

                self.neighbors.push(neighbors);
            }
        }

        if !fixed {
            self.update_mass_and_inertia();
        }

        self.radius = if min_val.cmplt(max_val).all() {
            (min_val.min(max_val) - max_val).as_vec2().length() / 2.0
        }
        else {
            0.0
        };
    }

    /// Calculates the mass and moment of inertia for the body.
    fn update_mass_and_inertia(&mut self) {
        let summed_positions = self.grid.iter().fold(UVec3::ZERO, |acc, v| acc + uvec3(v.x, v.y, 1));
        let sz = summed_positions.z as f32;
        let mass = 1.0 * sz;
        self.inverse_mass = mass.recip();
        self.local_center_of_mass = sz.recip() * summed_positions.xy().as_vec2() + Vec2::splat(0.5);
        
        let base_inertia = mass / 6.0;
        let inertial_displacement = self.grid.iter().fold(0.0, |acc, x| acc + (x.as_vec2() + Vec2::splat(0.5)  - self.local_center_of_mass).length_squared());
        let inertia_tensor = base_inertia + inertial_displacement;
        
        self.inverse_inertia_tensor = inertia_tensor.recip();
    }

    /// Determines the neighbors of the pixel at `position`.
    fn calculate_neighbors(grid: &PixelGrid, position: UVec2) -> PixelNeighbors {
        let mut result = 0;
        for (index, offset) in [IVec2::ZERO, -IVec2::X, IVec2::X, -IVec2::Y, IVec2::Y].into_iter().enumerate() {
            if grid.get_or_empty(position + offset.as_uvec2()) {
                result |= 1 << index;
            }
        }
        PixelNeighbors::from_bits_truncate(result)
    }
}

bitflags! {
    /// Denotes which neighbors of a pixel are `true`.
    #[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
    pub struct PixelNeighbors: u8 {
        /// Whether the pixel itself is `true`.
        const SELF = 1 << 0;
        /// The `-X` direction.
        const LEFT = 1 << 1;
        /// The `+X` direction.
        const RIGHT = 1 << 2;
        /// The `-Y` direction.
        const DOWN = 1 << 3;
        /// The `+Y` direction.
        const UP = 1 << 4;
    }
}

impl PixelNeighbors {
    /// When a pixel has these neighbors, what kind is it considered?
    pub fn kind(&self) -> PixelKind {
        /// A table for classification.
        const CLASSIFICATIONS: [PixelKind; 16] = [
            PixelKind::Corner,  //   - 
                                // - * - 
                                //   -
            PixelKind::Corner,  //   -
                                // * * -
                                //   -
            PixelKind::Corner,  //   -
                                // - * *
                                //   -
            PixelKind::EdgeY,   //   -
                                // * * *
                                //   -
            PixelKind::Corner,  //   -
                                // - * -
                                //   *
            PixelKind::Corner,  //   -
                                // * * -
                                //   *
            PixelKind::Corner,  //   -
                                // - * *
                                //   *
            PixelKind::EdgeY,   //   -
                                // * * *
                                //   *
            PixelKind::Corner,  //   *
                                // - * -
                                //   -
            PixelKind::Corner,  //   *
                                // * * -
                                //   -
            PixelKind::Corner,  //   *
                                // - * *
                                //   -
            PixelKind::EdgeY,   //   *
                                // * * *
                                //   -
            PixelKind::EdgeX,   //   *
                                // - * -
                                //   *
            PixelKind::EdgeX,   //   *
                                // * * -
                                //   *
            PixelKind::EdgeX,   //   *
                                // - * *
                                //   *
            PixelKind::Interior,  //   *
                                  // * * *
                                  //   *
        ];
        if self.contains(PixelNeighbors::SELF) {
            CLASSIFICATIONS[(self.bits() >> 1) as usize]
        }
        else {
            PixelKind::Empty
        }        
    }
}

/// Describes the local geometry of a pixel. The only collisions that can happen
/// are between corners and either corners or edges.
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
#[repr(u8)]
pub enum PixelKind {
    /// The pixel is set to `false`.
    Empty,
    /// The pixel has two adjacent neighbors that are `false`.
    Corner,
    /// The normal faces in the x direction (both neighbors in the y direction are `true`). 
    EdgeX,
    /// The normal faces in the y direction (both neighbors in the x direction are `true`).
    EdgeY,
    /// All neighbors are `true`.
    Interior
}

/// A 2D grid of `bool`s stored in a bitmap.
#[derive(Clone, Debug, PartialEq, Default)]
pub struct PixelGrid {
    /// The width and height of the grid.
    resolution: UVec2,
    /// The grid values in `x`-major order.
    values: BitVec
}

impl PixelGrid {
    /// Creates a new grid of `size` initialized to `false`.
    pub fn new(resolution: UVec2) -> Self {
        let mut values = BitVec::new();
        values.resize(resolution.element_product() as usize, false);

        Self {
            resolution,
            values
        }
    }

    /// Gets the value of the pixel at `position`.
    pub fn get(&self, position: UVec2) -> bool {
        assert!(position.cmplt(self.resolution).all(), "{position} was out-of-bounds for grid of resolution {}", self.resolution);
        self.values[(position.x as u32 + self.resolution.x * position.y as u32) as usize]
    }

    /// Gets the value of the pixel at `position`, or returns `false` if it was out-of-bounds.
    pub fn get_or_empty(&self, position: UVec2) -> bool {
        if position.cmplt(self.resolution).all() {
            self.values[(position.x as u32 + self.resolution.x * position.y as u32) as usize]
        }
        else {
            false
        }
    }

    /// Sets the value of the pixel at `position`.
    pub fn set(&mut self, position: UVec2, value: bool) -> bool {
        self.values.replace((position.x as u32 + self.resolution.x * position.y as u32) as usize, value)
    }

    /// The width and length of the grid.
    pub fn size(&self) -> UVec2 {
        self.resolution
    }

    /// Gets an iterator over the positions of pixels with value `true`.
    pub fn iter(&self) -> impl '_ + Iterator<Item = UVec2> {
        self.values.iter_ones().map(|index| uvec2(index as u32 % self.resolution.x, index as u32 / self.resolution.x))
    }
}