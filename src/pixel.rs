use bitflags::*;
use bitvec::prelude::*;
use crate::*;
use macroquad::prelude::*;
use slotmap::*;
use std::cell::*;
use std::ops::*;

new_key_type! {
    /// Uniquely identifies an object in a [`PixelWorld`].
    pub struct ObjectId;
}

/// Holds physical properties of a material.
#[derive(Copy, Clone, Debug)]
pub struct PixelMaterial {
    /// The maximum impulse that this material can withstand before breaking.
    pub breaking_impulse: f32,
    /// How the material should split when [`Self::breaking_impulse`] is exceeded.
    pub fracture_pattern: FracturePatternId,
    /// The friction along the material's surface.
    pub friction: f32,
    /// The bounciness of the material.
    pub restitution: f32
}

impl PixelMaterial {
    /// Combines two materials to describe the surface properties when they meet.
    pub fn mix(a: Self, b: Self) -> PixelMaterialPair {
        PixelMaterialPair {
            breaking_impulses: [a.breaking_impulse, b.breaking_impulse],
            fracture_patterns: [a.fracture_pattern, b.fracture_pattern],
            friction: 0.5 * (a.friction + b.friction),
            restitution: a.restitution.max(b.restitution)
        }
    }
}

/// Describes the physical properties at the contact point
/// between two materials.
#[derive(Copy, Clone, Debug)]
pub struct PixelMaterialPair {
    /// The breaking impulse of each object.
    pub breaking_impulses: [f32; 2],
    /// The fracture kind of each object.
    pub fracture_patterns: [FracturePatternId; 2],
    /// The combined friction.
    pub friction: f32,
    /// The combined restitution.
    pub restitution: f32
}

/// Holds the world being simulated.
#[derive(Debug, Default)]
pub struct PixelWorld {
    /// Joints which connect objects together.
    pub joints: Vec<Joint>,
    /// The objects in the world.
    pub objects: SlotMap<ObjectId, PixelObject>,
}

impl PixelWorld {
    /// Adds a joint at world `point` connecting `objects`.
    pub fn insert_joint(&mut self, objects: [ObjectId; 2], point: Transform, descriptor: JointDescriptor) {
        let id = JointId(self.joints.iter().fold(0, |acc, x| acc.max(x.id.0)) + 1);
        
        self.joints.push(Joint {
            objects,
            id,
            local_transform: objects.map(|id| self.objects[id].transform.inverse() * point),
            max_force: descriptor.max_force,
            max_torque: descriptor.max_torque,
            translation_max: descriptor.translation_max,
            translation_min: descriptor.translation_min,
            rotation_max: descriptor.rotation_max,
            rotation_min: descriptor.rotation_min
        });
    }

    /// Breaks the provided object up into components specified by `pieces`.
    /// Creates new objects in the world - one for each piece - which inherit
    /// their transform and velocity from the object with `id`. All grids specified by
    /// `pieces` must be the same size as the object, or this will panic.
    pub fn split_object(&mut self, id: ObjectId, pieces: impl IntoIterator<Item = PixelGrid>) {
        /// The number of pixels that a piece must have in order to stay static.
        /// All pieces less than this threshold become dynamic rigid bodies.
        const IMMOBILITY_THRESHOLD: usize = 100;

        let new_objects = self.objects[id].split(pieces, IMMOBILITY_THRESHOLD);

        self.objects.remove(id);

        for new_object in new_objects {
            self.objects.insert(new_object);
        }
    }
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
    /// The location of the object's center of mass.
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

    /// Adds a force in world space to the body.
    pub fn add_force(&mut self, point: Vec2, force: Vec2) {
        self.forces.force += force;
        self.forces.torque += (point - self.transform.position).perp_dot(force);
    }

    /// Breaks this object up into components specified by `pieces`.
    /// Returns a list of new objects - one for each piece - which inherit
    /// their transform and velocity from this object. All grids specified by
    /// `pieces` must be the same size as `self`, or this will panic.
    pub fn split(&self, pieces: impl IntoIterator<Item = PixelGrid>, immobility_threshold: usize) -> Vec<Self> {
        let min_corner = self.world_grid_matrix().transform_point2(Vec2::ZERO);

        let pieces_iter = pieces.into_iter();
        let mut to_insert = Vec::with_capacity(pieces_iter.size_hint().0);
        for piece in pieces_iter {
            assert_eq!(self.body.grid().resolution, piece.resolution, "piece size did not match original grid");

            let can_stay_immobile = immobility_threshold <= piece.count_ones();

            let new_body = PixelBody::new(
                piece,
                self.body.material(),
                can_stay_immobile && self.body.inverse_mass() == 0.0,
                can_stay_immobile && self.body.inverse_inertia_tensor() == 0.0);

            let new_transform = Transform::new(
                self.world_grid_matrix().transform_vector2(new_body.local_center_of_mass()) + min_corner,
                self.transform.rotation);

            let mut new_object = PixelObject::new(new_body, self.color, new_transform);
            new_object.velocity = Velocity {
                angular: self.velocity.angular,
                linear: self.velocity.linear + self.velocity.angular * (new_transform.position - self.transform.position).rotate(Vec2::Y)
            };

            to_insert.push(new_object);
        }
        
        to_insert
    }

    /// Gets a matrix that converts from grid space to world space.
    pub fn world_grid_matrix(&self) -> Mat3 {
        self.body.world_grid_matrix(self.transform)
    }
}

/// Holds intrinsic data about a rigid object.
#[derive(Clone, Debug)]
pub struct PixelBody {
    /// The center of mass in the grid frame.
    local_center_of_mass: Vec2,
    /// The inverse of `m`.
    inverse_mass: f32,
    /// The inverse of `I`.
    inverse_inertia_tensor: f32,
    /// The value of `m`.
    mass: f32,
    /// The material governing physical properties.
    material: PixelMaterial,
    /// A 2D grid with neighbor information for each pixel. Same size as [`Self::grid`].
    neighbors: Vec<PixelNeighbors>,
    /// A list of all positions in `grid` containing corner pixels.
    corners: Vec<UVec2>,
    /// Which pixels are part of the object?
    grid: PixelGrid,
    /// A Euclidean radius that bounds the object.
    radius: f32,
}

impl PixelBody {
    /// Creates a new body for the given geometry.
    /// If `fixed` is true, then the body is presumed not to move.
    pub fn new(grid: PixelGrid, material: PixelMaterial, fixed_position: bool, fixed_rotation: bool) -> Self {
        let mut result = Self {
            corners: Vec::new(),
            grid,
            inverse_mass: 0.0,
            inverse_inertia_tensor: 0.0,
            local_center_of_mass: Vec2::ZERO,
            mass: 0.0,
            material,
            neighbors: Vec::new(),
            radius: 0.0
        };

        result.rebuild(fixed_position, fixed_rotation);

        result
    }
    
    /// Gets a list of all corner voxels in the body.
    pub fn corners(&self) -> &[UVec2] {
        &self.corners
    }

    /// The grid of pixels associated with the body.
    pub fn grid(&self) -> &PixelGrid {
        &self.grid
    }

    /// Gets the mass of the object.
    pub fn mass(&self) -> f32 {
        self.mass
    }

    /// Gets the inverse mass of the object.
    pub fn inverse_mass(&self) -> f32 {
        self.inverse_mass
    }

    /// The inverse of the moment of inertia.
    pub fn inverse_inertia_tensor(&self) -> f32 {
        self.inverse_inertia_tensor
    }

    /// Gets the material properties for this object.
    pub fn material(&self) -> PixelMaterial {
        self.material
    }

    /// Gets a matrix that converts from grid space to world space.
    pub fn world_grid_matrix(&self, transform: Transform) -> Mat3 {
        transform.to_matrix() * Mat3::from_translation(-self.local_center_of_mass())
    }

    /// Gets the center of mass in grid space.
    pub fn local_center_of_mass(&self) -> Vec2 {
        self.local_center_of_mass
    }

    /// Gets a Euclidean radius bounding the object.
    pub fn radius(&self) -> f32 {
        self.radius
    }

    /// Which adjacent pixels are set?
    pub fn neighbors(&self, position: UVec2) -> PixelNeighbors {
        self.neighbors[(position.x + position.y * self.grid.resolution().x) as usize]
    }

    /// The area of the body in pixels.
    pub fn area(&self) -> f32 {
        // Mass and area are the same in this implementation;
        // all pixels have density = 1 kg/pixel
        self.mass
    }

    /// Recalculates all acceleration structure data for this body.
    fn rebuild(&mut self, fixed_position: bool, fixed_rotation: bool) {
        self.neighbors.clear();
        self.corners.clear();

        let mut min_val = UVec2::splat(u32::MAX);
        let mut max_val = UVec2::ZERO;
        for y in 0..self.grid.resolution().y {
            for x in 0..self.grid.resolution().x {
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

        self.update_mass_and_inertia(fixed_position, fixed_rotation);

        self.radius = if min_val.cmplt(max_val).all() {
            (max_val - min_val).as_vec2().length() / 2.0
        }
        else {
            0.0
        };
    }

    /// Calculates the mass and moment of inertia for the body.
    fn update_mass_and_inertia(&mut self, fixed_position: bool, fixed_rotation: bool) {
        let summed_positions = self.grid.iter().fold(UVec3::ZERO, |acc, v| acc + uvec3(v.x, v.y, 1));
        let sz = summed_positions.z as f32;
        let mass = 1.0 * sz;
        self.local_center_of_mass = sz.recip() * summed_positions.xy().as_vec2() + Vec2::splat(0.5);

        if fixed_position {
            self.mass = f32::MAX;
            self.inverse_mass = 0.0;
        }
        else {
            self.mass = mass;
            self.inverse_mass = mass.recip();
        }
        
        let base_inertia = mass / 6.0;
        let inertial_displacement = self.grid.iter().fold(0.0, |acc, x| acc + (x.as_vec2() + Vec2::splat(0.5)  - self.local_center_of_mass).length_squared());
        let inertia_tensor = base_inertia + inertial_displacement;
            
        if !fixed_rotation {
            self.inverse_inertia_tensor = inertia_tensor.recip();
        }
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

    /// Applies the bitwise `&` operator to `self` and `other`, writing the result into `self`.
    pub fn and_with(&mut self, other: &Self) {
        assert_eq!(self.resolution(), other.resolution());
        self.values &= &other.values;
    }

    /// Gets the total number of pixels set to `true`.
    pub fn count_ones(&self) -> usize {
        self.values.count_ones()
    }

    /// Gets the total number of pixels set to `false`.
    pub fn count_zeros(&self) -> usize {
        self.values.count_zeros()
    }

    /// Sets the entire grid to `value`.
    pub fn fill(&mut self, value: bool) {
        self.values.fill(value)
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

    /// Gets an iterator over all neighborhoods of this grid.
    pub fn neighborhoods(&self) -> Vec<Self> {
        let mut seen = Self::new(self.resolution);
        let mut result = Vec::new();

        for y in 0..self.resolution.y {
            for x in 0..self.resolution.x {
                let position = uvec2(x, y);
                if !seen.get(position) && let Some(neighborhood) = self.find_neighborhood(position) {
                    seen.or_with(&neighborhood);
                    result.push(neighborhood);
                }
            }
        }

        result
    }

    /// Applies the bitwise `|` operator to `self` and `other`, writing the result into `self`.
    pub fn or_with(&mut self, other: &Self) {
        assert_eq!(self.resolution(), other.resolution());
        self.values |= &other.values;
    }

    /// Sets the value of the pixel at `position`.
    pub fn set(&mut self, position: UVec2, value: bool) -> bool {
        self.values.replace((position.x as u32 + self.resolution.x * position.y as u32) as usize, value)
    }

    /// The width and length of the grid.
    pub fn resolution(&self) -> UVec2 {
        self.resolution
    }

    /// Gets an iterator over the positions of pixels with value `true`.
    pub fn iter(&self) -> impl '_ + Iterator<Item = UVec2> {
        self.values.iter_ones().map(|index| uvec2(index as u32 % self.resolution.x, index as u32 / self.resolution.x))
    }

    /// Searches for the connected component that contains `position`.
    /// Returns [`None`] if the pixel at `position` was `false`.
    fn find_neighborhood(&self, position: UVec2) -> Option<Self> {
        thread_local! {
            // Wrap in RefCell to allow mutation (push/pop/clear)
            static TEMPORARY_STACK: RefCell<Vec<UVec2>> = RefCell::new(Vec::new());
        }

        TEMPORARY_STACK.with_borrow_mut(|temporary_stack| {
            temporary_stack.clear();
            
            if self.get(position) {
                let mut result = Self::new(self.resolution);
                temporary_stack.push(position);

                while let Some(next) = temporary_stack.pop() {
                    if next.cmplt(self.resolution()).all()
                        && self.get(next)
                        && !result.get(next) {
                        result.set(next, true);
                        temporary_stack.extend([IVec2::NEG_X, IVec2::NEG_Y, IVec2::X, IVec2::Y].map(|x| next.wrapping_add(x.as_uvec2())));
                    }
                }

                Some(result)
            }
            else {
                None
            }
        })
    }
}