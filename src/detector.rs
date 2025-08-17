use crate::*;
use macroquad::prelude::*;
use std::ops::*;

/// Determines how collision detection will be performed.
#[derive(Copy, Clone, Debug)]
pub enum DetectorKind {
    /// Performs collision detection once at the beginning of the frame.
    Naive,
    /// Split *detection only* into multiple substeps based upon the unaffected motion of the objects.
    Speculative {
        /// Whether to include the effect of external forces in the stepped objects' trajectories.
        integrate_external_forces: bool,
        /// How to step.
        mode: SpeculativeStepMode,
    }
}

/// Handles detection of collisions between objects in the world.
#[derive(Debug)]
pub struct Detector {
    /// The last set of contacts to be generated.
    contacts: Vec<Contact>,
    /// Which algorithm will be used for detection.
    kind: DetectorKind,
}

impl Detector {
    /// Creates a new detector.
    pub fn new(kind: DetectorKind) -> Self {
        Self {
            contacts: Vec::new(),
            kind
        }
    }

    /// Gets the contacts that were generated for the last call to [`Self::update`].
    pub fn contacts(&self) -> &[Contact] {
        &self.contacts
    }

    /// Generates contact points between objects in `world` to prevent interpenetration.
    pub fn update(&mut self, world: &PixelWorld) {
        self.contacts.clear();
        match self.kind {
            DetectorKind::Naive => Self::update_naive(world, &mut self.contacts),
            DetectorKind::Speculative { .. } => todo!(),
        }
    }

    /// Detects all present contact points between objects in `world`. Does not account for fast-moving objects.
    fn update_naive(world: &PixelWorld, contacts: &mut Vec<Contact>) {
        for (id_a, object_a) in world {
            for (id_b, object_b) in world {
                if id_a < id_b {
                    if let Some(object_b) = world.get(id_b) {
                        Self::gather_contacts(DetectorObject {
                            body: &object_a.body,
                            id: id_a,
                            transform: object_a.transform
                        },
                        DetectorObject {
                            body: &object_b.body,
                            id: id_b,
                            transform: object_b.transform
                        }, contacts);
                    }
                }
            }
        }
    }

    /// Gathers all contacts between `a` and `b`.
    fn gather_contacts(a: DetectorObject, b: DetectorObject, contacts: &mut Vec<Contact>) {
        // Todo: don't duplicate corner-corner collisions?
        Self::gather_corner_contacts(a, b, contacts);
        Self::gather_corner_contacts(b, a, contacts);
    }

    /// Gathers all contact constraints produced by corners of object `a` colliding with `b`.
    fn gather_corner_contacts(a: DetectorObject, b: DetectorObject, contacts: &mut Vec<Contact>) {
        let a_to_world_space = a.body.world_grid_matrix(a.transform);
        let b_to_a = a_to_world_space.inverse() * b.body.world_grid_matrix(b.transform);

        for corner in b.body.corners().iter().copied() {
            let mut closest_edge_sq = None;
            let start_i = contacts.len();

            let b_pixel_position = b_to_a.transform_point2(corner.as_vec2() + Vec2::splat(0.5));
            let min_pixel = (b_pixel_position - Vec2::splat(0.5)).floor().as_ivec2().as_uvec2();
            for offset in [UVec2::ZERO, UVec2::X, UVec2::Y, UVec2::ONE] {
                let position = min_pixel + offset;
                if a.body.grid().get_or_empty(position) {
                    let delta = b_pixel_position - (position.as_vec2() + Vec2::splat(0.5));
                    let neighbors = a.body.neighbors(position);

                    if neighbors.is_all() {
                        continue;
                    }
                    let is_edge = neighbors.kind() != PixelKind::Corner;
                    if is_edge {
                        let delta_len_sq = delta.length_squared();
                        if closest_edge_sq.is_none_or(|x| delta_len_sq < x) {
                            contacts.truncate(start_i);
                            closest_edge_sq = Some(delta_len_sq);
                        }
                        else {
                            continue;
                        }
                    }
                    else if closest_edge_sq.is_some() {
                        continue;
                    }

                    let lower = BVec2::new(neighbors.contains(PixelNeighbors::LEFT), neighbors.contains(PixelNeighbors::DOWN));
                    let upper = BVec2::new(neighbors.contains(PixelNeighbors::RIGHT), neighbors.contains(PixelNeighbors::UP));
                    let clamped_lower = Vec2::select(lower, delta.max(Vec2::ZERO), delta);
                    let normal = Vec2::select(upper, clamped_lower.min(Vec2::ZERO), clamped_lower).normalize();

                    let contact_point = b_pixel_position - 0.5 * delta;
                    let separation = -((normal.signum() - delta) / normal).min_element();

                    let world_point = a_to_world_space.transform_point2(contact_point);

                    contacts.push(Contact {
                        objects: [a.id, b.id],
                        pixel_position: [position, corner],
                        relative_position: [world_point - a.transform.position, world_point - b.transform.position],
                        friction: 0.0,
                        restitution: 0.0,
                        separation,
                        normal: a_to_world_space.transform_vector2(normal),
                        position: world_point
                    });
                }
            }
        }
    }
}

/// How to place the speculative steps.
#[derive(Copy, Clone, Debug)]
pub enum SpeculativeStepMode {
    /// Place the first speculative step at `t = start`.
    Floor,
    /// Space speculative steps so that they are equally apart.
    Equidistant,
    /// Place the last speculative step at `t = end`.
    Ceil
}

/*
impl SpeculativeStepMode {
    /// For objects moving with relative `max_speed` over a total interval of `t_step`,
    /// determines the times at which substeps should occur.
    pub fn substep_times(&self, max_speed: f32, t_step: f32) -> Vec<f32> {
        /// The theoretical distance an object may move before tunneling occurs.
        const TUNNEL_THRESHOLD_DISTANCE: f32 = 0.5;

        let t_max = TUNNEL_THRESHOLD_DISTANCE / max_speed;
        let required_steps = ((t_step / t_max).ceil() as u32).max(1);
        let end_distance = (t_step - required_steps as f32 * t_max).max(0.0);

        match self {
            SpeculativeStepMode::Floor => (0..required_steps).map(|i| (i as f32) * t_max),
            SpeculativeStepMode::Equidistant => todo!(),
            SpeculativeStepMode::Ceil => (0..required_steps).map(|i| (i as f32) * t_max + end_distance),
        }
    }
} */

/// An object being processed by the [`Detector`].
#[derive(Copy, Clone, Debug)]
struct DetectorObject<'a> {
    /// The collider data.
    pub body: &'a PixelBody,
    /// The object ID.
    pub id: ObjectId,
    /// The position of the object's center of mass.
    pub transform: Transform,
}