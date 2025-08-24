use crate::*;
use macroquad::prelude::*;
use std::ops::*;

/// The theoretical distance an object may move before tunneling occurs.
const TUNNEL_THRESHOLD_DISTANCE: f32 = 0.4;

/// Determines how collision detection will be performed.
#[derive(Copy, Clone, Debug)]
pub enum DetectorKind {
    /// Performs collision detection once at the beginning of the frame.
    Naive,
    /// Split *detection only* into multiple substeps based upon the unaffected motion of the objects.
    Speculative {
        /// Whether to include external forces when computing objects' theoretical trajectories.
        include_external_forces: bool,
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
    pub fn update(&mut self, world: &PixelWorld, delta_time: f32) {
        self.contacts.clear();
        match self.kind {
            DetectorKind::Naive => Self::update_naive(world, &mut self.contacts),
            DetectorKind::Speculative { include_external_forces, mode }
                => Self::update_speculative(include_external_forces, mode, world, delta_time, &mut self.contacts),
        }
    }

    /// Detects all present contact points between objects in `world`. Does not account for fast-moving objects.
    fn update_naive(world: &PixelWorld, contacts: &mut Vec<Contact>) {
        for ((id_a, object_a), (id_b, object_b)) in Self::iter_object_pairs(world) {
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

    /// Detects contact points between objects. Uses substepping for fast pairs of objects to prevent interpenetration.
    fn update_speculative(include_external_forces: bool, mode: SpeculativeStepMode, world: &PixelWorld, delta_time: f32, contacts: &mut Vec<Contact>) {
        let initial_i = contacts.len();
        for ((id_a, object_a), (id_b, object_b)) in Self::iter_object_pairs(world) {
            let max_speed = Self::relative_speed_bound(object_a, object_b, delta_time);
            for (i, t_substep) in mode.substep_times(max_speed, delta_time).into_iter().enumerate() {
                let force_integration_time = if include_external_forces { t_substep } else { 0.0 };
                let a_transform_new = object_a.transform.integrate_velocity(t_substep,
                    object_a.velocity.integrate_force(force_integration_time, object_a.forces, &object_a.body, ));
                let b_transform_new = object_b.transform.integrate_velocity(t_substep,
                    object_b.velocity.integrate_force(force_integration_time, object_b.forces, &object_b.body));

                let start_i = contacts.len();
                Self::gather_contacts(DetectorObject {
                    body: &object_a.body,
                    id: id_a,
                    transform: a_transform_new
                },
                DetectorObject {
                    body: &object_b.body,
                    id: id_b,
                    transform: b_transform_new
                }, contacts);

                if i != 0 {
                    let mut clear_i = start_i;
                    while clear_i < contacts.len() {
                        let contact = &contacts[clear_i];
                        if contacts[initial_i..clear_i].iter().any(|x| x.id() == contact.id()) || contact.separation(world) < -TUNNEL_THRESHOLD_DISTANCE {
                            contacts.remove(clear_i);
                        }
                        else {
                            clear_i += 1;
                        }
                    }
                }
            }
        }
    }

    /// Computes an upper bound on the relative speed between any two points on `a` and `b`.
    fn relative_speed_bound(a: &PixelObject, b: &PixelObject, delta_time: f32) -> f32 {
        let relative_linear_speed = (a.velocity.linear - b.velocity.linear).length()
            + delta_time * (a.body.inverse_mass() * a.forces.force.length() + b.body.inverse_mass() * b.forces.force.length());
        let a_max_angular_speed = a.body.radius() * (a.velocity.angular.abs()
            + delta_time * a.body.inverse_inertia_tensor() * a.forces.torque.abs());
        let b_max_angular_speed = b.body.radius() * (b.velocity.angular.abs()
            + delta_time * b.body.inverse_inertia_tensor() * b.forces.torque.abs());
        relative_linear_speed + a_max_angular_speed + b_max_angular_speed
    }

    /// Gathers all contacts between `a` and `b`.
    fn gather_contacts(a: DetectorObject, b: DetectorObject, contacts: &mut Vec<Contact>) {
        Self::gather_corner_contacts(a, b, contacts);

        let start_i = contacts.len();
        Self::gather_corner_contacts(b, a, contacts);

        // Hack: if objects are not swapped, then updating the separation doesn't work in update_speculative
        // Make this less dumb!
        for contact in &mut contacts[start_i..] {
            *contact = contact.swap_objects();
        }
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
                    let clamped_upper = Vec2::select(upper, clamped_lower.min(Vec2::ZERO), clamped_lower);
                    let normal = clamped_upper.normalize_or_zero();

                    if normal == Vec2::ZERO {
                        continue;
                    }
                    
                    let contact_point = b_pixel_position - 0.5 * delta;
                    let penetration = ((normal.signum() - delta) / normal).min_element();

                    let world_point = a_to_world_space.transform_point2(contact_point);

                    contacts.push(Contact {
                        objects: [a.id, b.id],
                        pixel_position: [position, corner],
                        local_position: [
                            a.transform.to_matrix().inverse().transform_point2(world_point),
                            b.transform.to_matrix().inverse().transform_point2(world_point)
                        ],
                        material: PixelMaterial::mix(a.body.material(), b.body.material()),
                        penetration,
                        normal: a_to_world_space.transform_vector2(normal),
                        position: world_point
                    });
                }
            }
        }
    }

    /// Iterates over all object pairs in `world`. Each pair is only returned once,
    /// with lower [`ObjectId`]s coming first.
    fn iter_object_pairs<'a>(world: &'a PixelWorld) -> impl IntoIterator<Item = ((ObjectId, &'a PixelObject), (ObjectId, &'a PixelObject))> {
        world.iter().flat_map(|a| world.iter().map(move |b| (a, b)))
            .filter(|((id_a, _), (id_b, _))| *id_a < *id_b)
    }
}

/// How to place the speculative steps.
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub enum SpeculativeStepMode {
    /// Place the speculative steps so that they are equally far apart from each other
    /// and `t = start`/`t = end`.
    Equidistant,
    /// Place the first speculative step at `t = start`.
    Floor,
    /// Place the first and last speculative step so that `t_0 - start = end - t_n`.
    Midpoint,
    /// Place the last speculative step at `t = end`.
    Ceil
}

impl SpeculativeStepMode {
    /// For objects moving with relative `max_speed` over a total interval of `t_step`,
    /// determines the times at which substeps should occur.
    pub fn substep_times(&self, max_speed: f32, delta_time: f32) -> Vec<f32> {
        let t_max = TUNNEL_THRESHOLD_DISTANCE / max_speed;
        let required_steps = ((delta_time / t_max).ceil() as u32).max(1);

        if *self == Self::Equidistant {
            let t_substep = delta_time / (required_steps + 1) as f32;
            (1..=required_steps).map(|i| (i as f32) * t_substep).collect()
        }
        else {
            let t_end = (delta_time - t_max * (required_steps - 1) as f32).max(0.0);
            let t_offset = match self {
                Self::Floor => 0.0,
                Self::Midpoint => 0.5 * t_end,
                Self::Ceil => t_end,
                Self::Equidistant => unreachable!()
            };

            (0..required_steps).map(|i| (i as f32) * t_max + t_offset).collect()
        }
    }
}

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