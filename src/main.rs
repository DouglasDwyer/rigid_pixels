#![feature(let_chains)]

use bitvec::prelude::*;
use geese::*;
use macroquad::prelude::*;
use slab::*;
use std::mem::*;
use std::time::*;

pub struct CameraControl {
    ctx: GeeseContextHandle<Self>
}

impl CameraControl {
    fn update_camera_position(&mut self, _: &on::Frame) {
        const PIXELS_PER_WORLD_UNIT: f32 = 50.0;
        const CAMERA_MOVEMENT_COEFFICIENT: f32 = 5.0;

        let mut world = self.ctx.get_mut::<GameWorld>();
        let mut movement = Vec2::ZERO;

        for (key, delta) in [(KeyCode::Left, -Vec2::X), (KeyCode::Right, Vec2::X), (KeyCode::Down, -Vec2::Y), (KeyCode::Up, Vec2::Y)] {
            if is_key_down(key) {
                movement += delta;
            }
        }

        world.camera_position += CAMERA_MOVEMENT_COEFFICIENT * get_frame_time() * movement.normalize_or_zero();
        
        let projection = Mat4::from_translation(0.5 * vec3(screen_width(), screen_height(), 0.0)) * Mat4::from_scale(vec3(PIXELS_PER_WORLD_UNIT, -PIXELS_PER_WORLD_UNIT, 1.0));
        let view = Mat4::from_translation(-vec3(world.camera_position.x, world.camera_position.y, 0.0));
        world.camera_view_projection = projection * view;
        world.camera_inverse_view_projection = world.camera_view_projection.inverse();
    }
}

impl GeeseSystem for CameraControl {
    const DEPENDENCIES: Dependencies = dependencies()
        .with::<Mut<GameWorld>>();

    const EVENT_HANDLERS: EventHandlers<Self> = event_handlers()
        .with(Self::update_camera_position);

    fn new(ctx: GeeseContextHandle<Self>) -> Self {
        Self {
            ctx
        }
    }
}

pub struct CameraInteract {
    ctx: GeeseContextHandle<Self>,
    interacting: bool,
    selected_object: usize,
    object_space_point: Vec2
}

impl CameraInteract {
    fn select_new_object(&mut self) {
        let world_click_position = self.world_click_position();
        let mut world = self.ctx.get_mut::<GameWorld>();
        for (id, object) in &mut world.objects {
            let clicked_position = object.model_transform().inverse().transform_point3a(vec3a(world_click_position.x, world_click_position.y, 0.0)).xy();
            if object.physics_as.get_or_empty(clicked_position.floor().as_ivec2().as_uvec2()) {
                object.linear_velocity = Vec2::ZERO;
                object.angular_velocity = 0.0;
                self.selected_object = id;
                self.interacting = true;
                self.object_space_point = clicked_position;
                return;
            }
        }
    }

    fn apply_force_to_selected(&mut self, _: &on::PhysicsUpdate) {
        let world_click_position = self.world_click_position();
        let mut world = self.ctx.get_mut::<GameWorld>();
        if self.interacting && let Some(object) = world.objects.get_mut(self.selected_object) {
            let constants = SpringConstants {
                k: 100.0,
                rest_length: 0.0,
                drag: 10.0,
                origin: world_click_position
            };

            let world_space_point = object.model_transform().transform_point3a(vec3a(self.object_space_point.x, self.object_space_point.y, 0.0)).xy();

            let point_velocity = Vec2::Y.rotate(world_space_point - object.position) * object.angular_velocity + object.linear_velocity;

            let force = Spring::spring_force(&constants, world_space_point, point_velocity);
            object.add_force(world_space_point, force);
        }
        else {
            self.interacting = false;
        }
    }

    fn update_selected_object(&mut self, _: &on::Frame) {
        self.interacting &= is_mouse_button_down(MouseButton::Left);
        if !self.interacting && is_mouse_button_pressed(MouseButton::Left) {
            self.select_new_object();
        }
    }

    fn world_click_position(&self) -> Vec2 {
        let world = self.ctx.get::<GameWorld>();
        world.camera_inverse_view_projection.transform_point3a(vec3a(mouse_position().0, mouse_position().1, 0.0)).xy()
    }
}

impl GeeseSystem for CameraInteract {
    const DEPENDENCIES: Dependencies = dependencies()
        .with::<Mut<GameWorld>>();

    const EVENT_HANDLERS: EventHandlers<Self> = event_handlers()
        .with(Self::apply_force_to_selected)
        .with(Self::update_selected_object);

    fn new(ctx: GeeseContextHandle<Self>) -> Self {
        Self {
            ctx,
            interacting: false,
            selected_object: 0,
            object_space_point: Vec2::ZERO
        }
    }
}

pub struct CollisionDetector {
    contacts: Vec<CollisionContact>,
    ctx: GeeseContextHandle<Self>,
}

impl CollisionDetector {
    pub fn contacts(&self) -> &[CollisionContact] {
        &self.contacts
    }

    fn generate_contacts(&mut self) {
        let mut contacts = take(&mut self.contacts);
        contacts.clear();

        let world = self.ctx.get::<GameWorld>();
        for a in 0..world.objects.capacity() {
            if let Some(object_a) = world.objects.get(a) {
                for b in (a + 1)..world.objects.capacity() {
                    if let Some(object_b) = world.objects.get(b) {
                        let a_descriptor = CollisionDetectionObject {
                            id: a,
                            object: &world.objects[a]
                        };

                        let b_descriptor = CollisionDetectionObject {
                            id: b,
                            object: &world.objects[b]
                        };

                        self.generate_corner_contacts(a_descriptor, b_descriptor, &mut contacts);
                        self.generate_corner_contacts(b_descriptor, a_descriptor, &mut contacts);
                    }
                }
            }
        }

        self.contacts = contacts;
    }

    fn generate_corner_contacts(&self, a: CollisionDetectionObject, b: CollisionDetectionObject, contacts: &mut Vec<CollisionContact>) {
        let a_to_world_space = a.object.model_transform();
        let b_to_a = a_to_world_space.inverse() * b.object.model_transform();

        for corner in &b.object.physics_as.corners {
            let b_pixel_position = b_to_a.transform_point3(vec3(corner.x as f32 + 0.5, corner.y as f32 + 0.5, 0.0)).xy();
            let min_pixel = (b_pixel_position - Vec2::splat(0.5)).floor().as_ivec2().as_uvec2();
            for offset in [UVec2::ZERO, UVec2::X, UVec2::Y, UVec2::ONE] {
                let position = min_pixel + offset;
                if a.object.physics_as.get_or_empty(position) {
                    let delta = b_pixel_position - (position.as_vec2() + Vec2::splat(0.5));
                    let normal = match a.object.physics_as.classification(position) {
                        PixelKind::Empty => unreachable!(),
                        PixelKind::Corner => delta.normalize_or_zero(),
                        PixelKind::EdgeX => vec2(delta.x.signum(), 0.0),
                        PixelKind::EdgeY => vec2(0.0, delta.y.signum()),
                        PixelKind::Interior => continue,
                    };

                    //assert!(delta.try_normalize().is_some(), "Col between {corner:?} {position:?} (which lies at {b_pixel_position:?} for {b_to_a:?} (with {a_to_world_space:?})");

                    let contact_point = b_pixel_position - 0.5 * delta;
                    let penetration = ((normal.signum() - delta) / normal).min_element();

                    let world_point = a_to_world_space.transform_point3a(vec3a(contact_point.x, contact_point.y, 0.0)).xy();

                    contacts.push(CollisionContact {
                        objects: [a.id, b.id],
                        relative_position: [world_point - a.object.position, world_point - b.object.position],
                        friction: 0.9,
                        restitution: 0.01,
                        penetration,
                        normal: -a_to_world_space.transform_vector3a(vec3a(normal.x, normal.y, 0.0)).xy(),
                        world_pos: world_point
                    });
                }
            }
        }
    }

    fn detect_collisions(&mut self, _: &on::PhysicsUpdate) {
        self.generate_contacts();
    }
}

impl GeeseSystem for CollisionDetector {
    const DEPENDENCIES: Dependencies = dependencies()
        .with::<GameWorld>()
        .with::<ObjectIntegrator>();

    const EVENT_HANDLERS: EventHandlers<Self> = event_handlers()
        .with(Self::detect_collisions);

    fn new(ctx: GeeseContextHandle<Self>) -> Self {
        Self {
            contacts: Vec::new(),
            ctx
        }
    }
}

pub struct ContactResolver {
    ctx: GeeseContextHandle<Self>
}

impl ContactResolver {
    fn adjust_positions(&mut self, delta_time: f32) {
        let mut contacts = self.ctx.get::<CollisionDetector>().contacts().to_vec();
        let mut world = self.ctx.get_mut::<GameWorld>();
        let initial_poses = world.objects.iter().map(|x| x.1.position).collect::<Vec<_>>();

        let mut iterations = 0;
        while iterations < 2 * contacts.len() {
            let mut max_penetration = 0.001;
            let mut max_index = contacts.len();
            for (i, contact) in contacts.iter().enumerate() {
                if max_penetration < contact.penetration {
                    max_penetration = contact.penetration;
                    max_index = i;
                }
            }

            if max_index == contacts.len() {
                return;
            }

            let contact = &contacts[max_index];
            let (object_a, object_b) = world.objects.get2_mut(contact.objects[0], contact.objects[1]).unwrap();
            let movement_deltas = contact.apply_position(object_a, object_b);
            let affected_objects = contact.objects;

            for to_update in &mut contacts {
                for (body_index, body_id) in affected_objects.into_iter().enumerate() {
                    for (index, id) in to_update.objects.into_iter().enumerate() {
                        if body_id == id {
                            let obj = &world.objects[body_id];
                            let delta_position = movement_deltas[body_index].linear_delta + Vec2::Y.rotate(to_update.relative_position[index]) * movement_deltas[body_index].angular_delta;
                            to_update.penetration += [-1.0, 1.0][index] * delta_position.dot(to_update.normal);
                        }
                    }
                }
            }

            iterations += 1;
        }
    }

    fn adjust_velocities(&mut self, delta_time: f32) {
        let contacts = self.ctx.get::<CollisionDetector>().contacts().to_vec();
        let mut world = self.ctx.get_mut::<GameWorld>();
        let mut desired_velocities = Vec::from_iter(contacts.iter().map(|x| x.calculate_desired_velocity(delta_time, &world.objects[x.objects[0]], &world.objects[x.objects[1]])));

        let mut iterations = 0;
        while iterations < 2 * contacts.len() {
            let mut index_max = desired_velocities.len();
            let mut velocity_max = 0.001;
            for (index, velocity) in desired_velocities.iter().copied().enumerate() {
                if velocity_max < velocity {
                    velocity_max = velocity;
                    index_max = index;
                }
            }

            if index_max == desired_velocities.len() {
                return;
            }

            let max_contact = &contacts[index_max];
            let (object_a, object_b) = world.objects.get2_mut(max_contact.objects[0], max_contact.objects[1]).unwrap();
            let deltas = max_contact.apply_velocity_change(object_a, object_b, desired_velocities[index_max]);

            for (contact_index, contact) in contacts.iter().enumerate() {
                for (index, obj) in contact.objects.into_iter().enumerate() {
                    let matching_index = if obj == max_contact.objects[0] {
                        0
                    }
                    else if obj == max_contact.objects[1] {
                        1
                    }
                    else {
                        continue;
                    };

                    desired_velocities[contact_index] = contact.calculate_desired_velocity(delta_time, &world.objects[contact.objects[0]], &world.objects[contact.objects[1]]);
                }
            }
            iterations += 1;
        }
    }

    fn resolve_contacts(&mut self, event: &on::PhysicsUpdate) {
        self.adjust_positions(event.delta_time);
        self.adjust_velocities(event.delta_time);
    }
}

impl GeeseSystem for ContactResolver {
    const DEPENDENCIES: Dependencies = dependencies()
        .with::<CollisionDetector>()
        .with::<Mut<GameWorld>>();

    const EVENT_HANDLERS: EventHandlers<Self> = event_handlers()
        .with(Self::resolve_contacts);

    fn new(ctx: GeeseContextHandle<Self>) -> Self {
        Self {
            ctx
        }
    }
}

pub struct ObjectIntegrator {
    ctx: GeeseContextHandle<Self>
}

impl ObjectIntegrator {
    fn move_objects(&mut self, event: &on::PhysicsUpdate) {
        let mut world = self.ctx.get_mut::<GameWorld>();
        for (_, object) in &mut world.objects {
            object.integrate(event.delta_time);
        }
    }
}

impl GeeseSystem for ObjectIntegrator {
    const DEPENDENCIES: Dependencies = dependencies()
        .with::<Mut<GameWorld>>();

    const EVENT_HANDLERS: EventHandlers<Self> = event_handlers()
        .with(Self::move_objects);

    fn new(ctx: GeeseContextHandle<Self>) -> Self {
        Self {
            ctx
        }
    }
}

pub struct Gravity {
    ctx: GeeseContextHandle<Self>
}

impl Gravity {
    fn apply_gravity(&mut self, _: &on::PhysicsUpdate) {
        let mut world = self.ctx.get_mut::<GameWorld>();

        for (_, object) in &mut world.objects {
            object.add_force_at_center(vec2(0.0, -9.81 * object.physics_as.mass));
        }
    }
}

impl GeeseSystem for Gravity {
    const DEPENDENCIES: Dependencies = dependencies()
        .with::<Mut<GameWorld>>();

    const EVENT_HANDLERS: EventHandlers<Self> = event_handlers()
        .with(Self::apply_gravity);

    fn new(ctx: GeeseContextHandle<Self>) -> Self {
        Self {
            ctx
        }
    }
}

pub struct SpringConstants {
    pub k: f32,
    pub rest_length: f32,
    pub drag: f32,
    pub origin: Vec2
}

pub struct Spring {
    ctx: GeeseContextHandle<Self>
}

impl Spring {
    pub fn spring_force(constants: &SpringConstants, position: Vec2, velocity: Vec2) -> Vec2 {
        let displacement = position - constants.origin;
        let magnitude = constants.k * (constants.rest_length - displacement.length());
        let acceleration_term = magnitude * displacement.try_normalize().unwrap_or(Vec2::X);
        let drag_term = constants.drag * velocity;
        acceleration_term - drag_term
    }

    fn apply_force(&mut self, _: &on::PhysicsUpdate) {
        const CONSTANTS: SpringConstants = SpringConstants {
            k: 4.0,
            rest_length: 1.0,
            drag: 0.1,
            origin: Vec2::ZERO
        };

        let mut world = self.ctx.get_mut::<GameWorld>();

        for (_, object) in &mut world.objects {
            object.add_force_at_center(Self::spring_force(&CONSTANTS, object.position, object.linear_velocity));
        }
    }
}

impl GeeseSystem for Spring {
    const DEPENDENCIES: Dependencies = dependencies()
        .with::<Mut<GameWorld>>();

    const EVENT_HANDLERS: EventHandlers<Self> = event_handlers()
        .with(Self::apply_force);

    fn new(ctx: GeeseContextHandle<Self>) -> Self {
        Self {
            ctx
        }
    }
}

pub struct Renderer {
    ctx: GeeseContextHandle<Self>
}

impl Renderer {
    fn draw_object(&self, projection_view: Mat4, object: &PixelObject) {
        const OUTLINE_SIZE: f32 = 0.1;

        let projection_view_model = projection_view * object.model_transform();
        
        let horizontal_offset = projection_view_model.transform_vector3(vec3(1.0, 0.0, 0.0));
        let vertical_offset = projection_view_model.transform_vector3(vec3(0.0, 1.0, 0.0));
        let full_offset = 0.5 * OUTLINE_SIZE * (horizontal_offset + vertical_offset);

        for pixel in object.grid.iter() {
            let offset = projection_view_model.transform_point3(vec3(pixel.x as f32, pixel.y as f32, 0.0));
            draw_affine_parallelogram(offset, horizontal_offset, vertical_offset, None, Color::new(0.0, 0.5, 0.5, 1.0));
            draw_affine_parallelogram(offset + full_offset, (1.0 - OUTLINE_SIZE) * horizontal_offset, (1.0 - OUTLINE_SIZE) * vertical_offset, None, Color::new(0.2, 0.2, 0.2, 1.0));
        }
    }

    fn draw_objects(&self, projection_view: Mat4) {
        let world = self.ctx.get::<GameWorld>();
        for (_, object) in &world.objects {
            self.draw_object(projection_view, object);
        }
    }

    fn draw_origin(&self, projection_view: Mat4) {
        let origin = projection_view.transform_point3a(Vec3A::ZERO);
        draw_circle(origin.x, origin.y, 5.0, Color::new(1.0, 1.0, 0.0, 1.0));
    }

    fn draw_frame(&mut self, _: &on::Frame) {
        clear_background(Color::new(0.8, 0.2, 0.6, 1.0));
        let projection_view = self.ctx.get::<GameWorld>().camera_view_projection;
        self.draw_objects(projection_view);
        self.draw_origin(projection_view);

    }
}

impl GeeseSystem for Renderer {
    const DEPENDENCIES: Dependencies = dependencies()
        .with::<GameWorld>();

    const EVENT_HANDLERS: EventHandlers<Self> = event_handlers()
        .with(Self::draw_frame);

    fn new(ctx: GeeseContextHandle<Self>) -> Self {
        Self {
            ctx
        }
    }
}

pub struct GameWorld {
    camera_position: Vec2,
    camera_inverse_view_projection: Mat4,
    camera_view_projection: Mat4,
    objects: Slab<PixelObject>
}

impl GeeseSystem for GameWorld {
    fn new(_: GeeseContextHandle<Self>) -> Self {
        Self {
            camera_position: Vec2::ZERO,
            objects: Slab::new(),
            camera_inverse_view_projection: Mat4::IDENTITY,
            camera_view_projection: Mat4::IDENTITY
        }
    }
}

#[derive(Copy, Clone, Debug, Default)]
pub struct MovementDeltas {
    pub linear_delta: Vec2,
    pub angular_delta: f32
}

#[derive(Copy, Clone, Debug)]
pub struct CollisionContact {
    pub objects: [usize; 2],
    pub relative_position: [Vec2; 2],
    pub friction: f32,
    pub restitution: f32,
    pub penetration: f32,
    pub normal: Vec2,
    pub world_pos: Vec2
}

impl CollisionContact {
    pub fn apply_position(&self, object_a: &mut PixelObject, object_b: &mut PixelObject) -> [MovementDeltas; 2] {
        let mut total_inertia = Mat2::ZERO;
        let mut linear_inertia = [0.0; 2];
        let mut angular_inertia = [0.0; 2];
        self.calculate_velocity_per_impulse(object_a, object_b, &mut total_inertia, &mut linear_inertia, &mut angular_inertia, &mut Vec2::ZERO);
        let scalar_inertia = (total_inertia * self.normal).dot(self.normal);

        let mut result = [MovementDeltas::default(); 2];
        for (index, obj) in [object_a, object_b].into_iter().enumerate() {
            if 0.0 < linear_inertia[index] {
                let sign = [1.0, -1.0][index];
    
                let linear = sign * self.penetration * (linear_inertia[index] / scalar_inertia);
                let angular = sign * self.penetration * (angular_inertia[index] / scalar_inertia);
    
                result[index].linear_delta = linear * self.normal;
                obj.position += result[index].linear_delta;
    
                let target_angular_direction = self.relative_position[index].perp_dot(self.normal);
                result[index].angular_delta = obj.physics_as.inverse_inertia_tensor * target_angular_direction * angular / angular_inertia[index].max(0.00001);
                obj.rotation += result[index].angular_delta;
            }
        }

        result
    }

    pub fn apply_velocity_change(&self, object_a: &mut PixelObject, object_b: &mut PixelObject, desired_velocity: f32) -> [MovementDeltas; 2] {
        let mut velocity_change = Mat2::ZERO;
        let mut contact_velocity = Vec2::ZERO;
        self.calculate_velocity_per_impulse(object_a, object_b, &mut velocity_change, &mut [0.0; 2], &mut [0.0; 2], &mut contact_velocity);

        let velocity_to_kill = desired_velocity * self.normal - contact_velocity.reject_from_normalized(self.normal);
        let impulse_per_velocity = velocity_change.inverse();
        let mut impulse = impulse_per_velocity * velocity_to_kill;

        let contact_impulse = impulse.dot(self.normal);
        let planar_impulse = impulse.reject_from_normalized(self.normal);
        let planar_impulse_length = planar_impulse.length();
        if planar_impulse_length > contact_impulse * self.friction {
            let scaled_planar_impulse = self.friction * planar_impulse / planar_impulse_length;
            let new_contact_velocity = (velocity_change * (self.normal + scaled_planar_impulse)).dot(self.normal);
            let impulse_len = desired_velocity / new_contact_velocity;
            impulse = impulse_len * (self.normal + scaled_planar_impulse);
        }

        let mut results = [MovementDeltas::default(); 2];

        for (index, obj) in [object_a, object_b].into_iter().enumerate() {
            let impulsive_torque = self.relative_position[index].perp_dot(impulse);

            let deltas = &mut results[index];
            deltas.linear_delta = [1.0, -1.0][index] * obj.physics_as.inverse_mass * impulse;
            deltas.angular_delta = [1.0, -1.0][index] * obj.physics_as.inverse_inertia_tensor * impulsive_torque;

            obj.linear_velocity += deltas.linear_delta;

            if !obj.linear_velocity.is_finite() {
                panic!("STINKY {deltas:?} {velocity_change:?} {impulse:?} {desired_velocity}");
            }
            obj.angular_velocity += deltas.angular_delta;
            if !obj.angular_velocity.is_finite() {
                panic!("STINKY2Y {deltas:?} {velocity_change:?} {impulse:?} {desired_velocity}");
            }
        }

        results
    }

    pub fn calculate_desired_velocity(&self, delta_time: f32, object_a: &PixelObject, object_b: &PixelObject) -> f32 {
        let mut total_current_velocity = Vec2::ZERO;
        let mut velocity_from_acceleration = 0.0;

        for (index, obj) in [object_a, object_b].into_iter().enumerate() {
            let relative_contact = self.relative_position[index];
            total_current_velocity += [1.0, -1.0][index] * (Vec2::Y.rotate(relative_contact) * obj.angular_velocity + obj.linear_velocity);
            velocity_from_acceleration += [1.0, -1.0][index] * delta_time * obj.last_frame_acceleration.dot(self.normal);
        }

        let contact_velocity = total_current_velocity.dot(self.normal);
        let mut restitution = (0.25 < contact_velocity).then_some(self.restitution).unwrap_or_default();

        // todo: make things not jumpy.
        let res = -contact_velocity - self.restitution * (contact_velocity - velocity_from_acceleration);

        res
    }

    fn calculate_velocity_per_impulse(&self, object_a: &PixelObject, object_b: &PixelObject, total: &mut Mat2, linear: &mut [f32; 2], angular: &mut [f32; 2], total_current_velocity: &mut Vec2) {
        for (index, obj) in [object_a, object_b].into_iter().enumerate() {
            let relative_contact = self.relative_position[index];
            
            *total_current_velocity += [1.0, -1.0][index] * (Vec2::Y.rotate(relative_contact) * obj.angular_velocity + obj.linear_velocity);
            let relative_direction = relative_contact.normalize_or_zero();
            let rejection_matrix = Mat2::IDENTITY - mat2(relative_direction, relative_direction) * Mat2::from_diagonal(relative_direction);
            let angular_velocity_per_impulse = Mat2::from_diagonal(Vec2::splat(obj.physics_as.inverse_inertia_tensor * relative_contact.length())) * rejection_matrix;

            linear[index] = obj.physics_as.inverse_mass;
            angular[index] = (angular_velocity_per_impulse * self.normal).dot(self.normal);
            *total += Mat2::from_diagonal(Vec2::splat(linear[index])) + angular_velocity_per_impulse;
        }
    }
}

pub struct CollisionPoint {
    pub contact: CollisionContact,

}

#[derive(Copy, Clone)]
struct CollisionDetectionObject<'a> {
    pub id: usize,
    pub object: &'a PixelObject
}

#[derive(Copy, Clone, Default)]
pub struct ForceAccumulator {
    pub force: Vec2,
    pub torque: f32
}

pub struct PixelObject {
    pub forces: ForceAccumulator,
    pub position: Vec2,
    pub rotation: f32,
    pub last_frame_acceleration: Vec2,
    pub linear_velocity: Vec2,
    pub angular_velocity: f32,
    pub grid: PixelGrid,
    pub physics_as: PixelPhysicsAs,
    pub last_pos: Vec2
}

impl PixelObject {
    pub fn model_transform(&self) -> Mat4 {
        let offset = self.position - self.physics_as.local_center_of_mass;
        Mat4::from_rotation_translation(Quat::from_rotation_z(self.rotation), vec3(self.position.x, self.position.y, 0.0))
            * Mat4::from_translation(vec3(-self.physics_as.local_center_of_mass.x, -self.physics_as.local_center_of_mass.y, 0.0))
    }
    
    pub fn add_force(&mut self, point: Vec2, force: Vec2) {
        self.forces.force += force;
        self.forces.torque += (point - self.position).perp_dot(force);
    }

    pub fn add_force_at_center(&mut self, force: Vec2) {
        self.forces.force += force;
    }

    pub fn integrate(&mut self, delta_time: f32) {
        self.last_frame_acceleration = self.physics_as.inverse_mass * self.forces.force;
        let last_frame_angular_acceleration = self.physics_as.inverse_inertia_tensor * self.forces.torque;

        self.linear_velocity += delta_time * self.last_frame_acceleration;
        self.angular_velocity += delta_time * last_frame_angular_acceleration;

        self.linear_velocity *= 0.9999f32.powf(delta_time);
        self.angular_velocity *= 0.9999f32.powf(delta_time);

        self.last_pos = self.position;

        self.position += delta_time * self.linear_velocity;
        self.rotation += delta_time * self.angular_velocity;

        self.forces = ForceAccumulator::default();
    }
}

#[derive(Clone, Debug, PartialEq, Default)]
pub struct PixelGrid {
    size: UVec2,
    values: BitVec
}

impl PixelGrid {
    pub fn new(size: UVec2) -> Self {
        let mut values = BitVec::new();
        values.resize((size.x * size.y) as usize, false);

        Self {
            size,
            values
        }
    }

    pub fn get(&self, position: UVec2) -> bool {
        self.values[(position.x as u32 + self.size.x * position.y as u32) as usize]
    }

    pub fn set(&mut self, position: UVec2, value: bool) -> bool {
        self.values.replace((position.x as u32 + self.size.x * position.y as u32) as usize, value)
    }

    pub fn size(&self) -> UVec2 {
        self.size
    }

    pub fn iter(&self) -> impl '_ + Iterator<Item = UVec2> {
        self.values.iter_ones().map(|index| uvec2(index as u32 % self.size.x, index as u32 / self.size.x))
    }
}

pub struct PhysicsCacheData {
    pub position: Vec2,
    pub rotation: f32,
}

#[derive(Debug, Default)]
pub struct PixelPhysicsAs {
    local_center_of_mass: Vec2,
    mass: f32,
    inverse_mass: f32,
    inverse_inertia_tensor: f32,
    classifications: Vec<PixelKind>,
    corners: Vec<UVec2>,
    grid: PixelGrid,
}

impl PixelPhysicsAs {
    pub fn new(grid: &PixelGrid) -> Self {
        let mut result = Self {
            local_center_of_mass: Vec2::ZERO,
            classifications: Vec::new(),
            corners: Vec::new(),
            grid: PixelGrid::default(),
            mass: 0.0,
            inverse_mass: 0.0,
            inverse_inertia_tensor: 0.0
        };

        result.update(grid);
        result
    }

    pub fn classification(&self, position: UVec2) -> PixelKind {
        self.classifications[(position.x + position.y * self.grid.size().x) as usize]
    }

    fn update(&mut self, grid: &PixelGrid) {
        if grid != &self.grid {
            self.grid = grid.clone();
            self.rebuild();
        }
    }

    fn classify(&self, position: UVec2) -> PixelKind {
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

        if self.get_or_empty(position) {
            let mut result = 0;
            for (index, offset) in [-IVec2::X, IVec2::X, -IVec2::Y, IVec2::Y].into_iter().enumerate() {
                if self.get_or_empty(position + offset.as_uvec2()) {
                    result |= 1 << index;
                }
            }
            CLASSIFICATIONS[result]
        }
        else {
            PixelKind::Empty
        }
    }

    fn rebuild(&mut self) {
        self.classifications.clear();
        self.corners.clear();

        for y in 0..self.grid.size().y {
            for x in 0..self.grid.size().x {
                let classification = self.classify(uvec2(x, y));
                if classification == PixelKind::Corner {
                    self.corners.push(uvec2(x, y));
                }
                self.classifications.push(classification);
            }
        }

        self.update_mass_and_inertia();
    }

    fn get_or_empty(&self, position: UVec2) -> bool {
        if position.cmplt(self.grid.size()).all() {
            self.grid.get(position)
        }
        else {
            false
        }
    }

    fn update_mass_and_inertia(&mut self) {
        let summed_positions = self.grid.iter().fold(UVec3::ZERO, |acc, v| acc + uvec3(v.x, v.y, 1));
        let sz = summed_positions.z as f32;
        self.mass = 1.0 * sz;
        self.inverse_mass = self.mass.recip();
        self.local_center_of_mass = sz.recip() * summed_positions.xy().as_vec2() + Vec2::splat(0.5);
        
        let base_inertia = self.mass / 6.0;
        let inertial_displacement = self.grid.iter().fold(0.0, |acc, x| acc + (x.as_vec2() + Vec2::splat(0.5)  - self.local_center_of_mass).length_squared());
        let inertia_tensor = base_inertia + inertial_displacement;
        
        self.inverse_inertia_tensor = inertia_tensor.recip();
    }
}

#[derive(Copy, Clone, Debug, PartialEq, Eq)]
#[repr(u8)]
enum PixelKind {
    Empty,
    Corner,
    EdgeX,
    EdgeY,
    Interior
}

pub struct ForceGenerators;

impl GeeseSystem for ForceGenerators {
    const DEPENDENCIES: Dependencies = dependencies()
        .with::<Gravity>();
        //.with::<Spring>();

    fn new(_: GeeseContextHandle<Self>) -> Self {
        Self
    }
}

pub struct RigidPixels;

impl GeeseSystem for RigidPixels {
    const DEPENDENCIES: Dependencies = dependencies()
        .with::<CameraControl>()
        .with::<CameraInteract>()
        .with::<CollisionDetector>()
        .with::<ContactResolver>()
        .with::<ForceGenerators>()
        .with::<GameWorld>()
        .with::<ObjectIntegrator>()
        .with::<Renderer>();

    fn new(_: GeeseContextHandle<Self>) -> Self {
        Self
    }
}

mod on {
    pub struct Frame;

    pub struct PhysicsUpdate {
        pub delta_time: f32
    }
}

fn create_floor() -> PixelObject {
    const FLOOR_LENGTH: u32 = 512;

    let mut grid = PixelGrid::new(uvec2(FLOOR_LENGTH, 2));
    for i in 0..FLOOR_LENGTH {
        grid.set(uvec2(i, 0), true);
    }
    grid.set(uvec2(256, 1), true);

    let mut physics_as = PixelPhysicsAs::new(&grid);
    physics_as.inverse_inertia_tensor = 0.0;
    physics_as.inverse_mass = 0.0;

    PixelObject {
        position: vec2(0.0, -5.0),
        rotation: 0.0,
        linear_velocity: Vec2::ZERO,
        angular_velocity: 0.0,
        physics_as,
        forces: ForceAccumulator::default(),
        grid,
        last_frame_acceleration: Vec2::ZERO,
        last_pos: vec2(0.0, -5.0)
    }
}

fn create_floor2() -> PixelObject {
    const FLOOR_LENGTH: u32 = 512;

    let mut grid = PixelGrid::new(uvec2(FLOOR_LENGTH, 5));
    for x in 0..FLOOR_LENGTH {
        let sinx = 5.0 * (0.1 * (x as f32)).sin();
        for y in 0..5 {
            if (y as f32) < sinx {
                grid.set(uvec2(x, y), true);
            }
        }
        
        grid.set(uvec2(x, 0), true);
    }

    let mut physics_as = PixelPhysicsAs::new(&grid);
    physics_as.inverse_inertia_tensor = 0.0;
    physics_as.inverse_mass = 0.0;

    PixelObject {
        position: vec2(0.0, -5.0),
        rotation: 0.0,
        linear_velocity: Vec2::ZERO,
        angular_velocity: 0.0,
        physics_as,
        forces: ForceAccumulator::default(),
        grid,
        last_frame_acceleration: Vec2::ZERO,
        last_pos: vec2(0.0, -5.0)
    }
}

fn create_rectangle_body(position: Vec2, rotation: f32, extents: UVec2) -> PixelObject {
    let mut grid = PixelGrid::new(extents);
    for y in 0..extents.y {
        for x in 0..extents.x {
            grid.set(uvec2(x, y), true);
        }
    }

    let physics_as = PixelPhysicsAs::new(&grid);

    PixelObject {
        position,
        rotation,
        linear_velocity: vec2(0.0, 0.0),
        angular_velocity: 0.0,
        physics_as,
        forces: ForceAccumulator::default(),
        grid,
        last_frame_acceleration: Vec2::ZERO,
        last_pos: position
    }
}

fn create_circle_body(position: Vec2, rotation: f32, radius: f32) -> PixelObject {
    let extents = UVec2::splat((2.0 * radius + 1.0) as u32);
    let mut grid = PixelGrid::new(extents);
    for y in 0..extents.y {
        for x in 0..extents.x {
            if (x as f32 - radius).powi(2) + (y as f32 - radius).powi(2) <= radius.powi(2) {
                grid.set(uvec2(x, y), true);
            }
        }
    }

    let physics_as = PixelPhysicsAs::new(&grid);

    PixelObject {
        position,
        rotation,
        linear_velocity: vec2(0.0, 0.0),
        angular_velocity: 0.0,
        physics_as,
        forces: ForceAccumulator::default(),
        grid,
        last_frame_acceleration: Vec2::ZERO,
        last_pos: position
    }
}

fn setup_scene(ctx: &mut GameWorld) {
    //ctx.objects.insert(create_rectangle_body(vec2(0.0, -2.5), 0.0, uvec2(1, 3)));
    ctx.objects.insert(create_floor2());
    ctx.objects.insert(create_rectangle_body(vec2(5.0, 0.5), 0.0, uvec2(2, 3)));
    ctx.objects.insert(create_circle_body(vec2(-2.0, 0.0), 0.2, 2.0));
    //ctx.objects.insert(create_rectangle_body(vec2(-5.0, 2.0), 0.0, uvec2(3, 3)));
}

#[macroquad::main("Rigid pixels")]
async fn main() {
    let mut ctx = GeeseContext::default();
    ctx.flush()
        .with(geese::notify::add_system::<RigidPixels>());
    setup_scene(&mut ctx.get_mut::<GameWorld>());

    let mut next_physics_time = Instant::now();
    let physics_delta = Duration::from_millis(25);

    loop {
        ctx.flush()
            .with(on::Frame);

        while next_physics_time < Instant::now() {
            ctx.flush()
                .with(on::PhysicsUpdate { delta_time: physics_delta.as_secs_f32() });
            next_physics_time += physics_delta;
        }

        next_frame().await
    }
}