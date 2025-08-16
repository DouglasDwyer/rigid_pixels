use crate::*;

/// Determines which part of the world will be rendered.
#[derive(Copy, Clone, Debug)]
pub struct Camera {
    /// The position in the world that should be centered onscreen.
    pub position: Vec2,
    /// A scale factor that converts from world units to dots.
    pub zoom: f32
}

impl Camera {
    /// Gets a matrix that converts from world space to screen space (in pixels).
    pub fn screen_world_matrix(&self, resolution: Vec2, pixel_scale: f32) -> Mat3 {
        Mat3::from_translation(0.5 * resolution)
            * Mat3::from_scale(vec2(1.0, -1.0) * Vec2::splat(self.zoom * pixel_scale))
            * Mat3::from_translation(-self.position)
    }
}

impl Default for Camera {
    fn default() -> Self {
        Self {
            position: Vec2::ZERO,
            zoom: 21.0
        }
    }
}

/// Draws the simulation to the screen, including a visual representation
/// of the world and the UI.
pub fn draw_simulation(simulation: &mut Simulation) {
    draw_world(simulation);
    egui_macroquad::ui(|ctx| draw_ui(ctx, simulation));
    egui_macroquad::draw();
}

/// Draws the 2D world, including objects and physics engine debug output.
fn draw_world(simulation: &Simulation) {
    clear_background(Color::new(0.8, 0.2, 0.6, 1.0));
    draw_objects(&simulation.camera, &simulation.world);
    draw_contacts(&simulation.camera, simulation.physics.detector.contacts());
}

/// Draws all contacts that were generated this frame.
fn draw_contacts(camera: &Camera, contacts: &[Contact]) {
    let screen_world_matrix = camera.screen_world_matrix(vec2(screen_width(), screen_height()), screen_dpi_scale());
    for contact in contacts {
        let position = screen_world_matrix.transform_point2(contact.position);
        let end_position = screen_world_matrix.transform_point2(contact.position + contact.normal);
        draw_circle(position.x, position.y, 2.0, WHITE);
        draw_arrow(position, end_position, GREEN);
    }
}

/// Draws all objects in the world.
fn draw_objects(camera: &Camera, objects: &PixelWorld) {
    let screen_world_matrix = camera.screen_world_matrix(vec2(screen_width(), screen_height()), screen_dpi_scale());
    for object in objects.values() {
        draw_object(&screen_world_matrix, object);
    }
}

/// Draws a singular object.
fn draw_object(screen_world_matrix: &Mat3, object: &PixelObject) {
    /// The darkened outline size in world units.
    const OUTLINE_SIZE: f32 = 0.05;

    let screen_grid = *screen_world_matrix * object.world_grid_matrix();
    
    let horizontal_offset = screen_grid.transform_vector2(vec2(1.0, 0.0));
    let vertical_offset = screen_grid.transform_vector2(vec2(0.0, 1.0));
    let center_offset = 0.5 * OUTLINE_SIZE * (horizontal_offset + vertical_offset);
    let outline_color = Color::from_vec(0.9 * object.color.to_vec());

    for pixel in object.body.grid().iter() {
        let offset = screen_grid.transform_point2(pixel.as_vec2());
        draw_affine_parallelogram(with_z0(offset), with_z0(horizontal_offset), with_z0(vertical_offset), None, outline_color);
        draw_affine_parallelogram(with_z0(offset + center_offset), with_z0((1.0 - OUTLINE_SIZE) * horizontal_offset), with_z0((1.0 - OUTLINE_SIZE) * vertical_offset), None, object.color);
    }

    let center_of_mass = (*screen_world_matrix * object.transform.to_matrix()).transform_point2(Vec2::ZERO);
    draw_hexagon(center_of_mass.x, center_of_mass.y, 4.0, 2.0, false, DARKGRAY, YELLOW);
}

/// Draws all user interfaces for the simulation.
fn draw_ui(ctx: &egui::Context, simulation: &mut Simulation) {
    update_camera(ctx, simulation);
}

/// Updates the camera position and zoom based upon the `egui` input.
fn update_camera(ctx: &egui::Context, simulation: &mut Simulation) {
    const DRAG_SPEED: f32 = 1.0;
    const SCROLL_SPEED: f32 = 1.003;

    if !ctx.wants_pointer_input() {
        ctx.input(|i| {
            if i.pointer.is_decidedly_dragging() {
                simulation.camera.position += vec2(-i.pointer.delta().x, i.pointer.delta().y) / simulation.camera.zoom;
            }
            
            simulation.camera.zoom = (simulation.camera.zoom * SCROLL_SPEED.powf(i.smooth_scroll_delta.y))
                .clamp(0.5, 35.0);
        });
    }
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

/// Adds a `z = 0.0` component to the vector.
fn with_z0(v: Vec2) -> Vec3 {
    vec3(v.x, v.y, 0.0)
}