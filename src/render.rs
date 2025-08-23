use crate::*;

/// Stores state for rendering and handling user input.
#[derive(Debug, Default)]
pub struct Renderer {
    /// The camera dictating what will be rendered.
    camera: Camera,
    /// The ID of the object currently being dragged.
    dragged_object: Option<DraggedObject>
}

impl Renderer {
    /// Draws the simulation to the screen, including a visual representation
    /// of the world and the UI.
    pub fn draw(&mut self, physics: &PhysicsEngine, world: &mut PixelWorld) {
        self.draw_world(physics, world);
        egui_macroquad::ui(|ctx| self.draw_ui(ctx, physics, world));
        egui_macroquad::draw();
    }

    /// Draws the 2D world, including objects and physics engine debug output.
    fn draw_world(&self, physics: &PhysicsEngine, world: &PixelWorld) {
        clear_background(Color::new(0.8, 0.2, 0.6, 1.0));
        self.draw_objects(world);
        self.draw_contacts(physics.detector.contacts());
    }

    /// Draws all contacts that were generated this frame.
    fn draw_contacts(&self, contacts: &[Contact]) {
        let screen_world_matrix = self.camera.screen_world_matrix(vec2(screen_width(), screen_height()), screen_dpi_scale());
        for contact in contacts {
            let position = screen_world_matrix.transform_point2(contact.position);
            let end_position = screen_world_matrix.transform_point2(contact.position + contact.normal);
            draw_circle(position.x, position.y, 2.0, WHITE);
            Self::draw_arrow(position, end_position, GREEN);
        }
    }

    /// Draws all objects in the world.
    fn draw_objects(&self, objects: &PixelWorld) {
        let screen_world_matrix = self.camera.screen_world_matrix(vec2(screen_width(), screen_height()), screen_dpi_scale());
        for object in objects.values() {
            Self::draw_object(&screen_world_matrix, object);
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
            draw_affine_parallelogram(Self::with_z0(offset), Self::with_z0(horizontal_offset), Self::with_z0(vertical_offset), None, outline_color);
            draw_affine_parallelogram(Self::with_z0(offset + center_offset), Self::with_z0((1.0 - OUTLINE_SIZE) * horizontal_offset), Self::with_z0((1.0 - OUTLINE_SIZE) * vertical_offset), None, object.color);
        }

        let center_of_mass = (*screen_world_matrix * object.transform.to_matrix()).transform_point2(Vec2::ZERO);
        draw_hexagon(center_of_mass.x, center_of_mass.y, 4.0, 2.0, false, DARKGRAY, YELLOW);
    }

    /// Draws all user interfaces for the simulation.
    fn draw_ui(&mut self, ctx: &egui::Context, physics: &PhysicsEngine, world: &mut PixelWorld) {
        //self.draw_info_panel(ctx, physics, world);
        self.update_dragged_object(ctx, world);
        self.update_camera(ctx);
    }

    /// Draws an info UI panel on the right side of the screen.
    fn draw_info_panel(&mut self, ctx: &egui::Context, physics: &PhysicsEngine, world: &mut PixelWorld) {
        egui::SidePanel::right("Info panel")
            .min_width(500.0)
            .show(ctx, |ui| {
            for (id, body) in world {
                ui.label(format!("{id:?}"));
                ui.label(format!("Velocity: {:?}", body.velocity));
            }
        });
    }

    /// Updates the camera position and zoom based upon the `egui` input.
    fn update_camera(&mut self, ctx: &egui::Context) {
        const DRAG_SPEED: f32 = 1.0;
        const SCROLL_SPEED: f32 = 1.003;

        if !ctx.wants_pointer_input() {
            ctx.input(|i| {
                if i.pointer.is_decidedly_dragging() && i.pointer.secondary_down() {
                    self.camera.position += vec2(-i.pointer.delta().x, i.pointer.delta().y) / self.camera.zoom;
                }
                
                self.camera.zoom = (self.camera.zoom * SCROLL_SPEED.powf(i.smooth_scroll_delta.y))
                    .clamp(0.5, 50.0);
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

    fn update_dragged_object(&mut self, ctx: &egui::Context, world: &mut PixelWorld) {
        let wants_pointer_input = ctx.wants_pointer_input();
        ctx.input(|i| {
            let screen_world_matrix = self.camera.screen_world_matrix(vec2(screen_width(), screen_height()), screen_dpi_scale());
            if !wants_pointer_input
                && i.pointer.primary_down()
                && let Some(latest_pos) = i.pointer.latest_pos() {
                let position = screen_world_matrix.inverse().transform_point2(vec2(latest_pos.x, latest_pos.y));
                if let Some(dragged) = self.dragged_object {
                    let spring = Spring {
                        k: 1000.0,
                        rest_length: 0.0,
                        drag: 250.0,
                        origin: position
                    };

                    let object = &mut world[dragged.id];
                    let world_space_point = object.transform.to_matrix().transform_point2(dragged.relative_position);
                    let point_velocity = Vec2::Y.rotate(world_space_point - object.transform.position) * object.velocity.angular + object.velocity.linear;
                    //println!("{}", (world_space_point - object.transform.position).length() * object.velocity.angular);
                    //println!("{}", object.body.inverse_inertia_tensor());
                    //println!("{:?} {:?} {:?} {:?} {:?}", object.transform, object.velocity, dragged.relative_position, world_space_point, spring.force(world_space_point, point_velocity));
                    
                    object.velocity.angular = 0.0;
                    object.add_force(world_space_point, spring.force(world_space_point, point_velocity));
                }
                else if i.pointer.primary_pressed() {
                    self.select_dragged_object(position, world);
                }
            }
            else {
                self.dragged_object = None;
            }
        });
    }

    /// Selects a new object for a click at world-space `position`.
    fn select_dragged_object(&mut self, position: Vec2, world: &mut PixelWorld) {
        for (id, object) in world {
            let clicked_position = object.world_grid_matrix().inverse().transform_point2(position);
            if object.body.grid().get_or_empty(clicked_position.floor().as_ivec2().as_uvec2()) {
                object.velocity = Velocity::default();
                self.dragged_object = Some(DraggedObject {
                    id,
                    relative_position: object.transform.to_matrix().inverse().transform_point2(position)
                });
                break;
            }
        }
    }

    /// Adds a `z = 0.0` component to the vector.
    fn with_z0(v: Vec2) -> Vec3 {
        vec3(v.x, v.y, 0.0)
    }
}

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

/// Holds information about an object being dragged.
#[derive(Copy, Clone, Debug)]
struct DraggedObject {
    /// The ID of the object being dragged.
    pub id: ObjectId,
    /// The position in body space that was clicked.
    pub relative_position: Vec2
}