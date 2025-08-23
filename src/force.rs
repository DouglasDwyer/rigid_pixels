use macroquad::prelude::*;

/// Models a soft spring force.
#[derive(Copy, Clone, Debug)]
pub struct Spring {
    /// The spring constant.
    pub k: f32,
    /// The length at rest.
    pub rest_length: f32,
    /// Damping to apply to the motion.
    pub drag: f32,
    /// The origin of the spring.
    pub origin: Vec2
}

impl Spring {
    /// Gets the force that will be applied at point `position` (relative to `origin`)
    /// which is moving at `velocity`.
    pub fn force(&self, position: Vec2, velocity: Vec2) -> Vec2 {
        let displacement = position - self.origin;
        let magnitude = self.k * (self.rest_length - displacement.length());
        let acceleration_term = magnitude * displacement.try_normalize().unwrap_or(Vec2::X);
        let drag_term = self.drag * velocity;
        acceleration_term - drag_term
    }
}