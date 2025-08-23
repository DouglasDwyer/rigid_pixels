use crate::pixel::*;
use macroquad::prelude::*;
use std::ops::*;

/// Describes the current location and orientation of an object.
#[derive(Copy, Clone, Debug, Default)]
pub struct Transform {
    /// The position of the object's center of mass.
    pub position: Vec2,
    /// The rotation from the origin.
    pub rotation: f32
}

impl Transform {
    /// A transformation that keeps all points the same.
    pub const IDENTITY: Self = Self::new(Vec2::ZERO, 0.0);

    /// Creates a new transform with the specified position and rotation.
    pub const fn new(position: Vec2, rotation: f32) -> Self {
        Self { position, rotation }
    }

    /// Updates the position and rotation based upon `velocity` over a time interval `dt`.
    pub fn integrate_velocity(self, dt: f32, velocity: Velocity) -> Self {
        Self::new(self.position + dt * velocity.linear, (self.rotation + dt * velocity.angular).rem_euclid(std::f32::consts::TAU))
    }

    /// Produces a matrix converting from model space to world space.
    pub fn to_matrix(self) -> Mat3 {
        Mat3::from_scale_angle_translation(Vec2::ONE, self.rotation, self.position)
    }
}

impl Mul<Vec2> for Transform {
    type Output = Vec2;

    fn mul(self, rhs: Vec2) -> Self::Output {
        rhs.rotate(Vec2::from_angle(self.rotation)) + self.position
    }
}

/// How fast an object is moving, and in what direction.
#[derive(Copy, Clone, Debug, Default)]
pub struct Velocity {
    /// The linear portion of motion.
    pub linear: Vec2,
    /// The rotational portion of motion.
    pub angular: f32
}

impl Velocity {
    /// Updates this velocity based upon an application of `force` over a time interval `dt`.
    pub fn integrate_force(self, dt: f32, force: ForceAccumulator, body: &PixelBody) -> Self {
        Self {
            linear: self.linear + dt * body.inverse_mass() * force.force,
            angular: self.angular + dt * body.inverse_inertia_tensor() * force.torque
        }
    }
}

impl Add for Velocity {
    type Output = Self;

    fn add(self, rhs: Self) -> Self::Output {
        Self {
            linear: self.linear + rhs.linear,
            angular: self.angular + rhs.angular
        }
    }
}

impl AddAssign for Velocity {
    fn add_assign(&mut self, rhs: Self) {
        *self = *self + rhs;
    }
}

impl Sub for Velocity {
    type Output = Self;

    fn sub(self, rhs: Self) -> Self::Output {
        Self {
            linear: self.linear - rhs.linear,
            angular: self.angular - rhs.angular
        }
    }
}

impl SubAssign for Velocity {
    fn sub_assign(&mut self, rhs: Self) {
        *self = *self - rhs;
    }
}

impl Mul<f32> for Velocity {
    type Output = Self;

    fn mul(self, rhs: f32) -> Self::Output {
        Self {
            linear: rhs * self.linear,
            angular: rhs * self.angular
        }
    }
}

impl MulAssign<f32> for Velocity {
    fn mul_assign(&mut self, rhs: f32) {
        *self = *self * rhs;
    }
}

impl Mul<Velocity> for f32 {
    type Output = Velocity;

    fn mul(self, rhs: Velocity) -> Self::Output {
        Velocity {
            linear: self * rhs.linear,
            angular: self * rhs.angular
        }
    }
}

/// Computes the outer product of two [`Vec2`]s.
pub fn vec2_outer_product(lhs: Vec2, rhs: Vec2) -> Mat2 {
    Mat2::from_cols(lhs * rhs.x, lhs * rhs.y)
}