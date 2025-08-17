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
    /// Produces a matrix converting from model space to world space.
    pub fn to_matrix(&self) -> Mat3 {
        Mat3::from_scale_angle_translation(Vec2::ONE, self.rotation, self.position)
    }
}

/// A generic three-vector representing motion (such as velocity or acceleration).
#[derive(Copy, Clone, Debug, Default)]
pub struct Motion {
    /// The linear portion of motion.
    pub linear: Vec2,
    /// The rotational portion of motion.
    pub angular: f32
}

impl Motion {
    /// Computes the dot product of this motion vector.
    pub fn dot(self, rhs: Self) -> f32 {
        self.linear.dot(rhs.linear) + self.angular * rhs.angular
    }
}

impl Add for Motion {
    type Output = Self;

    fn add(self, rhs: Self) -> Self::Output {
        Self {
            linear: self.linear + rhs.linear,
            angular: self.angular + rhs.angular
        }
    }
}

impl AddAssign for Motion {
    fn add_assign(&mut self, rhs: Self) {
        *self = *self + rhs;
    }
}


impl Sub for Motion {
    type Output = Self;

    fn sub(self, rhs: Self) -> Self::Output {
        Self {
            linear: self.linear - rhs.linear,
            angular: self.angular - rhs.angular
        }
    }
}

impl SubAssign for Motion {
    fn sub_assign(&mut self, rhs: Self) {
        *self = *self - rhs;
    }
}

impl Mul<f32> for Motion {
    type Output = Self;

    fn mul(self, rhs: f32) -> Self::Output {
        Self {
            linear: rhs * self.linear,
            angular: rhs * self.angular
        }
    }
}

impl MulAssign<f32> for Motion {
    fn mul_assign(&mut self, rhs: f32) {
        *self = *self * rhs;
    }
}

impl Mul<Motion> for f32 {
    type Output = Motion;

    fn mul(self, rhs: Motion) -> Self::Output {
        Motion {
            linear: self * rhs.linear,
            angular: self * rhs.angular
        }
    }
}

/// A generic six-vector representing the motion of two objects.
#[derive(Copy, Clone, Debug, Default)]
pub struct MotionPair(pub [Motion; 2]);

impl MotionPair {
    /// Computes the dot product of this motion vector.
    pub fn dot(self, rhs: Self) -> f32 {
        self[0].dot(rhs[0]) + self[1].dot(rhs[1])
    }
}

impl Add for MotionPair {
    type Output = Self;

    fn add(self, rhs: Self) -> Self::Output {
        Self([self[0] + rhs[0], self[1] + rhs[1]])
    }
}

impl AddAssign for MotionPair {
    fn add_assign(&mut self, rhs: Self) {
        *self = *self + rhs;
    }
}


impl Sub for MotionPair {
    type Output = Self;

    fn sub(self, rhs: Self) -> Self::Output {
        Self([self[0] - rhs[0], self[1] - rhs[1]])
    }
}

impl SubAssign for MotionPair {
    fn sub_assign(&mut self, rhs: Self) {
        *self = *self - rhs;
    }
}

impl Mul<f32> for MotionPair {
    type Output = Self;

    fn mul(self, rhs: f32) -> Self::Output {
        Self(self.0.map(|x| rhs * x))
    }
}

impl MulAssign<f32> for MotionPair {
    fn mul_assign(&mut self, rhs: f32) {
        *self = *self * rhs;
    }
}

impl Mul<MotionPair> for f32 {
    type Output = MotionPair;

    fn mul(self, rhs: MotionPair) -> Self::Output {
        rhs * self
    }
}

impl Deref for MotionPair {
    type Target = [Motion; 2];

    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

impl DerefMut for MotionPair {
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.0
    }
}