use crate::*;

/*
TODO:
- NGS
- Restitution support
- PGS
- PGS soft
- Research normals issue
- Joints
  > Position constraint
  > Rotation constraint
  > See about force limits? How to deal with friction?
- Try other solvers
*/

/// A simple world with two boxes for testing collision detection.
pub fn simple() -> PixelWorld {
    let mut world = PixelWorld::default();
    world.insert(create_floor1(0.2));
    world.insert(create_box(GOLD, 0.2, Transform { position: vec2(8.25, 5.25), rotation: 0.0 }, uvec2(12, 5)));
    world.insert(create_box(GOLD, 0.2, Transform { position: vec2(10.0, 9.5), rotation: 0.0 }, uvec2(7, 3)));
    world
}

/// A world with a plane and a single box.
pub fn single_box() -> PixelWorld {
    let mut world = PixelWorld::default();
    world.insert(create_floor1(0.2));
    world.insert(create_box(GOLD, 0.2, Transform { position: vec2(9.0, 12.5), rotation: 0.0 }, uvec2(3, 2)));
    world
}

/// A world with two boxes atop one another.
pub fn double_box() -> PixelWorld {
    let mut world = PixelWorld::default();
    world.insert(create_floor1(0.2));
    world.insert(create_box(GOLD, 0.2, Transform { position: vec2(8.5, 9.5), rotation: 0.0 }, uvec2(8, 3)));
    world.insert(create_box(GOLD, 0.2, Transform { position: vec2(9.0, 12.5), rotation: 0.0 }, uvec2(5, 2)));
    world
}

/// A world where several boxes (each smaller in size) are stacked atop one another.
pub fn box_pyramid() -> PixelWorld {
    let mut world = PixelWorld::default();
    world.insert(create_floor1(0.2));
    world.insert(create_box(GOLD, 0.2, Transform { position: vec2(8.25, 15.25), rotation: 0.0 }, uvec2(12, 5)));
    world.insert(create_box(GOLD, 0.2, Transform { position: vec2(8.5, 19.5), rotation: 0.0 }, uvec2(8, 3)));
    world.insert(create_box(GOLD, 0.2, Transform { position: vec2(9.0, 22.5), rotation: 0.0 }, uvec2(5, 2)));
    world
}

/// A world where several boxes (each bigger in size) are stacked atop one another.
pub fn upside_down_box_pyramid() -> PixelWorld {
    let mut world = PixelWorld::default();
    world.insert(create_floor1(0.2));
    world.insert(create_box(GOLD, 0.2, Transform { position: vec2(8.25, 19.25), rotation: 0.0 }, uvec2(12, 5)));
    world.insert(create_box(GOLD, 0.2, Transform { position: vec2(8.5, 14.5), rotation: 0.0 }, uvec2(8, 3)));
    world.insert(create_box(GOLD, 0.2, Transform { position: vec2(9.0, 8.5), rotation: 0.0 }, uvec2(5, 2)));
    world
}

/// A world where the corners of stacked boxes exhibit some strange artifacts from the contact normals.
pub fn weird_normals() -> PixelWorld {
    let mut world = PixelWorld::default();
    world.insert(create_floor1(0.2));
    world.insert(create_box(GOLD, 0.2, Transform { position: vec2(8.25, 5.25), rotation: 0.0 }, uvec2(12, 5)));
    world.insert(create_box(GOLD, 0.2, Transform { position: vec2(10.0, 9.5), rotation: 0.0 }, uvec2(7, 3)));
    world
}

/// Creates a world with a circular object that had jitter issues in the previous engine.
pub fn circle_rotation_jitter() -> PixelWorld {
    let mut world = PixelWorld::default();
    world.insert(create_floor2(0.0));
    world.insert(create_circle(ORANGE, 0.0, Transform { position: vec2(-30.0, -2.5), rotation: 0.2 }, 9.0));
    world
}

/// Creates a box with `color` at `transform`.
fn create_box(color: Color, friction: f32, transform: Transform, extents: UVec2) -> PixelObject {
    let mut grid = PixelGrid::new(extents);
    for y in 0..extents.y {
        for x in 0..extents.x {
            grid.set(uvec2(x, y), true);
        }
    }

    let body = PixelBody::new(grid, friction, false);
    PixelObject::new(body, color, transform)
}

/// Creates a filled circle.
fn create_circle(color: Color, friction: f32, transform: Transform, radius: f32) -> PixelObject {
    let extents = UVec2::splat((2.0 * radius + 1.0) as u32);
    let mut grid = PixelGrid::new(extents);
    for y in 0..extents.y {
        for x in 0..extents.x {
            if (x as f32 - radius).powi(2) + (y as f32 - radius).powi(2) <= radius.powi(2) {
                grid.set(uvec2(x, y), true);
            }
        }
    }

    let body = PixelBody::new(grid, friction, false);

    PixelObject::new(body, color, transform)
}

/// Creates a flat floor object for testing.
fn create_floor1(friction: f32) -> PixelObject {
    const FLOOR_LENGTH: u32 = 512;

    let mut grid = PixelGrid::new(uvec2(FLOOR_LENGTH, 25));
    for x in 0..FLOOR_LENGTH {
        grid.set(uvec2(x, 0), true);
        grid.set(uvec2(x, 1), true);
        grid.set(uvec2(x, 2), true);
        grid.set(uvec2(x, 3), true);
        grid.set(uvec2(x, 4), true);
        grid.set(uvec2(x, 5), true);
    }

    let transform = Transform { position: -0.5 * grid.resolution().x as f32 * Vec2::X, rotation: 0.0 };
    let mut body = PixelBody::new(grid, friction, true);

    PixelObject::new(body, DARKGRAY, transform)
}

/// Creates a floor object for testing.
fn create_floor2(friction: f32) -> PixelObject {
    const FLOOR_LENGTH: u32 = 512;

    let mut grid = PixelGrid::new(uvec2(FLOOR_LENGTH, 25));
    for x in 0..FLOOR_LENGTH {
        let sinx = 5.0 * (0.1 * (x as f32)).sin();
        for y in 0..5 {
            if (y as f32) < sinx {
                grid.set(uvec2(x, y), true);
            }
        }
        
        grid.set(uvec2(x, 0), true);
    }
    for y in 0..25 {
        grid.set(uvec2(256, y), true);
    }
    grid.set(uvec2(256, 1), true);

    grid.set(uvec2(239, 10), true);

    let transform = Transform { position: -0.5 * grid.resolution().as_vec2(), rotation: 0.0 };
    let mut body = PixelBody::new(grid, friction, true);

    PixelObject::new(body, DARKGRAY, transform)
}