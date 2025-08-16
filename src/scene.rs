use crate::*;

/// A simple world with two boxes for testing collision detection.
pub fn simple() -> PixelWorld {
    let mut world = PixelWorld::default();
    world.insert(create_box(ORANGE, Transform { position: vec2(-20.0, -5.0), rotation: 0.0 }, UVec2::splat(2)));
    world.insert(create_box(GOLD, Transform { position: vec2(-18.5, -3.5), rotation: 0.0 }, UVec2::splat(2)));
    world
}

/// Creates a world with a circular object that had jitter issues in the previous engine.
pub fn circle_rotation_jitter() -> PixelWorld {
    let mut world = PixelWorld::default();
    world.insert(create_floor2());
    world.insert(create_circle(ORANGE, Transform { position: vec2(-30.0, -2.5), rotation: 0.2 }, 9.0));
    world
}

/// Creates a box with `color` at `transform`.
fn create_box(color: Color, transform: Transform, extents: UVec2) -> PixelObject {
    let mut grid = PixelGrid::new(extents);
    for y in 0..extents.y {
        for x in 0..extents.x {
            grid.set(uvec2(x, y), true);
        }
    }

    let body = PixelBody::new(grid, false);
    PixelObject::new(body, color, transform)
}

/// Creates a filled circle.
fn create_circle(color: Color, transform: Transform, radius: f32) -> PixelObject {
    let extents = UVec2::splat((2.0 * radius + 1.0) as u32);
    let mut grid = PixelGrid::new(extents);
    for y in 0..extents.y {
        for x in 0..extents.x {
            if (x as f32 - radius).powi(2) + (y as f32 - radius).powi(2) <= radius.powi(2) {
                grid.set(uvec2(x, y), true);
            }
        }
    }

    let body = PixelBody::new(grid, false);

    PixelObject::new(body, color, transform)
}

/// Creates a floor object for testing.
fn create_floor2() -> PixelObject {
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
    let mut body = PixelBody::new(grid, true);

    PixelObject::new(body, DARKGRAY, transform)
}