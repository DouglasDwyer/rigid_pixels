use crate::*;

/// A surface with no friction and no bounciness.
const SMOOTH_HARD: PixelMaterial = PixelMaterial {
    breaking_impulse: 6000.0,
    fracture_pattern: FracturePatternId::Clumps,
    friction: 0.0,
    restitution: 0.0
};

/// A surface with some friction and moderate bounciness.
const ROUGH_SOFT: PixelMaterial = PixelMaterial {
    breaking_impulse: 2000.0,
    fracture_pattern: FracturePatternId::Clumps,
    friction: 0.3,
    restitution: 0.3
};

/// A brittle surface representing glass.
const GLASS: PixelMaterial = PixelMaterial {
    breaking_impulse: 500.0,
    fracture_pattern: FracturePatternId::Shards,
    friction: 0.05,
    restitution: 0.05
};

/// A simple world with two boxes for testing collision detection.
pub fn simple() -> PixelWorld {
    let mut world = PixelWorld::default();
    world.objects.insert(create_floor1(ROUGH_SOFT));
    world.objects.insert(create_box(GOLD, ROUGH_SOFT, Transform { position: vec2(8.25, 5.25), rotation: 0.0 }, uvec2(12, 5)));
    world.objects.insert(create_box(GOLD, ROUGH_SOFT, Transform { position: vec2(10.0, 9.5), rotation: 0.0 }, uvec2(7, 3)));
    world
}

/// A world with a plane and a single box.
pub fn single_box() -> PixelWorld {
    let mut world = PixelWorld::default();
    world.objects.insert(create_floor1(SMOOTH_HARD));
    world.objects.insert(create_box(GOLD, SMOOTH_HARD, Transform { position: vec2(9.0, 12.5), rotation: 0.0 }, uvec2(3, 2)));
    world
}

/// A world with two boxes atop one another.
pub fn double_box() -> PixelWorld {
    let mut world = PixelWorld::default();
    world.objects.insert(create_floor1(ROUGH_SOFT));
    world.objects.insert(create_box(GOLD, ROUGH_SOFT, Transform { position: vec2(8.5, 9.5), rotation: 0.0 }, uvec2(8, 3)));
    world.objects.insert(create_box(GOLD, ROUGH_SOFT, Transform { position: vec2(9.0, 12.5), rotation: 0.0 }, uvec2(5, 2)));
    world
}

/// A world where several boxes (each smaller in size) are stacked atop one another.
pub fn box_pyramid() -> PixelWorld {
    let mut world = PixelWorld::default();
    world.objects.insert(create_floor1(ROUGH_SOFT));
    world.objects.insert(create_box(GOLD, ROUGH_SOFT, Transform { position: vec2(8.25, 15.25), rotation: 0.0 }, uvec2(12, 5)));
    world.objects.insert(create_box(GOLD, ROUGH_SOFT, Transform { position: vec2(8.5, 19.5), rotation: 0.0 }, uvec2(8, 3)));
    world.objects.insert(create_box(GOLD, ROUGH_SOFT, Transform { position: vec2(9.0, 22.5), rotation: 0.0 }, uvec2(5, 2)));
    world
}

/// A world where several boxes (each bigger in size) are stacked atop one another.
pub fn upside_down_box_pyramid() -> PixelWorld {
    let mut world = PixelWorld::default();
    world.objects.insert(create_floor1(ROUGH_SOFT));
    world.objects.insert(create_box(ORANGE, ROUGH_SOFT, Transform { position: vec2(8.25, 17.25), rotation: 0.0 }, uvec2(12, 5)));
    world.objects.insert(create_box(GREEN, ROUGH_SOFT, Transform { position: vec2(8.5, 12.5), rotation: 0.0 }, uvec2(8, 3)));
    world.objects.insert(create_box(SKYBLUE, ROUGH_SOFT, Transform { position: vec2(9.0, 8.5), rotation: 0.0 }, uvec2(5, 2)));
    world
}


/// A world with many boxes inside a container that can rotate.
pub fn tumbler() -> PixelWorld {
    let mut world = PixelWorld::default();
    world.objects.insert(create_tumbler_box(DARKGRAY, ROUGH_SOFT));

    for i in 0..7 {
        for j in 0..7 {
            let position = 3.1 * uvec2(i, j).as_vec2() + Vec2::splat(-12.0);
            world.objects.insert(create_box(GOLD, ROUGH_SOFT, Transform { position, rotation: 0.0 }, uvec2(3, 3)));
        }
    }

    world
}

/// A world where the corners of stacked boxes exhibit some strange artifacts from the contact normals.
pub fn weird_normals() -> PixelWorld {
    let mut world = PixelWorld::default();
    world.objects.insert(create_floor1(ROUGH_SOFT));
    world.objects.insert(create_box(GOLD, ROUGH_SOFT, Transform { position: vec2(8.25, 5.25), rotation: 0.0 }, uvec2(12, 5)));
    world.objects.insert(create_box(GOLD, ROUGH_SOFT, Transform { position: vec2(10.0, 9.5), rotation: 0.0 }, uvec2(7, 3)));
    world
}

/// Creates a world with a circular object that had jitter issues in the previous engine.
pub fn circle_rotation_jitter() -> PixelWorld {
    let mut world = PixelWorld::default();
    world.objects.insert(create_floor2(SMOOTH_HARD));
    world.objects.insert(create_circle(ORANGE, SMOOTH_HARD, Transform { position: vec2(-30.0, 10.5), rotation: 0.2 }, 9.0));
    world
}

/// Creates a world to test the hinge joint.
pub fn hinge_joint() -> PixelWorld {
    let mut world = PixelWorld::default();
    let g = world.objects.insert(create_floor1(ROUGH_SOFT));
    let a = world.objects.insert(create_box(GOLD, ROUGH_SOFT, Transform { position: vec2(15.25, 8.25), rotation: 0.0 }, uvec2(2, 3)));
    let b = world.objects.insert(create_box(GOLD, ROUGH_SOFT, Transform { position: vec2(15.25, 14.25), rotation: 0.0 }, uvec2(3, 4)));
    world.insert_joint([b, a], Transform::new(vec2(8.25, 15.25), 0.0), JointDescriptor::slider(-5.0..=5.0));
    //world.insert_joint([a, g], Transform::new(vec2(8.25, 15.25), 0.0), JointDescriptor::hinge().x_limits(0.0..=5.0));
    world
}

/// Creates a world with a breakable, vertical glass pane.
pub fn glass_pane() -> PixelWorld {
    let mut world = PixelWorld::default();
    
    world.objects.insert(create_floor1(ROUGH_SOFT));
    world.objects.insert(create_circle(ORANGE, SMOOTH_HARD, Transform { position: vec2(-30.0, 6.5), rotation: 0.2 }, 5.0));
    world.objects.insert(create_immobile_box(SKYBLUE, GLASS, Transform { position: vec2(30.0, 26.5), rotation: 0.0 }, uvec2(3, 50)));

    world
}

/// Creates a box with `color` at `transform`.
fn create_box(color: Color, material: PixelMaterial, transform: Transform, extents: UVec2) -> PixelObject {
    let mut grid = PixelGrid::new(extents);
    for y in 0..extents.y {
        for x in 0..extents.x {
            grid.set(uvec2(x, y), true);
        }
    }

    let body = PixelBody::new(grid, material, false, false);
    PixelObject::new(body, color, transform)
}

/// Creates a box with `color` at `transform` that cannot be moved.
fn create_immobile_box(color: Color, material: PixelMaterial, transform: Transform, extents: UVec2) -> PixelObject {
    let mut grid = PixelGrid::new(extents);
    for y in 0..extents.y {
        for x in 0..extents.x {
            grid.set(uvec2(x, y), true);
        }
    }

    let body = PixelBody::new(grid, material, true, true);
    PixelObject::new(body, color, transform)
}

/// Creates a hollow box with a fixed center.
fn create_tumbler_box(color: Color, material: PixelMaterial) -> PixelObject {
    let extents = UVec2::splat(32);
    let thickness = 3;
    let mut grid = PixelGrid::new(extents);
    for y in 0..extents.y {
        for x in 0..extents.x {
            grid.set(uvec2(x, y), true);
        }
    }

    for y in thickness..extents.y - thickness {
        for x in thickness..extents.x - thickness {
            grid.set(uvec2(x, y), false);
        }
    }

    let body = PixelBody::new(grid, material, true, false);
    PixelObject::new(body, color, Transform::IDENTITY)
}

/// Creates a filled circle.
fn create_circle(color: Color, material: PixelMaterial, transform: Transform, radius: f32) -> PixelObject {
    let extents = UVec2::splat((2.0 * radius + 1.0) as u32);
    let mut grid = PixelGrid::new(extents);
    for y in 0..extents.y {
        for x in 0..extents.x {
            if (x as f32 - radius).powi(2) + (y as f32 - radius).powi(2) <= radius.powi(2) {
                grid.set(uvec2(x, y), true);
            }
        }
    }

    let body = PixelBody::new(grid, material, false, false);

    PixelObject::new(body, color, transform)
}

/// Creates a flat floor object for testing.
fn create_floor1(material: PixelMaterial) -> PixelObject {
    const FLOOR_HEIGHT: u32 = 16;
    const FLOOR_LENGTH: u32 = 512;

    let mut grid = PixelGrid::new(uvec2(FLOOR_LENGTH, 25));
    for x in 0..FLOOR_LENGTH {
        for y in 0..FLOOR_HEIGHT {
            grid.set(uvec2(x, y), true);
        }
    }

    let transform = Transform { position: vec2(0.0, -6.5), rotation: 0.0 };
    let mut body = PixelBody::new(grid, material, true, true);

    PixelObject::new(body, DARKGRAY, transform)
}

/// Creates a floor object for testing.
fn create_floor2(material: PixelMaterial) -> PixelObject {
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

    let transform = Transform { position: Vec2::ZERO, rotation: 0.0 };
    let mut body = PixelBody::new(grid, material, true, true);

    PixelObject::new(body, DARKGRAY, transform)
}