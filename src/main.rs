#![allow(unused)]

#![feature(vec_from_fn)]

use egui_macroquad::egui;
use macroquad::miniquad::window::set_window_size;
use macroquad::prelude::*;
use macroquad::rand::*;

use self::detector::*;
use self::force::*;
use self::fracturing::*;
use self::math::*;
use self::pixel::*;
use self::render::*;
use self::render::Camera;
use self::solver::*;
use slotmap::*;
use std::ops::*;

/// Implements logic for detecting collisions between objects.
mod detector;

/// Breaks objects up into smaller pieces for destruction physics.
mod fracturing;

/// Defines external force generators.
mod force;

/// Defines various math types used for calculations.
mod math;

/// Defines rigid bodies made of pixels.
mod pixel;

/// Handles drawing the simulation.
mod render;

/// Defines worlds for testing.
mod scene;

/// Provides algorithms for resolving contacts and constraints.
mod solver;

/// Holds all state for the world simulation.
pub struct Simulation {
    /// Lookup table for fracture patterns.
    fracture_lut: FractureLut,
    /// The time of the last physics update.
    last_update: f64,
    /// The selected algorithms for physics.
    physics: PhysicsEngine,
    /// Handles user input and drawing the world to the screen.
    renderer: Renderer,
    /// All objects in the simulation.
    world: PixelWorld
}

impl Simulation {
    /// Initializes a new simulation.
    pub fn new(physics: PhysicsEngine, start_time: f64, world: PixelWorld) -> Self {
        Self {
            fracture_lut: FractureLut::generate(),
            last_update: start_time,
            physics,
            renderer: Renderer::default(),
            world
        }
    }

    /// Updates the simulation, advancing to `time` (since simulation start in seconds).
    pub fn update(&mut self, time: f64) {
        clear_background(Color::new(0.8, 0.2, 0.6, 1.0));

        let mouse_object = &mut self.world.objects[ObjectId::stat()];
        let mut new_mouse_position = mouse_object.transform.position;
        egui_macroquad::cfg(|ctx| ctx.input(|i| {
            if let Some(latest_pos) = i.pointer.latest_pos() {
                let screen_world_matrix = self.renderer.camera().screen_world_matrix(vec2(screen_width(), screen_height()), screen_dpi_scale());
                new_mouse_position = screen_world_matrix.inverse().transform_point2(vec2(latest_pos.x, latest_pos.y));
            }
        }));

        mouse_object.velocity = Screw { linear: (new_mouse_position - mouse_object.transform.position) / get_frame_time(), angular: 0.0 };

        self.test_force_generator();
        
        while self.last_update < time {
            self.physics.detector.update(&self.world, self.physics.delta_time);
            self.physics.solver.solve(self.physics.detector.contacts(), &mut self.world, self.physics.delta_time);

            for event in self.physics.solver.events().iter().copied() {
                match event {
                    SolverEvent::Fracture(fracture) => if self.world.objects.contains_key(fracture.object) {
                        self.fracture_lut.apply(rand(), fracture, &mut self.world)
                    }
                }
            }

            self.last_update += self.physics.delta_time as f64;
        }

        self.despawn_far_objects();
        self.clear_force_accumulators();

        self.world.objects[ObjectId::stat()].transform.position = new_mouse_position;
        
        self.renderer.draw(&self.physics, &mut self.world);
    }

    /// Clears all external forces to progress to the next frame.
    fn clear_force_accumulators(&mut self) {
        for object in self.world.objects.values_mut() {
            object.forces = ForceAccumulator::default()
        }
    }

    /// Removes objects that are far away from the screen's center.
    fn despawn_far_objects(&mut self) {
        /// The maximum distance that any object may be before being removed.
        const MAX_DISTANCE: f32 = 1000.0;

        for (id, object) in &mut self.world.objects {
            assert!(object.transform.is_finite(), "{:?}", object.transform);

            if MAX_DISTANCE < object.transform.position.abs().max_element() {
                println!("Despawn {id:?}");
                self.world.objects.remove(id);
                break;
            }
        }
    }

    /// Adds a test force to all objects in the scene.
    fn test_force_generator(&mut self) {
        for object in self.world.objects.values_mut() {
            if 0.0 < object.body.inverse_mass() {
                object.forces.force += -G * Vec2::Y / object.body.inverse_mass();
            }
        }
    }
}

const G: f32 = 9.81 * 16.0;

/// Holds the algorithms and data for running the physics engine.
#[derive(Debug)]
pub struct PhysicsEngine {
    /// The detector to use for identifying collisions.
    pub detector: Detector,
    /// The solver to use for constraint resolution.
    pub solver: Solver,
    /// The amount of time (in seconds) between full simulation steps.
    pub delta_time: f32
}

/// Information about a hard contact found by the collision [`Detector`].
#[derive(Copy, Clone, Debug)]
pub struct Contact {
    /// The objects involved in the collision.
    pub objects: [ObjectId; 2],
    /// The position of each pixel involved in the collision. 
    pub pixel_position: [UVec2; 2],
    /// The offset from each object's origin to the contact in local space.
    pub local_position: [Vec2; 2],
    /// The material properties at the contact.
    pub material: PixelMaterialPair,
    /// The normal (in world space) of object `0`'s surface.
    pub normal: Vec2,
    /// The position of the contact in world space.
    pub position: Vec2
}

impl Contact {
    /// Gets a lightweight ID that can be used to track the contact across frames.
    pub fn id(&self) -> ContactId {
        ContactId { objects: self.objects, pixel_position: self.pixel_position }
    }

    /// Computes the current displacement of the contacts along the normal axis,
    /// relative to the required `penetration` for this contact.
    pub fn separation(&self, world: &PixelWorld) -> f32 {
        self.normal.dot(world.objects[self.objects[1]].transform * self.local_position[1]
            - world.objects[self.objects[0]].transform * self.local_position[0])
    }

    /// Swaps the order of the objects involved in the collision.
    pub fn swap_objects(&self) -> Self {
        Self {
            objects: [self.objects[1], self.objects[0]],
            pixel_position: [self.pixel_position[1], self.pixel_position[0]],
            local_position: [self.local_position[1], self.local_position[0]],
            normal: -self.normal,
            ..*self
        }
    }
}

/// A persistent constraint that limits the motion between a point on two bodies.
#[derive(Clone, Debug)]
pub struct Joint {
    /// The objects that are constrained.
    pub objects: [ObjectId; 2],
    /// The ID of the joint.
    pub id: JointId,
    /// The location of the joint in each object's local coordinate system.
    pub local_transform: [Transform; 2],
    /// The maximum amount of force that this joint may exert.
    pub max_force: f32,
    /// The maximum amount of torque that this joint may exert.
    pub max_torque: f32,
    /// The minimum allowed translation along the joint axes.
    pub translation_min: Vec2,
    /// The maximum allowed translation along the joint axes.
    pub translation_max: Vec2,
    /// The minimum allowed rotation along the joint axes.
    pub rotation_min: f32,
    /// The maximum allowed rotation along the joint axes.
    pub rotation_max: f32
}

/// Defines the local properties of a [`Joint`].
#[derive(Clone, Debug)]
pub struct JointDescriptor {
    /// The maximum amount of force that this joint may exert.
    max_force: f32,
    /// The maximum amount of torque that this joint may exert.
    max_torque: f32,
    /// The maximum allowed translation along the joint axes.
    translation_max: Vec2,
    /// The minimum allowed translation along the joint axes.
    translation_min: Vec2,
    /// The maximum allowed rotation along the joint axes.
    rotation_max: f32,
    /// The minimum allowed rotation along the joint axes.
    rotation_min: f32
}

impl JointDescriptor {
    /// Allow no forms of motion.
    pub fn fixed() -> Self {
        Self {
            max_force: f32::MAX,
            max_torque: f32::MAX,
            translation_max: Vec2::ZERO,
            translation_min: Vec2::ZERO,
            rotation_max: 0.0,
            rotation_min: 0.0
        }
    }

    /// Allow all forms of motion.
    pub fn free() -> Self {
        Self {
            max_force: f32::MAX,
            max_torque: f32::MAX,
            translation_max: Vec2::MAX,
            translation_min: Vec2::MIN,
            rotation_max: f32::MAX,
            rotation_min: f32::MIN
        }
    }

    /// Prevent translation along two axes.
    /// Allow rotation around one axis.
    pub fn hinge() -> Self {
        Self {
            translation_min: Vec2::ZERO,
            translation_max: Vec2::ZERO,
            ..Self::free()
        }
    }
    
    /// Prevent translation along one axis.
    /// Prevent rotation around one axis.
    pub fn slider(range: RangeInclusive<f32>) -> Self {
        Self {
            rotation_max: 0.0,
            rotation_min: 0.0,
            translation_max: vec2(*range.end(), 0.0),
            translation_min: vec2(*range.start(), 0.0),
            ..Self::free()
        }
    }

    /// Allow translation along both axes.
    /// Prevent rotation around one axis.
    pub fn rotational() -> Self {
        Self {
            rotation_min: 0.0,
            rotation_max: 0.0,
            ..Self::free()
        }
    }

    /// Specifies the maximum force magnitude that this joint exerts
    /// to satisfy positional constraints.
    pub fn max_force(self, value: f32) -> Self {
        Self {
            max_force: value,
            ..self
        }
    }

    /// Specifies the maximum torque magnitude that this joint exerts
    /// to satisfy rotational constraints.
    pub fn max_torque(self, value: f32) -> Self {
        Self {
            max_torque: value,
            ..self
        }
    }
}

/// An event triggered by the interaction of objects during the collision solving process.
#[derive(Copy, Clone, Debug)]
pub enum SolverEvent {
    /// A fracture occurred on an object.
    Fracture(Fracture)
}

/// Describes a fracture that should be applied to a body.
#[derive(Copy, Clone, Debug)]
pub struct Fracture {
    /// The impulse that generated the fracture.
    pub impulse: Vec2,
    /// The objects involved in the fracture.
    pub object: ObjectId,
    /// How the fracture should occur.
    pub pattern: FracturePatternId,
    /// The origin of the fracture, in pixel coordinates.
    pub pixel_position: UVec2,
    /// The ratio of [`Self::impulse`] to the breaking impulse.
    /// Always greater than `1.0`.
    pub strength_ratio: f32
}

/// Allows for uniquely identifying a contact across frames.
#[derive(Copy, Clone, Debug, Hash, PartialEq, Eq)]
pub struct ContactId {
    /// The objects involved in the collision.
    objects: [ObjectId; 2],
    /// The position of each pixel involved in the collision. 
    pixel_position: [UVec2; 2]
}

/// Allows for uniquely identifying a joint across frames.
#[derive(Copy, Clone, Debug, Hash, PartialEq, Eq)]
pub struct JointId(u32);

/// Executes the main loop.
#[macroquad::main("Rigid pixels")]
async fn main() {
    set_window_size(1000, 1000);
    let mut simulation = Simulation::new(PhysicsEngine {
        detector: Detector::new(DetectorKind::Speculative { include_external_forces: true, mode: SpeculativeStepMode::Equidistant }, GeometryKind::Surface),
        solver: Solver::SequentialImpulse(SequentialImpulse::new(SolverConfig {
            baumgarte: 0.2,
            relaxation_iterations: 1,
            substeps: 4,
            velocity_iterations: 1,
            warm_starting: true
        })),
        delta_time: 0.02
    }, get_time(), scene::box_pyramid());

    loop {
        simulation.update(get_time());
        next_frame().await;
    }
}

/*

Improvements:
- Integrate position AFTER applying constraints to velocity
  > Jitter on boxes is muched reduced
- Use warm starting to increase stability
  > Improvement with big circle lying on ground; previously jittered but now does not
  > Box stack slides without warm starting
  > Tried scaling warm starting by some factor (like 0.8), but this led to sliding. Doesn't seem to be worth it.
- Speculative contacts can work for CCD. External forces NEED to be included in the objects' speculative trajectory.
  > With naive collision detection, falling objects or dragging could cause clipping
  > Without external forces, dragging could clip objects through floor
  > Only speculative contacts with separation greater than zero should count. Separation less implies object clipped through wall.
  > Speculative contacts can be deduplicated since multiple may occur on the same voxel. Earlier contacts should be preferred
  > In the solver, speculative contacts should be handled by removing JUST ENOUGH velocity that separation is zero at the end of the time step
    (this implies restitution must be handled after solving, because otherwise objects will not hit the ground before bouncing)
- Using sequential impulses allows for robust handling of normal/friction simultaneously
  > The impulse vector must be clipped AFTER dynamic friction is calculated
- Linear slop is necessary to ensure the collision detector doesn't miss contacts that are close together
  > Speculative collisions alone did not fix the issue of occasionally missing contacts
- TGS/Baumgarte performs better than pure PGS or NGS on the upside-down pyramid
  > This is using 5 substeps w/ 3 velocity iterations apiece
  > There is some bounciness to the collisions but the stack is otherwise stable
  > The tumbler works as well
  > Apply restitution once on the final impulses output by the solver
  > NGS does not seem to be a signficant improvement; TGS is the big win
> Corner-corner normal clamping: check (if in either reference frame) the normal is pointing into a voxel's neighbors.
  If so, clamp the normal so that it's perpendicular to the edge.
  > This prevents artifacts and strange "bumps" when corners slide past one another

Things to think about:
- Dropping the box stack from high up causes interpenetration
  > Is this a problem with speculative contacts? Or is it an issue with the solver?
    Or the order of detection, solving, and velocity integration?
  > My guess is speculative contacts in general don't handle this case. Is there any workaround?

- There were NaNs in collision detection when voxels clipped through the opposite side of an edge. How did the 3D engine handle this?

- Box2D tracks something called totalNormalImpulse to check whether any impulse was EVER generated for a speculative contact.
  It prevents restitution from being applied to contacts without it. How important is this?



- Big flaw in the joints math - constraints were adding energy. Constraints should only act along axes which are fixed
  > How do we simulateneously solve for multiple constrained axes? How do we handle axes with limits?

- Maintaining constraint velocity is important.
  > This means for things like mouse constraints, the position can't be "teleported" while the velocity stays 0
  > To get good-looking results, either need to tie all constraints to kinematic bodies, or have kinematic tracking of constraint positions
  > Does it make sense to allow for modifying a constraint after its creation? If yes, does it make sense to allow for a "kinematic constraint"
    where the target position moves dynamically?
  > What does Unity's API look like for this?
- How should we handle joints if a body gets deleted?
  > A: require the removal of all joints referencing that body first?
  > B: Implicitly delete the joint? But then what happens when user code has existing references?
- Capping constraint force/torque still yields strange results for dragging objects, if constraint error larger than some threshold should
  probably switch to pointwise constraint for mouse clicking

- Technical implementation of joints. Need to specify how to support at a minimum:
  - Axis constraints (0, 1, 2, 3 translational DoF and 0, 1, 3 rotational DOF)
  - Axis limits
  - Force/torque limits (do we just clamp the force OR do we perform another solve at the boundary..?)
  - Motors (can this be incorporated as a minimum torque limit? Should the motors have a target speed)
  - Speculative contacts
  - Restitution (have stuff bounce)
  - Springs (can this be done by introducing a generalized "constraint hardness")?


- In 3D, rotation is a quaternion. How to independently limit each axis?
  > Consider using the Jolt engine as a reference. It only keeps the "twist" from the swing/twist decomposition.


- Teardown supports:
  > Ball (0/3 DoF), hinge (0/1 DoF), cone (0/2 DoF), prismatic (1/0 DoF)
  > Doors that are 2 vx wide don't work very well - need either a 1 vx thick door or 1 vx thick border
    (so we may need to improve 1 vx thick texturing)

*/