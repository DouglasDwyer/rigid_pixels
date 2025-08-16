use macroquad::prelude::*;

/// Determines how collision detection will be performed.
#[derive(Copy, Clone, Debug)]
pub enum Detector {
    /// Performs collision detection once at the beginning of the frame.
    Naive,
    /// Dynamically splits the frame into smaller substeps to prevent tunneling.
    Substepper,
    /// Split *detection only* into multiple substeps based upon the unaffected motion of the objects.
    Speculative {
        /// Whether to include the effect of external forces in the stepped objects' trajectories.
        integrate_external_forces: bool,
        /// How to step.
        mode: SpeculativeStepMode,
    }
}

/// How to place the speculative steps.
#[derive(Copy, Clone, Debug)]
pub enum SpeculativeStepMode {
    /// Place the first speculative step at `t = start`.
    Floor,
    /// Space speculative steps so that they are equally apart.
    Equidistant,
    /// Place the last speculative step at `t = end`.
    Ceil
}