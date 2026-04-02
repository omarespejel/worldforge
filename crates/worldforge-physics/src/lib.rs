//! Lightweight physics simulation engine for WorldForge.
//!
//! Provides Euler-integration-based rigid body simulation, spatial queries,
//! and physics validation utilities.

pub mod spatial;
pub mod validation;
pub mod world;

pub use spatial::{RayHit, SpatialQuery};
pub use validation::{ValidationResult, validate_collision, validate_energy, validate_gravity};
pub use world::{PhysicsObject, PhysicsState, PhysicsWorld};
