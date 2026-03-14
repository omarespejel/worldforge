//! WorldForge Provider Adapters
//!
//! Concrete implementations of the `WorldModelProvider` trait for
//! various world foundation models, plus a mock provider for testing.
//!
//! # Providers
//!
//! - [`mock`] — Deterministic mock for testing and development
//! - [`cosmos`] — NVIDIA Cosmos (Predict, Transfer, Reason, Embed)
//! - [`runway`] — Runway GWM (Worlds, Robotics, Avatars)
//! - [`jepa`] — Meta JEPA (local inference, ZK-compatible)
//! - [`genie`] — Google Genie (research preview, stubbed)

pub mod cosmos;
pub mod genie;
pub mod jepa;
pub mod mock;
pub mod runway;

pub use cosmos::CosmosProvider;
pub use genie::GenieProvider;
pub use jepa::JepaProvider;
pub use mock::MockProvider;
pub use runway::RunwayProvider;
