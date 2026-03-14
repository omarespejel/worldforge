//! World state management and orchestration.
//!
//! The `World` struct is the primary user-facing object for interacting
//! with a simulated world through one or more providers.

use tracing::instrument;

use crate::action::Action;
use crate::error::{Result, WorldForgeError};
use crate::guardrail::{evaluate_guardrails, has_blocking_violation};
use crate::prediction::{
    ComparisonReport, MultiPrediction, Prediction, PredictionConfig, ProviderScore,
};
use crate::provider::ProviderRegistry;
use crate::state::{HistoryEntry, PredictionSummary, WorldState};
use crate::types::SimTime;

/// A live world instance backed by one or more providers.
pub struct World {
    /// Current world state.
    pub state: WorldState,
    /// Default provider name for predictions.
    pub default_provider: String,
    /// Reference to the provider registry.
    registry: std::sync::Arc<ProviderRegistry>,
}

impl World {
    /// Create a new world with the given state and provider registry.
    pub fn new(
        state: WorldState,
        default_provider: impl Into<String>,
        registry: std::sync::Arc<ProviderRegistry>,
    ) -> Self {
        Self {
            state,
            default_provider: default_provider.into(),
            registry,
        }
    }

    /// Get the world's unique ID.
    pub fn id(&self) -> crate::types::WorldId {
        self.state.id
    }

    /// Predict the next world state after applying an action.
    #[instrument(skip(self, config))]
    pub async fn predict(
        &mut self,
        action: &Action,
        config: &PredictionConfig,
    ) -> Result<Prediction> {
        self.predict_with_provider(action, config, &self.default_provider.clone())
            .await
    }

    /// Predict using a specific provider.
    #[instrument(skip(self, config))]
    pub async fn predict_with_provider(
        &mut self,
        action: &Action,
        config: &PredictionConfig,
        provider_name: &str,
    ) -> Result<Prediction> {
        let provider = self.registry.get(provider_name)?;
        let prediction = provider.predict(&self.state, action, config).await?;

        // Evaluate guardrails on the predicted state
        if !config.guardrails.is_empty() {
            let results = evaluate_guardrails(&config.guardrails, &prediction.output_state);
            if has_blocking_violation(&results) {
                return Err(WorldForgeError::GuardrailBlocked {
                    reason: results
                        .iter()
                        .filter(|r| !r.passed)
                        .map(|r| {
                            format!(
                                "{}: {}",
                                r.guardrail_name,
                                r.violation_details.as_deref().unwrap_or("violation")
                            )
                        })
                        .collect::<Vec<_>>()
                        .join("; "),
                });
            }
        }

        // Update world state and history
        let hash = compute_state_hash(&prediction.output_state);
        self.state.history.push(HistoryEntry {
            time: self.state.time,
            state_hash: hash,
            action: Some(action.clone()),
            prediction: Some(PredictionSummary {
                confidence: prediction.confidence,
                physics_score: prediction.physics_scores.overall,
                latency_ms: prediction.latency_ms,
            }),
            provider: provider_name.to_string(),
        });

        self.state.time = SimTime {
            step: self.state.time.step + config.steps as u64,
            seconds: self.state.time.seconds + (config.steps as f64 / config.fps as f64),
            dt: 1.0 / config.fps as f64,
        };
        self.state.scene = prediction.output_state.scene.clone();

        Ok(prediction)
    }

    /// Run prediction with multiple providers and compare results.
    #[instrument(skip(self, config))]
    pub async fn predict_multi(
        &self,
        action: &Action,
        provider_names: &[&str],
        config: &PredictionConfig,
    ) -> Result<MultiPrediction> {
        let mut predictions = Vec::new();

        for &name in provider_names {
            let provider = self.registry.get(name)?;
            let pred = provider.predict(&self.state, action, config).await?;
            predictions.push(pred);
        }

        if predictions.is_empty() {
            return Err(WorldForgeError::InternalError(
                "no predictions generated".to_string(),
            ));
        }

        // Find best prediction by overall physics score
        let best_idx = predictions
            .iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| {
                a.physics_scores
                    .overall
                    .partial_cmp(&b.physics_scores.overall)
                    .unwrap_or(std::cmp::Ordering::Equal)
            })
            .map(|(i, _)| i)
            .unwrap_or(0);

        // Compute agreement score as average pairwise physics score similarity
        let agreement = if predictions.len() > 1 {
            let mut total = 0.0f32;
            let mut count = 0;
            for i in 0..predictions.len() {
                for j in (i + 1)..predictions.len() {
                    let diff = (predictions[i].physics_scores.overall
                        - predictions[j].physics_scores.overall)
                        .abs();
                    total += 1.0 - diff;
                    count += 1;
                }
            }
            if count > 0 {
                total / count as f32
            } else {
                1.0
            }
        } else {
            1.0
        };

        let scores = predictions
            .iter()
            .map(|p| ProviderScore {
                provider: p.provider.clone(),
                physics_scores: p.physics_scores,
                latency_ms: p.latency_ms,
                cost: p.cost.clone(),
            })
            .collect();

        Ok(MultiPrediction {
            agreement_score: agreement,
            best_prediction: best_idx,
            comparison: ComparisonReport {
                scores,
                summary: format!(
                    "Compared {} providers, best: {}",
                    predictions.len(),
                    predictions[best_idx].provider
                ),
            },
            predictions,
        })
    }

    /// Get the current world state.
    pub fn current_state(&self) -> &WorldState {
        &self.state
    }
}

/// Compute a simple hash of a world state for history tracking.
fn compute_state_hash(state: &WorldState) -> [u8; 32] {
    use std::collections::hash_map::DefaultHasher;
    use std::hash::{Hash, Hasher};

    let mut hasher = DefaultHasher::new();
    // Hash the world ID and time as a simple fingerprint
    state.id.hash(&mut hasher);
    state.time.step.hash(&mut hasher);
    state.scene.objects.len().hash(&mut hasher);
    let h = hasher.finish();

    let mut result = [0u8; 32];
    result[..8].copy_from_slice(&h.to_le_bytes());
    result
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_compute_state_hash() {
        let state = WorldState::new("test", "mock");
        let hash = compute_state_hash(&state);
        assert_ne!(hash, [0u8; 32]);
    }
}
