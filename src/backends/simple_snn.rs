// SPDX-License-Identifier: MIT OR Apache-2.0

//! Minimal spiking neural network with Leaky Integrate-and-Fire dynamics.
//!
//! `SimpleSnn` maintains a membrane potential per channel. Each step:
//! 1. Leak the membrane toward zero by `leak_factor`.
//! 2. Add stimulus, scaled by dopamine (excitatory gain) and dampened by
//!    cortisol (inhibitory gain).
//! 3. If potential exceeds `threshold`, record the neuron as fired and reset
//!    its potential.

use crate::error::Result;
use crate::traits::{NeuroModulators, SpikingNetwork};

/// Leaky Integrate-and-Fire spiking network.
pub struct SimpleSnn {
    membrane_potentials: Vec<f32>,
    threshold: f32,
    leak_factor: f32,
    num_channels: usize,
}

impl SimpleSnn {
    /// Create a new SNN with `num_channels` neurons and sensible defaults
    /// (threshold = 1.0, leak = 0.2).
    pub fn new(num_channels: usize) -> Self {
        Self::with_params(num_channels, 1.0, 0.2)
    }

    /// Create with explicit threshold and leak factor.
    pub fn with_params(num_channels: usize, threshold: f32, leak_factor: f32) -> Self {
        Self {
            membrane_potentials: vec![0.0; num_channels],
            threshold,
            leak_factor: leak_factor.clamp(0.0, 1.0),
            num_channels,
        }
    }

    /// Reset all membrane potentials to zero.
    pub fn reset(&mut self) {
        self.membrane_potentials.fill(0.0);
    }
}

impl SpikingNetwork for SimpleSnn {
    fn step(&mut self, stimuli: &[f32], modulators: &NeuroModulators) -> Result<Vec<usize>> {
        // Apply leak to every channel; only channels with a corresponding
        // stimulus receive new input (short stimuli never leave stale membrane
        // state without decay).
        let mut fired = Vec::new();

        let dopamine_gain = 1.0 + modulators.dopamine * 0.5 + modulators.aux_dopamine * 0.25;
        let cortisol_suppress = (1.0 - modulators.cortisol * 0.3).max(0.1);
        let ach_leak_scale = 1.0 + modulators.acetylcholine * 0.3;
        // Clamp tempo to non-negative so leak decays (never amplifies) membrane.
        let tempo = modulators.tempo.max(0.0);
        let effective_leak = (self.leak_factor * ach_leak_scale * tempo).clamp(0.0, 1.0);

        for (i, pot) in self.membrane_potentials.iter_mut().enumerate() {
            // 1. Leak toward zero.
            *pot *= 1.0 - effective_leak;

            // 2. Add stimulus with modulator scaling (only where provided).
            if let Some(&stim) = stimuli.get(i) {
                *pot += stim * dopamine_gain * cortisol_suppress;
            }

            // 3. Spike if above threshold.
            if *pot >= self.threshold {
                fired.push(i);
                *pot = 0.0;
            }
        }

        Ok(fired)
    }

    fn num_channels(&self) -> usize {
        self.num_channels
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn fires_when_above_threshold() {
        let mut snn = SimpleSnn::with_params(4, 1.0, 0.0);
        let fired = snn
            .step(&[1.5, 0.0, 0.0, 0.0], &NeuroModulators::default())
            .unwrap();
        assert_eq!(fired, vec![0usize]);
    }

    #[test]
    fn no_fire_below_threshold() {
        let mut snn = SimpleSnn::with_params(4, 1.0, 0.0);
        let fired = snn
            .step(&[0.5, 0.0, 0.0, 0.0], &NeuroModulators::default())
            .unwrap();
        assert!(fired.is_empty());
    }

    #[test]
    fn negative_tempo_does_not_amplify_membrane() {
        let mut snn = SimpleSnn::with_params(1, 10.0, 0.5);
        let mods = NeuroModulators {
            tempo: -2.0,
            ..Default::default()
        };
        snn.step(&[0.4], &mods).unwrap();
        // After leak with clamped tempo=0, potential stays at stimulus (no amp).
        // A second step with zero stimulus and negative tempo must not grow it.
        snn.step(&[0.0], &mods).unwrap();
        let fired = snn.step(&[0.0], &mods).unwrap();
        assert!(
            fired.is_empty(),
            "membrane must not amplify under neg tempo"
        );
    }

    #[test]
    fn leak_decays_potential() {
        let mut snn = SimpleSnn::with_params(2, 1.0, 1.0);
        snn.step(&[0.8, 0.0], &NeuroModulators::default()).unwrap();
        let fired = snn.step(&[0.0, 0.0], &NeuroModulators::default()).unwrap();
        assert!(fired.is_empty());
    }

    #[test]
    fn dopamine_boosts_excitation() {
        let mut snn = SimpleSnn::with_params(1, 1.0, 0.0);
        let mods = NeuroModulators {
            dopamine: 2.0,
            ..Default::default()
        };
        let fired = snn.step(&[0.8], &mods).unwrap();
        assert_eq!(fired, vec![0usize]);
    }

    #[test]
    fn reset_clears_potentials() {
        let mut snn = SimpleSnn::new(4);
        snn.step(&[0.5; 4], &NeuroModulators::default()).unwrap();
        snn.reset();
        let fired = snn.step(&[0.5; 4], &NeuroModulators::default()).unwrap();
        assert!(fired.is_empty());
    }

    #[test]
    fn accumulates_until_fire() {
        let mut snn = SimpleSnn::with_params(1, 1.0, 0.0);
        let neutral = NeuroModulators {
            dopamine: 0.0,
            cortisol: 0.0,
            acetylcholine: 0.0,
            tempo: 1.0,
            aux_dopamine: 0.0,
        };
        snn.step(&[0.4], &neutral).unwrap();
        snn.step(&[0.4], &neutral).unwrap();
        let fired = snn.step(&[0.3], &neutral).unwrap();
        // 0.4 + 0.4 + 0.3 = 1.1 >= 1.0
        assert_eq!(fired, vec![0usize]);
    }
}
