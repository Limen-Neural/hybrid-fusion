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
        let n = self.num_channels.min(stimuli.len());
        let mut fired = Vec::new();

        let dopamine_gain = 1.0 + modulators.dopamine * 0.5
            + modulators.aux_dopamine * 0.25;
        let cortisol_suppress = (1.0 - modulators.cortisol * 0.3).max(0.1);
        let ach_leak_scale = 1.0 + modulators.acetylcholine * 0.3;
        let effective_leak = (self.leak_factor * ach_leak_scale * modulators.tempo).min(1.0);

        for i in 0..n {
            // 1. Leak toward zero.
            self.membrane_potentials[i] *= 1.0 - effective_leak;

            // 2. Add stimulus with modulator scaling.
            self.membrane_potentials[i] += stimuli[i] * dopamine_gain * cortisol_suppress;

            // 3. Spike if above threshold.
            if self.membrane_potentials[i] >= self.threshold {
                fired.push(i);
                self.membrane_potentials[i] = 0.0;
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
        let fired = snn.step(&[1.5, 0.0, 0.0, 0.0], &NeuroModulators::default()).unwrap();
        assert_eq!(fired, vec![0usize]);
    }

    #[test]
    fn no_fire_below_threshold() {
        let mut snn = SimpleSnn::with_params(4, 1.0, 0.0);
        let fired = snn.step(&[0.5, 0.0, 0.0, 0.0], &NeuroModulators::default()).unwrap();
        assert!(fired.is_empty());
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
