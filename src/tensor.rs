// SPDX-License-Identifier: MIT OR Apache-2.0

use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Tensor {
    data: Vec<f32>,
    shape: Vec<usize>,
}

impl Tensor {
    pub fn from_vec(data: Vec<f32>, shape: &[usize]) -> Self {
        Self {
            data,
            shape: shape.to_vec(),
        }
    }

    pub fn zeros(shape: &[usize]) -> Self {
        let len: usize = shape.iter().product();
        Self {
            data: vec![0.0; len],
            shape: shape.to_vec(),
        }
    }

    pub fn data(&self) -> &[f32] {
        &self.data
    }

    pub fn data_mut(&mut self) -> &mut [f32] {
        &mut self.data
    }

    pub fn shape(&self) -> &[usize] {
        &self.shape
    }

    pub fn ndim(&self) -> usize {
        self.shape.len()
    }

    pub fn numel(&self) -> usize {
        self.shape.iter().product()
    }
}
