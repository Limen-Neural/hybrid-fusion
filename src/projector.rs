// SPDX-License-Identifier: MIT OR Apache-2.0

use crate::tensor::Tensor;

fn mean_pool(embedding: &Tensor) -> Vec<f32> {
    match embedding.ndim() {
        0 | 1 => embedding.data().to_vec(),
        2 => {
            let shape = embedding.shape();
            let seq = shape[0].max(1);
            let dim = shape[1];
            let data = embedding.data();
            let mut pooled = vec![0.0f32; dim];
            for t in 0..seq {
                let row = &data[t * dim..(t + 1) * dim];
                for (i, v) in row.iter().enumerate() {
                    pooled[i] += *v;
                }
            }
            let inv = 1.0 / seq as f32;
            for v in &mut pooled {
                *v *= inv;
            }
            pooled
        }
        _ => {
            // Treat higher-rank tensors as flat — no silent semantic change.
            embedding.data().to_vec()
        }
    }
}

fn resize_to(src: &[f32], target_width: usize) -> Vec<f32> {
    if target_width == 0 {
        return Vec::new();
    }
    let src_len = src.len();
    if src_len == 0 {
        return vec![0.0; target_width];
    }
    if src_len == target_width {
        return src.to_vec();
    }

    if src_len > target_width {
        let mut out = Vec::with_capacity(target_width);
        for i in 0..target_width {
            let start = (i * src_len) / target_width;
            let end = ((i + 1) * src_len) / target_width;
            let end = end.max(start + 1).min(src_len);
            let slice = &src[start..end];
            let mean = slice.iter().sum::<f32>() / slice.len() as f32;
            out.push(mean);
        }
        out
    } else {
        let mut out = Vec::with_capacity(target_width);
        out.extend_from_slice(src);
        out.resize(target_width, 0.0);
        out
    }
}

fn squash_inplace(v: &mut [f32]) {
    for x in v.iter_mut() {
        *x = x.tanh();
    }
}

pub fn embed_to_stimuli(embedding: &Tensor) -> Vec<f32> {
    let mut pooled = mean_pool(embedding);
    squash_inplace(&mut pooled);
    pooled
}

pub fn embed_to_stimuli_with_width(embedding: &Tensor, snn_width: usize) -> Vec<f32> {
    let pooled = mean_pool(embedding);
    let mut resized = resize_to(&pooled, snn_width);
    squash_inplace(&mut resized);
    resized
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_embed_to_stimuli_1d_bounded() {
        let t = Tensor::from_vec(vec![100.0, -100.0, 0.0, 50.0], &[4]);
        let out = embed_to_stimuli(&t);
        assert_eq!(out.len(), 4);
        for v in &out {
            assert!(v.abs() <= 1.0, "tanh must bound values in [-1, 1], got {v}");
        }
        assert!(out[0] > 0.99);
        assert!(out[1] < -0.99);
        assert!((out[2]).abs() < 1e-6);
    }

    #[test]
    fn test_embed_to_stimuli_2d_mean_pool() {
        let t = Tensor::from_vec(vec![1.0, 2.0, 3.0, 3.0, 4.0, 5.0], &[2, 3]);
        let out = embed_to_stimuli(&t);
        assert_eq!(out.len(), 3);
        assert!((out[0] - 2.0f32.tanh()).abs() < 1e-5);
        assert!((out[1] - 3.0f32.tanh()).abs() < 1e-5);
        assert!((out[2] - 4.0f32.tanh()).abs() < 1e-5);
    }

    #[test]
    fn test_embed_to_stimuli_with_width_downsamples() {
        let t = Tensor::from_vec(vec![0.0, 0.0, 1.0, 1.0, 2.0, 2.0, 3.0, 3.0], &[8]);
        let out = embed_to_stimuli_with_width(&t, 4);
        assert_eq!(out.len(), 4);
        assert!((out[0] - 0.0f32.tanh()).abs() < 1e-5);
        assert!((out[1] - 1.0f32.tanh()).abs() < 1e-5);
        assert!((out[2] - 2.0f32.tanh()).abs() < 1e-5);
        assert!((out[3] - 3.0f32.tanh()).abs() < 1e-5);
    }

    #[test]
    fn test_embed_to_stimuli_with_width_pads() {
        let t = Tensor::from_vec(vec![0.5, -0.5], &[2]);
        let out = embed_to_stimuli_with_width(&t, 5);
        assert_eq!(out.len(), 5);
        assert!((out[0] - 0.5f32.tanh()).abs() < 1e-5);
        assert!((out[1] - (-0.5f32).tanh()).abs() < 1e-5);
        for v in &out[2..] {
            assert!(v.abs() < 1e-6);
        }
    }

    #[test]
    fn test_embed_to_stimuli_with_width_strictly_bounded_after_resize() {
        let t = Tensor::from_vec(vec![500.0; 128], &[128]);
        let out = embed_to_stimuli_with_width(&t, 16);
        assert_eq!(out.len(), 16);
        for v in &out {
            assert!(v.abs() <= 1.0, "tanh bound violated: {v}");
            assert!(*v > 0.99);
        }
    }

    #[test]
    fn test_empty_width_returns_empty() {
        let t = Tensor::from_vec(vec![1.0, 2.0, 3.0], &[3]);
        let out = embed_to_stimuli_with_width(&t, 0);
        assert!(out.is_empty());
    }
}
