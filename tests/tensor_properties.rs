use hybrid_fusion::projector::{embed_to_stimuli, embed_to_stimuli_with_width};
use hybrid_fusion::tensor::Tensor;
use proptest::prelude::*;

// ── Tensor shape invariants ──────────────────────────────────────────────────

proptest! {
    #[test]
    fn from_vec_rejects_zero_dimensions(shape in prop::collection::vec(0usize..10, 1..6)) {
        // Every generated shape has at least one dimension in 0..10, so if any
        // dim is 0 the construction must panic.
        if shape.iter().any(|&d| d == 0) {
            let result = std::panic::catch_unwind(|| {
                let len: usize = shape.iter().product();
                Tensor::from_vec(vec![0.0; len], &shape);
            });
            prop_assert!(result.is_err(), "from_vec should panic on zero-dim shape");
        }
    }

    #[test]
    fn data_len_equals_shape_product(
        dims in prop::collection::vec(1usize..8, 1..5)
    ) {
        let len: usize = dims.iter().product();
        let data: Vec<f32> = (0..len).map(|i| i as f32).collect();
        let t = Tensor::from_vec(data, &dims);
        prop_assert_eq!(t.data().len(), t.shape().iter().product::<usize>());
        prop_assert_eq!(t.ndim(), dims.len());
    }

    #[test]
    fn zeros_rejects_zero_dimensions(shape in prop::collection::vec(0usize..10, 1..6)) {
        if shape.iter().any(|&d| d == 0) {
            let result = std::panic::catch_unwind(|| {
                Tensor::zeros(&shape);
            });
            prop_assert!(result.is_err(), "zeros should panic on zero-dim shape");
        }
    }
}

// ── Projector invariants ─────────────────────────────────────────────────────

proptest! {
    #[test]
    fn embed_to_stimuli_output_bounded(
        data in prop::collection::vec(-1e6f32..1e6, 1..128)
    ) {
        let len = data.len();
        // Try ranks 0-D (scalar = len 1), 1-D, and 2-D
        // 1-D
        let t1 = Tensor::from_vec(data.clone(), &[len]);
        let out1 = embed_to_stimuli(&t1);
        for v in &out1 {
            prop_assert!(v.abs() <= 1.0 + 1e-6, "1-D bound violated: {}", v);
        }

        // 2-D (split into 2 rows if even length)
        if len >= 2 && len % 2 == 0 {
            let rows = 2;
            let cols = len / rows;
            let t2 = Tensor::from_vec(data.clone(), &[rows, cols]);
            let out2 = embed_to_stimuli(&t2);
            for v in &out2 {
                prop_assert!(v.abs() <= 1.0 + 1e-6, "2-D bound violated: {}", v);
            }
        }

        // 0-D scalar — requires exactly 1 element
        let t0 = Tensor::from_vec(vec![data[0]], &[]);
        let out0 = embed_to_stimuli(&t0);
        for v in &out0 {
            prop_assert!(v.abs() <= 1.0 + 1e-6, "0-D bound violated: {}", v);
        }
    }

    #[test]
    fn embed_to_stimuli_with_width_exact_length(
        data in prop::collection::vec(-1e4f32..1e4, 1..64),
        width in 0usize..128,
    ) {
        let len = data.len();
        let t = Tensor::from_vec(data, &[len]);
        let out = embed_to_stimuli_with_width(&t, width);
        prop_assert_eq!(out.len(), width);
        // Also verify bounds
        for v in &out {
            prop_assert!(v.abs() <= 1.0 + 1e-6, "bound violated: {}", v);
        }
    }

    #[test]
    fn pool_resize_2d_never_panics(
        data in prop::collection::vec(-1e4f32..1e4, 2..64),
        width in 0usize..128,
    ) {
        let len = data.len();
        // Force into 2-D: rows x cols where rows >= 2
        let rows = 2.min(len);
        let cols = len / rows;
        if cols == 0 { return Ok(()); }
        let usable = rows * cols;
        let t = Tensor::from_vec(data[..usable].to_vec(), &[rows, cols]);
        // Must not panic
        let out = embed_to_stimuli_with_width(&t, width);
        prop_assert_eq!(out.len(), width);
    }
}
