// SPDX-License-Identifier: MIT OR Apache-2.0

#[cfg(feature = "backends")]
pub mod backends;
pub mod error;
pub mod hybrid;
pub mod projector;
pub mod tensor;
pub mod traits;
pub mod types;

pub use error::{HybridError, Result};
pub use hybrid::HybridNetwork;
pub use tensor::Tensor;
pub use traits::{GgufLayout, GgufLoader, NeuroModulators, SpikingNetwork, Transformer};
pub use types::{HybridConfig, HybridOutput, TransformerConfig};

#[cfg(feature = "backends")]
pub use backends::simple_snn::SimpleSnn;
#[cfg(feature = "backends")]
pub use backends::simple_transformer::SimpleTransformer;
