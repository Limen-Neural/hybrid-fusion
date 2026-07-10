// SPDX-License-Identifier: MIT OR Apache-2.0

//! Optional Sentry reporting helpers.
//!
//! When the `sentry` feature is enabled, errors returned from the public
//! orchestration path are captured on the global Sentry hub. Downstream
//! applications must still call [`crate::sentry::init`] (re-exported) once at
//! startup and hold the returned guard for the process lifetime.

use crate::error::HybridError;

/// Capture a [`HybridError`] on the active Sentry hub (no-op without feature).
#[inline]
pub fn capture_error(err: &HybridError) {
    #[cfg(feature = "sentry")]
    {
        sentry::capture_error(err as &dyn std::error::Error);
    }
    #[cfg(not(feature = "sentry"))]
    {
        let _ = err;
    }
}

/// Report `err` to Sentry (when enabled) and return it unchanged.
///
/// Intended for `return` sites:
/// ```ignore
/// return report(HybridError::InvalidConfig("…".into()));
/// ```
#[inline]
pub fn report<T>(err: HybridError) -> Result<T, HybridError> {
    capture_error(&err);
    Err(err)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn report_preserves_error_variant() {
        let err = HybridError::InvalidConfig("test".into());
        let returned = report::<()>(err).expect_err("should be err");
        match returned {
            HybridError::InvalidConfig(msg) => assert_eq!(msg, "test"),
            other => panic!("unexpected variant: {other:?}"),
        }
    }

    #[test]
    fn capture_error_is_callable_without_hub() {
        // Must not panic when no Sentry client is initialised.
        capture_error(&HybridError::SnnStep("noop".into()));
    }
}
