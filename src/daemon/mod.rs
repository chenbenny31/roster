use std::sync::Arc; // atomic reference counting for shared ownership across threads

/// Shared daemon state, accessed behind Arch<DaemonState>
pub struct DaemonState {}

impl DaemonState {
    pub fn new() -> Arc<Self> {
        Arc::new(Self {})
    }
}