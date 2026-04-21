//! Re-exported Prost-generated types for the agent.v1 gRPC contract.
//!
//! The heavy lifting happens at build time in `build.rs`; here we
//! simply include the generated module and re-export the pieces this
//! crate's server actually uses.

#![allow(clippy::all)] // generated code

tonic::include_proto!("agent.v1");
