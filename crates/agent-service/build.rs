// Compile the agent.v1 proto contract pulled in via the
// `proto/agent-rs-proto` git submodule. The generated Rust code lands
// in OUT_DIR and is included by `src/pb.rs`.

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let proto_root = "../../proto/agent-rs-proto/proto";
    let proto_file = format!("{}/agent.proto", proto_root);

    tonic_build::configure()
        .build_server(true)
        .build_client(false) // server-only crate; clients (web) compile their own
        .compile_protos(&[&proto_file], &[proto_root])?;

    println!("cargo:rerun-if-changed={}", proto_file);
    Ok(())
}
