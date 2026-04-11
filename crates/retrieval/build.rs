fn main() -> Result<(), Box<dyn std::error::Error>> {
    tonic_prost_build::configure()
        .build_client(true)
        .build_server(false)
        .compile_protos(
            &["../../proto/agentic_rag/v1/chunker.proto"],
            &["../../proto"],
        )?;
    Ok(())
}
