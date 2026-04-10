fn main() -> Result<(), Box<dyn std::error::Error>> {
    tonic_prost_build::configure()
        .build_server(true)
        .build_client(false)
        .compile_protos(
            &[
                "../../proto/agentic_rag/v1/types.proto",
                "../../proto/agentic_rag/v1/query.proto",
                "../../proto/agentic_rag/v1/health.proto",
            ],
            &["../../proto"],
        )?;
    Ok(())
}
