use uuid::Uuid;

pub fn generate_chat_cmpl_id() -> String {
    format!("chatcmpl-{}", Uuid::new_v4().as_simple())
}

pub fn unix_timestamp() -> u32 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap_or_default()
        .as_secs() as u32
}

#[cfg(feature = "backend-bedrock")]
pub mod bedrock;

#[cfg(feature = "backend-vertex")]
pub mod vertex;
