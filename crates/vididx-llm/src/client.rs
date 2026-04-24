use reqwest::Client;
use serde_json::json;
use std::sync::Arc;
use std::time::Duration;
use tokio::sync::Semaphore;
use vididx_core::VididxError;

/// Anthropic API client with JSON mode, retries, and concurrency control.
pub struct AnthropicClient {
    api_key: String,
    model: String,
    #[allow(dead_code)]
    max_concurrency: usize,
    http_client: Client,
    semaphore: Arc<Semaphore>,
}

impl AnthropicClient {
    /// Create a new Anthropic client.
    pub fn new(api_key: &str, model: &str, max_concurrency: usize) -> Self {
        Self {
            api_key: api_key.to_string(),
            model: model.to_string(),
            max_concurrency,
            http_client: Client::new(),
            semaphore: Arc::new(Semaphore::new(max_concurrency)),
        }
    }

    /// Call the Anthropic API with JSON mode.
    /// Only retries on rate limit (429), server errors (5xx), and timeouts.
    /// Does not retry client errors (4xx except 429).
    pub async fn call_with_json_mode(
        &self,
        system_prompt: &str,
        user_message: &str,
    ) -> Result<serde_json::Value, VididxError> {
        // Acquire concurrency permit
        let _permit = self
            .semaphore
            .acquire()
            .await
            .map_err(|_| VididxError::Llm("Failed to acquire semaphore".to_string()))?;

        // Retry loop with exponential backoff (max 3 attempts)
        let mut attempt = 0;
        let max_attempts = 3;
        let mut delay = Duration::from_millis(100);

        loop {
            attempt += 1;
            match self.call_api_internal(system_prompt, user_message).await {
                Ok(response) => return Ok(response),
                Err(e) => {
                    if !e.is_retryable() || attempt >= max_attempts {
                        return Err(e);
                    }
                    tokio::time::sleep(delay).await;
                    delay = Duration::from_millis((delay.as_millis() as u64 * 2).min(10000));
                }
            }
        }
    }

    async fn call_api_internal(
        &self,
        system_prompt: &str,
        user_message: &str,
    ) -> Result<serde_json::Value, VididxError> {
        let url = "https://api.anthropic.com/v1/messages";

        let request_body = json!({
            "model": self.model,
            "max_tokens": 4096,
            "system": system_prompt,
            "messages": [
                {
                    "role": "user",
                    "content": user_message
                }
            ]
        });

        let response = tokio::time::timeout(
            Duration::from_secs(30),
            self.http_client
                .post(url)
                .header("x-api-key", &self.api_key)
                .header("anthropic-version", "2023-06-01")
                .json(&request_body)
                .send(),
        )
        .await
        .map_err(|_| VididxError::Llm("Request timeout (30s)".to_string()))?
        .map_err(|e| VididxError::Llm(format!("HTTP request failed: {}", e)))?;

        if !response.status().is_success() {
            let status = response.status();
            let body = response.text().await.unwrap_or_default();
            return Err(VididxError::Llm(format!(
                "API error ({}): {}",
                status, body
            )));
        }

        let body = response
            .json::<serde_json::Value>()
            .await
            .map_err(|e| VididxError::Llm(format!("Failed to parse response: {}", e)))?;

        // Extract the text content from the response
        let text = body
            .get("content")
            .and_then(|c| c.get(0))
            .and_then(|c| c.get("text"))
            .and_then(|t| t.as_str())
            .ok_or_else(|| VididxError::Llm("No text in response".to_string()))?;

        // Parse JSON from response text, stripping markdown code fences if present
        let stripped = strip_code_fences(text);
        serde_json::from_str(&stripped)
            .map_err(|e| VididxError::Llm(format!("Failed to parse JSON from response: {}", e)))
    }
}

/// Strip markdown code fences (```json ... ```) from LLM response text.
fn strip_code_fences(text: &str) -> String {
    let trimmed = text.trim();
    if trimmed.starts_with("```") && trimmed.ends_with("```") {
        let without_fences = trimmed
            .strip_prefix("```")
            .unwrap()
            .strip_suffix("```")
            .unwrap()
            .trim();
        // Remove optional language identifier (e.g., "json\n")
        if let Some(newline_pos) = without_fences.find('\n') {
            let maybe_lang = &without_fences[..newline_pos];
            if maybe_lang
                .chars()
                .all(|c| c.is_alphanumeric() || c == '_' || c == '-')
            {
                without_fences[newline_pos + 1..].to_string()
            } else {
                without_fences.to_string()
            }
        } else {
            without_fences.to_string()
        }
    } else {
        trimmed.to_string()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_anthropic_client_new() {
        let client = AnthropicClient::new("test_key", "claude-opus", 5);
        assert_eq!(client.api_key, "test_key");
        assert_eq!(client.model, "claude-opus");
        assert_eq!(client.max_concurrency, 5);
    }

    #[tokio::test]
    async fn test_call_with_json_mode_timeout() {
        let client = AnthropicClient::new("invalid_key", "claude-opus", 1);
        let result = client
            .call_with_json_mode("Respond with JSON", "Return {\"test\": true}")
            .await;
        // Should fail with timeout or network error
        assert!(result.is_err());
    }

    #[test]
    fn test_semaphore_concurrency() {
        let sem = Arc::new(Semaphore::new(2));
        assert_eq!(sem.available_permits(), 2);
    }
}
