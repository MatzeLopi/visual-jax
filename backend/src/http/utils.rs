use std::sync::Arc;

use crate::http::{AppState, error::Error as HTTPError};
use anyhow::Error;
use mail_send::mail_builder::MessageBuilder;
use rand::{Rng, distributions::Alphanumeric};

const VERIFICATION_TEMPLATE: &str = r#"
<!DOCTYPE html>
<html>
<head>
    <meta charset='UTF-8'>
    <meta name='viewport' content='width=device-width, initial-scale=1.0'>
    <title>Email Verification</title>
    <style>
        body {
            font-family: Arial, sans-serif;
            background-color: #f4f4f4;
            padding: 20px;
        }
        .container {
            max-width: 600px;
            margin: 0 auto;
            background: #ffffff;
            padding: 20px;
            border-radius: 8px;
            box-shadow: 0 0 10px rgba(0, 0, 0, 0.1);
            text-align: center;
        }
        .button {
            display: inline-block;
            background-color: #007bff;
            color: #ffffff;
            padding: 10px 20px;
            text-decoration: none;
            border-radius: 5px;
            font-weight: bold;
        }
    </style>
</head>
<body>
    <div class='container'>
        <h2>Email Verification</h2>
        <p>Click the button below to verify your email address:</p>
        <a href='{{verification_link}}' class='button'>Verify Email</a>
        <p>If you did not request this, you can safely ignore this email.</p>
    </div>
</body>
</html>
"#;

pub fn random_string(length: usize) -> String {
    rand::thread_rng()
        .sample_iter(&Alphanumeric)
        .take(length)
        .map(char::from)
        .collect()
}

pub async fn send_verification(
    to: String,
    username: String,
    token: String,
    state: Arc<AppState>,
) -> Result<(), HTTPError> {
    let verification_link = format!("http://localhost:8080/auth-user/{}/{}", username, token);
    let body = VERIFICATION_TEMPLATE.replace("{{verification_link}}", &verification_link);

    send_mail(&to, "Email Verification", &body, &state).await?;

    Ok(())
}

pub async fn send_mail(to: &str, subject: &str, html: &str, state: &AppState) -> Result<(), Error> {
    // send mail
    let mut smtp_client = state.smtp_pool.get().await?;

    let message = MessageBuilder::new()
        .from((
            state.config.mail_sender.to_string(),
            state.config.mail_from.to_string(),
        ))
        .to(to)
        .subject(subject)
        .html_body(html);

    smtp_client.send(message).await?;

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_random_string_length() {
        let len = 10;
        let s = random_string(len);
        assert_eq!(s.len(), len);
    }

    #[test]
    fn test_random_string_randomness() {
        let s1 = random_string(10);
        let s2 = random_string(10);
        assert_ne!(s1, s2);
    }
}
