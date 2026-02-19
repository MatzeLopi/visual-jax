use serde::{Deserialize, Serialize};
use uuid::Uuid;

#[derive(Debug, Serialize, Deserialize)]
pub struct NewUser {
    pub username: String,
    pub password: String,
    pub email: String,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct UserLogin {
    pub username: String,
    pub password: String,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct User {
    pub id: Uuid,
    pub username: String,
    pub email: String,
    pub verified: bool,
}

impl User {
    pub fn default() -> Self {
        User {
            id: Uuid::nil(),
            username: String::from(""),
            email: String::from(""),
            verified: false,
        }
    }
}

#[derive(Debug, Serialize, Deserialize)]
pub struct UpdatePassword {
    pub old_password: String,
    pub new_password: String,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_user_default() {
        let user = User::default();
        assert_eq!(user.id, Uuid::nil());
        assert_eq!(user.username, "");
        assert_eq!(user.email, "");
        assert!(!user.verified);
    }
}
