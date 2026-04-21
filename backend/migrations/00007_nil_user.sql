INSERT INTO "users" (
    user_id, 
    username, 
    email, 
    is_verified, 
    password_hash
) VALUES (
    '00000000-0000-0000-0000-000000000000', 
    'public', 
    'public@localhost', 
    true, 
    'SYSTEM_DEFAULT_NO_PASSWORD'
)
ON CONFLICT (user_id) DO NOTHING;