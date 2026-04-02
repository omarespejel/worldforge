/// User roles.
#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub enum Role {
    Admin,
    Developer,
    Viewer,
}

/// System permissions.
#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub enum Permission {
    CreateWorld,
    ReadWorld,
    Predict,
    Plan,
    Evaluate,
    ManageProviders,
    ManageUsers,
}

/// Check whether a role has a given permission.
pub fn role_has_permission(role: Role, perm: Permission) -> bool {
    match role {
        Role::Admin => true, // Admin has all permissions
        Role::Developer => matches!(
            perm,
            Permission::CreateWorld
                | Permission::ReadWorld
                | Permission::Predict
                | Permission::Plan
                | Permission::Evaluate
        ),
        Role::Viewer => matches!(perm, Permission::ReadWorld),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_admin_has_all_permissions() {
        let perms = [
            Permission::CreateWorld,
            Permission::ReadWorld,
            Permission::Predict,
            Permission::Plan,
            Permission::Evaluate,
            Permission::ManageProviders,
            Permission::ManageUsers,
        ];
        for perm in perms {
            assert!(role_has_permission(Role::Admin, perm), "Admin should have {:?}", perm);
        }
    }

    #[test]
    fn test_developer_permissions() {
        assert!(role_has_permission(Role::Developer, Permission::CreateWorld));
        assert!(role_has_permission(Role::Developer, Permission::ReadWorld));
        assert!(role_has_permission(Role::Developer, Permission::Predict));
        assert!(role_has_permission(Role::Developer, Permission::Plan));
        assert!(role_has_permission(Role::Developer, Permission::Evaluate));
        assert!(!role_has_permission(Role::Developer, Permission::ManageProviders));
        assert!(!role_has_permission(Role::Developer, Permission::ManageUsers));
    }

    #[test]
    fn test_viewer_permissions() {
        assert!(role_has_permission(Role::Viewer, Permission::ReadWorld));
        assert!(!role_has_permission(Role::Viewer, Permission::CreateWorld));
        assert!(!role_has_permission(Role::Viewer, Permission::Predict));
        assert!(!role_has_permission(Role::Viewer, Permission::ManageUsers));
    }
}
