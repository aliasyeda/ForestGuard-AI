# auth.py
"""
Authentication System for ForestGuard AI
User management, JWT tokens, role-based access control
"""

import hashlib
import json
import os
import jwt
from datetime import datetime, timedelta
from typing import Dict, Optional, List
import secrets
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class AuthManager:
    """Authentication and authorization manager"""
    
    def __init__(self, users_file="data/users.json", secret_key=None):
        self.users_file = users_file
        self.secret_key = secret_key or secrets.token_hex(32)
        self.users = self.load_users()
    
    def load_users(self) -> Dict:
        """Load users from file"""
        if os.path.exists(self.users_file):
            with open(self.users_file, 'r') as f:
                return json.load(f)
        return {}
    
    def save_users(self):
        """Save users to file"""
        os.makedirs(os.path.dirname(self.users_file), exist_ok=True)
        with open(self.users_file, 'w') as f:
            json.dump(self.users, f, indent=2)
    
    def hash_password(self, password: str) -> str:
        """Hash password using SHA-256"""
        return hashlib.sha256(password.encode()).hexdigest()
    
    def create_user(self, username: str, password: str, role: str = "user") -> bool:
        """Create a new user"""
        if username in self.users:
            return False
        
        self.users[username] = {
            'username': username,
            'password_hash': self.hash_password(password),
            'role': role,
            'created_at': datetime.now().isoformat(),
            'last_login': None,
            'is_active': True
        }
        
        self.save_users()
        logger.info(f"Created user: {username}")
        return True
    
    def authenticate(self, username: str, password: str) -> Optional[Dict]:
        """Authenticate user and return token"""
        user = self.users.get(username)
        
        if not user or not user.get('is_active', True):
            return None
        
        if user['password_hash'] != self.hash_password(password):
            return None
        
        # Update last login
        user['last_login'] = datetime.now().isoformat()
        self.save_users()
        
        # Generate JWT token
        token = self.generate_token(username, user['role'])
        
        return {
            'token': token,
            'username': username,
            'role': user['role'],
            'expires_in': 3600
        }
    
    def generate_token(self, username: str, role: str) -> str:
        """Generate JWT token"""
        payload = {
            'username': username,
            'role': role,
            'exp': datetime.utcnow() + timedelta(hours=1),
            'iat': datetime.utcnow()
        }
        
        token = jwt.encode(payload, self.secret_key, algorithm='HS256')
        return token
    
    def verify_token(self, token: str) -> Optional[Dict]:
        """Verify JWT token"""
        try:
            payload = jwt.decode(token, self.secret_key, algorithms=['HS256'])
            return payload
        except jwt.ExpiredSignatureError:
            logger.warning("Token expired")
            return None
        except jwt.InvalidTokenError:
            logger.warning("Invalid token")
            return None
    
    def change_password(self, username: str, old_password: str, new_password: str) -> bool:
        """Change user password"""
        user = self.users.get(username)
        
        if not user:
            return False
        
        if user['password_hash'] != self.hash_password(old_password):
            return False
        
        user['password_hash'] = self.hash_password(new_password)
        user['password_changed_at'] = datetime.now().isoformat()
        self.save_users()
        
        logger.info(f"Password changed for user: {username}")
        return True
    
    def delete_user(self, username: str) -> bool:
        """Delete a user"""
        if username not in self.users:
            return False
        
        del self.users[username]
        self.save_users()
        logger.info(f"Deleted user: {username}")
        return True
    
    def get_user_info(self, username: str) -> Optional[Dict]:
        """Get user information"""
        user = self.users.get(username)
        if user:
            # Remove sensitive data
            user_info = user.copy()
            user_info.pop('password_hash', None)
            return user_info
        return None
    
    def list_users(self) -> List[Dict]:
        """List all users (admin only)"""
        users_list = []
        for username, user in self.users.items():
            users_list.append({
                'username': username,
                'role': user['role'],
                'created_at': user['created_at'],
                'last_login': user['last_login'],
                'is_active': user.get('is_active', True)
            })
        return users_list
    
    def deactivate_user(self, username: str) -> bool:
        """Deactivate a user account"""
        user = self.users.get(username)
        if not user:
            return False
        
        user['is_active'] = False
        self.save_users()
        logger.info(f"Deactivated user: {username}")
        return True
    
    def activate_user(self, username: str) -> bool:
        """Activate a user account"""
        user = self.users.get(username)
        if not user:
            return False
        
        user['is_active'] = True
        self.save_users()
        logger.info(f"Activated user: {username}")
        return True

# Role-based access control
class RoleBasedAccess:
    """RBAC for API endpoints"""
    
    ROLES = {
        'admin': ['*'],  # All permissions
        'manager': ['predict', 'monitor', 'export'],
        'analyst': ['predict', 'monitor'],
        'user': ['predict'],
        'viewer': ['monitor']
    }
    
    def __init__(self, auth_manager: AuthManager):
        self.auth_manager = auth_manager
    
    def has_permission(self, token: str, action: str) -> bool:
        """Check if user has permission for action"""
        payload = self.auth_manager.verify_token(token)
        
        if not payload:
            return False
        
        role = payload.get('role', 'user')
        allowed_actions = self.ROLES.get(role, [])
        
        return action in allowed_actions or '*' in allowed_actions
    
    def require_auth(self, token: str):
        """Decorator-like function for API endpoints"""
        payload = self.auth_manager.verify_token(token)
        
        if not payload:
            raise PermissionError("Invalid or expired token")
        
        return payload

# Example usage
if __name__ == "__main__":
    # Initialize auth manager
    auth = AuthManager()
    
    # Create default admin user
    auth.create_user("admin", "admin123", "admin")
    auth.create_user("analyst", "analyst123", "analyst")
    
    # Authenticate
    result = auth.authenticate("admin", "admin123")
    if result:
        print("Authentication successful")
        print("Token:", result['token'])
        
        # Verify token
        payload = auth.verify_token(result['token'])
        print("Token payload:", payload)
    
    # Check permissions
    rbac = RoleBasedAccess(auth)
    token = result['token'] if result else None
    
    if token:
        has_permission = rbac.has_permission(token, "predict")
        print("Has predict permission:", has_permission)