"""
User Management and Authentication System for SDG Synergy Mapper v2
"""

import streamlit as st
import hashlib
import json
import sqlite3
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
import logging
from pathlib import Path
import secrets
import jwt
from functools import wraps

logger = logging.getLogger(__name__)

class UserManager:
    """User management system"""
    
    def __init__(self, db_path: str = "users.db"):
        self.db_path = db_path
        self.init_database()
        self.secret_key = self._get_or_create_secret_key()
    
    def init_database(self):
        """Initialize user database"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Create users table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS users (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                username TEXT UNIQUE NOT NULL,
                email TEXT UNIQUE NOT NULL,
                password_hash TEXT NOT NULL,
                role TEXT DEFAULT 'user',
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                last_login TIMESTAMP,
                is_active BOOLEAN DEFAULT 1,
                preferences TEXT
            )
        ''')
        
        # Create projects table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS projects (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id INTEGER,
                name TEXT NOT NULL,
                description TEXT,
                data_config TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (user_id) REFERENCES users (id)
            )
        ''')
        
        # Create sessions table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS sessions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id INTEGER,
                session_token TEXT UNIQUE NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                expires_at TIMESTAMP,
                FOREIGN KEY (user_id) REFERENCES users (id)
            )
        ''')
        
        conn.commit()
        conn.close()
    
    def _get_or_create_secret_key(self) -> str:
        """Get or create JWT secret key"""
        key_file = Path("secret.key")
        if key_file.exists():
            return key_file.read_text()
        else:
            key = secrets.token_hex(32)
            key_file.write_text(key)
            return key
    
    def hash_password(self, password: str) -> str:
        """Hash password using SHA-256"""
        return hashlib.sha256(password.encode()).hexdigest()
    
    def create_user(self, username: str, email: str, password: str, role: str = "user") -> bool:
        """Create new user"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            password_hash = self.hash_password(password)
            
            cursor.execute('''
                INSERT INTO users (username, email, password_hash, role)
                VALUES (?, ?, ?, ?)
            ''', (username, email, password_hash, role))
            
            conn.commit()
            conn.close()
            
            logger.info(f"User created: {username}")
            return True
            
        except sqlite3.IntegrityError:
            logger.error(f"User already exists: {username}")
            return False
        except Exception as e:
            logger.error(f"Error creating user: {e}")
            return False
    
    def authenticate_user(self, username: str, password: str) -> Optional[Dict]:
        """Authenticate user"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            password_hash = self.hash_password(password)
            
            cursor.execute('''
                SELECT id, username, email, role, is_active
                FROM users
                WHERE username = ? AND password_hash = ? AND is_active = 1
            ''', (username, password_hash))
            
            user = cursor.fetchone()
            
            if user:
                # Update last login
                cursor.execute('''
                    UPDATE users
                    SET last_login = CURRENT_TIMESTAMP
                    WHERE id = ?
                ''', (user[0],))
                
                conn.commit()
                
                user_data = {
                    "id": user[0],
                    "username": user[1],
                    "email": user[2],
                    "role": user[3],
                    "is_active": user[4]
                }
                
                logger.info(f"User authenticated: {username}")
                return user_data
            
            conn.close()
            return None
            
        except Exception as e:
            logger.error(f"Error authenticating user: {e}")
            return None
    
    def create_session(self, user_id: int) -> str:
        """Create user session"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            session_token = secrets.token_urlsafe(32)
            expires_at = datetime.now() + timedelta(hours=24)
            
            cursor.execute('''
                INSERT INTO sessions (user_id, session_token, expires_at)
                VALUES (?, ?, ?)
            ''', (user_id, session_token, expires_at))
            
            conn.commit()
            conn.close()
            
            return session_token
            
        except Exception as e:
            logger.error(f"Error creating session: {e}")
            return None
    
    def validate_session(self, session_token: str) -> Optional[Dict]:
        """Validate user session"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute('''
                SELECT s.user_id, u.username, u.email, u.role, s.expires_at
                FROM sessions s
                JOIN users u ON s.user_id = u.id
                WHERE s.session_token = ? AND s.expires_at > CURRENT_TIMESTAMP
            ''', (session_token,))
            
            session = cursor.fetchone()
            
            if session:
                user_data = {
                    "id": session[0],
                    "username": session[1],
                    "email": session[2],
                    "role": session[3],
                    "expires_at": session[4]
                }
                
                conn.close()
                return user_data
            
            conn.close()
            return None
            
        except Exception as e:
            logger.error(f"Error validating session: {e}")
            return None
    
    def logout_user(self, session_token: str) -> bool:
        """Logout user by invalidating session"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute('''
                DELETE FROM sessions
                WHERE session_token = ?
            ''', (session_token,))
            
            conn.commit()
            conn.close()
            
            return True
            
        except Exception as e:
            logger.error(f"Error logging out user: {e}")
            return False
    
    def get_user_projects(self, user_id: int) -> List[Dict]:
        """Get user's projects"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute('''
                SELECT id, name, description, data_config, created_at, updated_at
                FROM projects
                WHERE user_id = ?
                ORDER BY updated_at DESC
            ''', (user_id,))
            
            projects = cursor.fetchall()
            
            project_list = []
            for project in projects:
                project_list.append({
                    "id": project[0],
                    "name": project[1],
                    "description": project[2],
                    "data_config": json.loads(project[3]) if project[3] else {},
                    "created_at": project[4],
                    "updated_at": project[5]
                })
            
            conn.close()
            return project_list
            
        except Exception as e:
            logger.error(f"Error getting user projects: {e}")
            return []
    
    def create_project(self, user_id: int, name: str, description: str = "", 
                      data_config: Dict = None) -> bool:
        """Create new project"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute('''
                INSERT INTO projects (user_id, name, description, data_config)
                VALUES (?, ?, ?, ?)
            ''', (user_id, name, description, json.dumps(data_config or {})))
            
            conn.commit()
            conn.close()
            
            logger.info(f"Project created: {name}")
            return True
            
        except Exception as e:
            logger.error(f"Error creating project: {e}")
            return False
    
    def update_project(self, project_id: int, user_id: int, name: str = None,
                     description: str = None, data_config: Dict = None) -> bool:
        """Update project"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            updates = []
            params = []
            
            if name is not None:
                updates.append("name = ?")
                params.append(name)
            
            if description is not None:
                updates.append("description = ?")
                params.append(description)
            
            if data_config is not None:
                updates.append("data_config = ?")
                params.append(json.dumps(data_config))
            
            updates.append("updated_at = CURRENT_TIMESTAMP")
            params.extend([project_id, user_id])
            
            query = f'''
                UPDATE projects
                SET {', '.join(updates)}
                WHERE id = ? AND user_id = ?
            '''
            
            cursor.execute(query, params)
            
            conn.commit()
            conn.close()
            
            return True
            
        except Exception as e:
            logger.error(f"Error updating project: {e}")
            return False
    
    def delete_project(self, project_id: int, user_id: int) -> bool:
        """Delete project"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute('''
                DELETE FROM projects
                WHERE id = ? AND user_id = ?
            ''', (project_id, user_id))
            
            conn.commit()
            conn.close()
            
            return True
            
        except Exception as e:
            logger.error(f"Error deleting project: {e}")
            return False

class AuthenticationUI:
    """Authentication UI components"""
    
    def __init__(self, user_manager: UserManager):
        self.user_manager = user_manager
    
    def render_login_form(self) -> Optional[Dict]:
        """Render login form"""
        st.subheader("🔐 Login")
        
        with st.form("login_form"):
            username = st.text_input("Username")
            password = st.text_input("Password", type="password")
            submit_button = st.form_submit_button("Login")
            
            if submit_button:
                if username and password:
                    user = self.user_manager.authenticate_user(username, password)
                    if user:
                        st.success("✅ Login successful!")
                        return user
                    else:
                        st.error("❌ Invalid username or password")
                else:
                    st.error("❌ Please fill in all fields")
        
        return None
    
    def render_register_form(self) -> bool:
        """Render registration form"""
        st.subheader("📝 Register")
        
        with st.form("register_form"):
            username = st.text_input("Username")
            email = st.text_input("Email")
            password = st.text_input("Password", type="password")
            confirm_password = st.text_input("Confirm Password", type="password")
            submit_button = st.form_submit_button("Register")
            
            if submit_button:
                if username and email and password and confirm_password:
                    if password != confirm_password:
                        st.error("❌ Passwords do not match")
                    elif len(password) < 6:
                        st.error("❌ Password must be at least 6 characters")
                    else:
                        success = self.user_manager.create_user(username, email, password)
                        if success:
                            st.success("✅ Registration successful! Please login.")
                            return True
                        else:
                            st.error("❌ Username or email already exists")
                else:
                    st.error("❌ Please fill in all fields")
        
        return False
    
    def render_user_profile(self, user: Dict):
        """Render user profile"""
        st.subheader("👤 User Profile")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.write(f"**Username:** {user['username']}")
            st.write(f"**Email:** {user['email']}")
            st.write(f"**Role:** {user['role']}")
        
        with col2:
            if st.button("Logout"):
                st.session_state.clear()
                st.rerun()

class ProjectManager:
    """Project management system"""
    
    def __init__(self, user_manager: UserManager):
        self.user_manager = user_manager
    
    def render_project_list(self, user_id: int):
        """Render project list"""
        st.subheader("📁 My Projects")
        
        projects = self.user_manager.get_user_projects(user_id)
        
        if not projects:
            st.info("No projects found. Create your first project!")
            return
        
        for project in projects:
            with st.expander(f"📊 {project['name']}"):
                st.write(f"**Description:** {project['description']}")
                st.write(f"**Created:** {project['created_at']}")
                st.write(f"**Updated:** {project['updated_at']}")
                
                col1, col2, col3 = st.columns(3)
                with col1:
                    if st.button(f"Load {project['name']}", key=f"load_{project['id']}"):
                        st.session_state.current_project = project
                        st.success(f"Project '{project['name']}' loaded!")
                
                with col2:
                    if st.button(f"Edit {project['name']}", key=f"edit_{project['id']}"):
                        st.session_state.editing_project = project
                
                with col3:
                    if st.button(f"Delete {project['name']}", key=f"delete_{project['id']}"):
                        if self.user_manager.delete_project(project['id'], user_id):
                            st.success(f"Project '{project['name']}' deleted!")
                            st.rerun()
    
    def render_create_project_form(self, user_id: int):
        """Render create project form"""
        st.subheader("➕ Create New Project")
        
        with st.form("create_project_form"):
            name = st.text_input("Project Name")
            description = st.text_area("Description")
            
            # Data configuration
            st.write("**Data Configuration:**")
            countries = st.multiselect("Countries", ["USA", "China", "India", "Brazil", "Germany", "Japan"])
            indicators = st.multiselect("SDG Indicators", list(SDG_INDICATORS.keys()))
            start_year = st.number_input("Start Year", value=2015, min_value=2000, max_value=2030)
            end_year = st.number_input("End Year", value=2023, min_value=2000, max_value=2030)
            
            submit_button = st.form_submit_button("Create Project")
            
            if submit_button:
                if name:
                    data_config = {
                        "countries": countries,
                        "indicators": indicators,
                        "start_year": start_year,
                        "end_year": end_year
                    }
                    
                    success = self.user_manager.create_project(
                        user_id, name, description, data_config
                    )
                    
                    if success:
                        st.success(f"Project '{name}' created successfully!")
                        st.rerun()
                    else:
                        st.error("Error creating project")
                else:
                    st.error("Please enter a project name")

def require_auth(func):
    """Decorator to require authentication"""
    @wraps(func)
    def wrapper(*args, **kwargs):
        if 'user' not in st.session_state:
            st.error("Please login to access this feature")
            return None
        return func(*args, **kwargs)
    return wrapper

def require_role(required_role: str):
    """Decorator to require specific role"""
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            if 'user' not in st.session_state:
                st.error("Please login to access this feature")
                return None
            
            user_role = st.session_state.user.get('role', 'user')
            if user_role != required_role and user_role != 'admin':
                st.error(f"This feature requires {required_role} role")
                return None
            
            return func(*args, **kwargs)
        return wrapper
    return decorator

# Initialize user manager
user_manager = UserManager()
auth_ui = AuthenticationUI(user_manager)
project_manager = ProjectManager(user_manager)

