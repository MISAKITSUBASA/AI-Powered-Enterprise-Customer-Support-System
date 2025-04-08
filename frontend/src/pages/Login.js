import React, { useState } from 'react';
import axios from 'axios';
import { useNavigate, Link } from 'react-router-dom';
import { useAuth } from '../context/AuthContext';

function Login() {
  const [username, setUsername] = useState('');
  const [password, setPassword] = useState('');
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState('');
  const navigate = useNavigate();
  const { login } = useAuth();

  // Define API base URL - use relative path for Docker setup
  const API_BASE_URL = '';  // Empty for relative paths with Nginx proxy

  const handleSubmit = async (e) => {
    e.preventDefault();
    console.log("Login form submitted");
    
    // Basic validation
    if (!username.trim() || !password.trim()) {
      setError('Username and password are required');
      return;
    }
    
    setIsLoading(true);
    setError('');
    
    try {
      console.log(`Attempting to login with username: ${username}`);
      const response = await axios.post(`${API_BASE_URL}/login`, {
        username,
        password,
      });
      
      console.log("Login successful:", response.data);
      const { access_token, is_admin } = response.data;
      
      // Use the login function from AuthContext
      login(access_token);
      
      // Navigate to appropriate page (admin dashboard or chat)
      if (is_admin) {
        navigate('/admin');
      } else {
        navigate('/chat');
      }
    } catch (error) {
      console.error("Login error:", error);
      setError(
        error.response?.data?.detail || 
        error.message || 
        'Login failed. Please check your credentials.'
      );
    } finally {
      setIsLoading(false);
    }
  };

  return (
    <div className="auth-container" style={{ 
      minHeight: '100vh',
      display: 'flex',
      alignItems: 'center',
      justifyContent: 'center',
      backgroundImage: 'url("https://images.unsplash.com/photo-1557683316-973673baf926?auto=format&fit=crop&w=1920&q=80")',
      backgroundSize: 'cover',
      backgroundPosition: 'center',
    }}>
      <div className="auth-card" style={{ 
        width: '100%',
        maxWidth: '450px', 
        padding: '2.5rem',
        borderRadius: '8px',
        boxShadow: '0 10px 25px rgba(0, 0, 0, 0.15)',
        backgroundColor: 'rgba(255, 255, 255, 0.95)',
      }}>
        <div className="text-center" style={{ textAlign: 'center', marginBottom: '2rem' }}>
          <img 
            src="https://cdn-icons-png.flaticon.com/512/4233/4233830.png" 
            alt="AI Support Logo" 
            style={{ width: '80px', marginBottom: '1rem' }}
          />
          <h1 style={{ fontSize: '2rem', marginBottom: '0.5rem' }}>Welcome Back</h1>
          <p style={{ color: '#6c757d' }}>Log in to your AI support account</p>
        </div>
        
        {error && (
          <div style={{ 
            backgroundColor: '#ffebee', 
            color: '#c62828', 
            padding: '0.75rem 1rem',
            borderRadius: '8px',
            marginBottom: '1.5rem',
            borderLeft: '4px solid #f44336'
          }}>
            {error}
          </div>
        )}
        
        <form onSubmit={handleSubmit}>
          <div style={{ marginBottom: '1.5rem' }}>
            <label style={{ display: 'block', marginBottom: '0.5rem', fontWeight: '500' }}>Username</label>
            <input
              type="text"
              value={username}
              onChange={(e) => setUsername(e.target.value)}
              style={{ 
                width: '100%',
                padding: '0.75rem',
                fontSize: '1rem',
                borderRadius: '8px',
                border: '1px solid #dee2e6'
              }}
              disabled={isLoading}
              required
            />
          </div>
          
          <div style={{ marginBottom: '1.5rem' }}>
            <label style={{ display: 'block', marginBottom: '0.5rem', fontWeight: '500' }}>Password</label>
            <input
              type="password"
              value={password}
              onChange={(e) => setPassword(e.target.value)}
              style={{ 
                width: '100%',
                padding: '0.75rem',
                fontSize: '1rem',
                borderRadius: '8px',
                border: '1px solid #dee2e6'
              }}
              disabled={isLoading}
              required
            />
          </div>
          
          <button 
            type="submit" 
            style={{ 
              width: '100%',
              padding: '0.85rem',
              fontSize: '1rem',
              marginTop: '1rem',
              backgroundColor: '#3f51b5',
              color: 'white',
              border: 'none',
              borderRadius: '8px',
              cursor: 'pointer',
              opacity: isLoading ? 0.7 : 1
            }}
            disabled={isLoading}
          >
            {isLoading ? 'Logging in...' : 'Login'}
          </button>
        </form>
        
        <div style={{ marginTop: '2rem', textAlign: 'center' }}>
          <p>
            Don't have an account? <Link to="/register" style={{ fontWeight: '500', color: '#3f51b5' }}>Create Account</Link>
          </p>
        </div>
      </div>
    </div>
  );
}

export default Login;
