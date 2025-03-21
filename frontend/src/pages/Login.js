import React, { useState } from 'react';
import axios from 'axios';
import { useNavigate } from 'react-router-dom';

function Login() {
  const [username, setUsername] = useState('');
  const [password, setPassword] = useState('');
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState('');
  const navigate = useNavigate();

  const handleSubmit = async (e) => {
    e.preventDefault();
    
    // Basic validation
    if (!username.trim() || !password.trim()) {
      setError('Username and password are required');
      return;
    }
    
    setIsLoading(true);
    setError('');
    
    try {
      const response = await axios.post('http://localhost:8000/login', {
        username,
        password,
      });
      
      const { access_token, is_admin } = response.data;
      
      // Save token
      localStorage.setItem('token', access_token);
      
      // Navigate to appropriate page (admin dashboard or chat)
      if (is_admin) {
        navigate('/admin');
      } else {
        navigate('/chat');
      }
    } catch (error) {
      console.error(error);
      setError(error.response?.data?.detail || 'Login failed. Please check your credentials.');
    } finally {
      setIsLoading(false);
    }
  };

  return (
    <div style={{ 
      margin: '2rem auto',
      maxWidth: '400px', 
      padding: '2rem',
      borderRadius: '5px',
      boxShadow: '0 2px 8px rgba(0, 0, 0, 0.1)',
      backgroundColor: 'white'
    }}>
      <h1 style={{ textAlign: 'center', marginBottom: '1.5rem' }}>Login</h1>
      
      {error && (
        <div style={{ 
          backgroundColor: '#f8d7da', 
          color: '#721c24', 
          padding: '0.75rem', 
          borderRadius: '5px',
          marginBottom: '1rem'
        }}>
          {error}
        </div>
      )}
      
      <form onSubmit={handleSubmit}>
        <div style={{ marginBottom: '1rem' }}>
          <label style={{ display: 'block', marginBottom: '0.5rem' }}>Username:</label>
          <input
            type="text"
            value={username}
            onChange={(e) => setUsername(e.target.value)}
            style={{ 
              width: '100%', 
              padding: '0.5rem',
              borderRadius: '3px',
              border: '1px solid #ccc'
            }}
            disabled={isLoading}
            required
          />
        </div>
        
        <div style={{ marginBottom: '1.5rem' }}>
          <label style={{ display: 'block', marginBottom: '0.5rem' }}>Password:</label>
          <input
            type="password"
            value={password}
            onChange={(e) => setPassword(e.target.value)}
            style={{ 
              width: '100%', 
              padding: '0.5rem',
              borderRadius: '3px',
              border: '1px solid #ccc'
            }}
            disabled={isLoading}
            required
          />
        </div>
        
        <button 
          type="submit" 
          style={{ 
            width: '100%',
            padding: '0.75rem',
            backgroundColor: '#007BFF',
            color: 'white',
            border: 'none',
            borderRadius: '3px',
            cursor: 'pointer',
            opacity: isLoading ? 0.7 : 1
          }}
          disabled={isLoading}
        >
          {isLoading ? 'Logging in...' : 'Login'}
        </button>
      </form>
      
      <p style={{ marginTop: '1.5rem', textAlign: 'center' }}>
        Don't have an account? <a href="/register" style={{ color: '#007BFF' }}>Register here</a>
      </p>
    </div>
  );
}

export default Login;
