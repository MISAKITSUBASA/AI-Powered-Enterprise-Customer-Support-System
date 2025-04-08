import React, { useState } from 'react';
import axios from 'axios';
import { useNavigate, Link } from 'react-router-dom';

function Register() {
  const [username, setUsername] = useState('');
  const [password, setPassword] = useState('');
  const [email, setEmail] = useState('');
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
      const response = await axios.post('/register', {
        username,
        password,
        email: email.trim() || undefined, // Only send if not empty
      });
      
      // Check if this user is the first (admin)
      if (response.data.is_admin) {
        alert('Registration successful. You have been assigned as the system administrator.');
      } else {
        alert('Registration successful. Please login.');
      }
      
      // Navigate to login page
      navigate('/');
    } catch (error) {
      console.error(error);
      setError(error.response?.data?.detail || 'Registration failed. Please try again.');
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
          <h1 style={{ fontSize: '2rem', marginBottom: '0.5rem' }}>Create Account</h1>
          <p style={{ color: '#6c757d' }}>Register for the AI support system</p>
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
            <label style={{ display: 'block', marginBottom: '0.5rem', fontWeight: '500' }}>Email (optional)</label>
            <input
              type="email"
              value={email}
              onChange={(e) => setEmail(e.target.value)}
              style={{ 
                width: '100%',
                padding: '0.75rem',
                fontSize: '1rem',
                borderRadius: '8px',
                border: '1px solid #dee2e6'
              }}
              disabled={isLoading}
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
            {isLoading ? 'Registering...' : 'Register'}
          </button>
        </form>
        
        <div style={{ marginTop: '2rem', textAlign: 'center' }}>
          <p>
            Already have an account? <Link to="/" style={{ fontWeight: '500', color: '#3f51b5' }}>Login here</Link>
          </p>
        </div>
      </div>
    </div>
  );
}

export default Register;
