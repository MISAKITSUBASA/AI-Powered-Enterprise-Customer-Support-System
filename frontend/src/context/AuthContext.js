import React, { createContext, useState, useEffect, useContext } from 'react';
import { useNavigate } from 'react-router-dom';

// Create the context
const AuthContext = createContext();

// Create a custom hook to use the auth context
export const useAuth = () => {
  return useContext(AuthContext);
};

// Auth provider component
export const AuthProvider = ({ children }) => {
  const [isAuthenticated, setIsAuthenticated] = useState(false);
  const [isAdmin, setIsAdmin] = useState(false);
  const [isLoading, setIsLoading] = useState(true);
  const navigate = useNavigate();

  // Check if user is authenticated on component mount
  useEffect(() => {
    const checkAuth = () => {
      try {
        const token = localStorage.getItem('token');
        
        if (!token) {
          setIsAuthenticated(false);
          setIsAdmin(false);
          setIsLoading(false);
          return;
        }

        // Decode JWT to check expiration and role
        const base64Url = token.split('.')[1];
        const base64 = base64Url.replace(/-/g, '+').replace(/_/g, '/');
        const payload = JSON.parse(window.atob(base64));
        
        // Check if token is expired
        if (payload.exp * 1000 < Date.now()) {
          // Token expired
          handleLogout();
          return;
        }
        
        setIsAuthenticated(true);
        setIsAdmin(payload.is_admin || false);
        setIsLoading(false);
      } catch (error) {
        console.error('Error checking authentication:', error);
        setIsAuthenticated(false);
        setIsAdmin(false);
        setIsLoading(false);
      }
    };
    
    checkAuth();
  }, []); // eslint-disable-line react-hooks/exhaustive-deps

  // Logout function (renamed to handleLogout to avoid confusion)
  const handleLogout = () => {
    localStorage.removeItem('token');
    sessionStorage.removeItem('user');
    setIsAuthenticated(false);
    setIsAdmin(false);
    setIsLoading(false);
    if (navigate) {
      navigate('/');
    }
  };

  // Login function
  const login = (token) => {
    try {
      localStorage.setItem('token', token);
      
      // Decode JWT to check role
      const base64Url = token.split('.')[1];
      const base64 = base64Url.replace(/-/g, '+').replace(/_/g, '/');
      const payload = JSON.parse(window.atob(base64));
      
      setIsAuthenticated(true);
      setIsAdmin(payload.is_admin || false);
    } catch (error) {
      console.error('Error logging in:', error);
    }
  };

  // Context value
  const value = {
    isAuthenticated,
    isAdmin,
    isLoading,
    login,
    logout: handleLogout
  };

  return (
    <AuthContext.Provider value={value}>
      {children}
    </AuthContext.Provider>
  );
};
