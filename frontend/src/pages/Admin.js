import React, { useState, useEffect } from 'react';
import axios from 'axios';
import { useNavigate } from 'react-router-dom';
import { useAuth } from '../context/AuthContext'; // Add this import

function Admin() {
  const [loading, setLoading] = useState(true);
  const [analytics, setAnalytics] = useState(null);
  const [users, setUsers] = useState([]);
  const [activeTab, setActiveTab] = useState('analytics');
  const [uploadFile, setUploadFile] = useState(null);
  const [uploadStatus, setUploadStatus] = useState('');
  const [searchQuery, setSearchQuery] = useState('');
  const [searchResults, setSearchResults] = useState([]);
  const [isSearching, setIsSearching] = useState(false);
  const navigate = useNavigate();
  const { logout } = useAuth(); // Add this line to get the logout function

  // Define API base URL - use relative path for Docker setup
  const API_BASE_URL = '';  // Empty for relative paths with Nginx proxy

  useEffect(() => {
    const token = localStorage.getItem('token');
    if (!token) {
      navigate('/');
      return;
    }

    try {
      const base64Url = token.split('.')[1];
      const base64 = base64Url.replace(/-/g, '+').replace(/_/g, '/');
      const payload = JSON.parse(window.atob(base64));
      
      if (!payload.is_admin) {
        alert('Admin access required');
        navigate('/chat');
      }
    } catch (error) {
      console.error('Error parsing token:', error);
      navigate('/');
    }

    fetchAnalytics();
    fetchUsers();
  }, [navigate]);

  const fetchAnalytics = async () => {
    try {
      setLoading(true);
      const token = localStorage.getItem('token');
      console.log("Fetching analytics data...");
      const response = await axios.get(`${API_BASE_URL}/admin/analytics`, {
        headers: {
          Authorization: `Bearer ${token}`,
        },
      });
      console.log("Analytics data received:", response.data);
      setAnalytics(response.data);
    } catch (error) {
      console.error('Error fetching analytics:', error);
      if (error.response && error.response.status === 401) {
        alert('Session expired. Please login again.');
        navigate('/');
      }
    } finally {
      setLoading(false);
    }
  };

  const fetchUsers = async () => {
    try {
      const token = localStorage.getItem('token');
      console.log("Fetching users data...");
      const response = await axios.get(`${API_BASE_URL}/admin/users`, {
        headers: {
          Authorization: `Bearer ${token}`,
        },
      });
      console.log("Users data received:", response.data);
      setUsers(response.data);
    } catch (error) {
      console.error('Error fetching users:', error);
    }
  };

  const handleFileChange = (e) => {
    if (e.target.files.length > 0) {
      setUploadFile(e.target.files[0]);
    }
  };

  const handleFileUpload = async () => {
    if (!uploadFile) {
      setUploadStatus('Please select a file first');
      return;
    }

    try {
      setUploadStatus('Uploading...');
      const token = localStorage.getItem('token');
      
      const formData = new FormData();
      formData.append('file', uploadFile);

      const response = await axios.post(`${API_BASE_URL}/knowledge/upload`, formData, {
        headers: {
          Authorization: `Bearer ${token}`,
          'Content-Type': 'multipart/form-data',
        },
      });
      
      setUploadStatus('File uploaded successfully');
      setUploadFile(null);
      document.getElementById('fileInput').value = '';
    } catch (error) {
      console.error('Error uploading file:', error);
      setUploadStatus(error.response?.data?.detail || 'Error uploading file');
    }
  };

  const promoteToAdmin = async (userId) => {
    try {
      const token = localStorage.getItem('token');
      await axios.post(`${API_BASE_URL}/admin/users/${userId}/make-admin`, {}, {
        headers: {
          Authorization: `Bearer ${token}`,
        },
      });
      
      fetchUsers();
      alert('User promoted to admin successfully');
    } catch (error) {
      console.error('Error promoting user:', error);
      alert('Error promoting user');
    }
  };

  const handleSearch = async () => {
    if (!searchQuery.trim() || searchQuery.length < 3) {
      alert('Please enter at least 3 characters to search');
      return;
    }
    
    setIsSearching(true);
    
    try {
      const token = localStorage.getItem('token');
      const response = await axios.get(`${API_BASE_URL}/knowledge/search?query=${encodeURIComponent(searchQuery)}&top_k=5`, {
        headers: {
          Authorization: `Bearer ${token}`,
        },
      });
      
      setSearchResults(response.data.results || []);
      
      if (response.data.results.length === 0) {
        alert('No results found');
      }
    } catch (error) {
      console.error('Error searching knowledge base:', error);
      alert('Error searching knowledge base');
    } finally {
      setIsSearching(false);
    }
  };

  const handleLogout = () => {
    // Use the AuthContext logout function instead of handling it directly
    logout();
    // No need to navigate as the logout function will handle that
  };

  return (
    <div style={{ backgroundColor: '#f5f7fb', minHeight: '100vh' }}>
      <header style={{ 
        padding: '1rem 2rem',
        backgroundColor: 'white',
        boxShadow: '0 2px 4px rgba(0,0,0,0.08)',
        display: 'flex',
        justifyContent: 'space-between',
        alignItems: 'center',
        marginBottom: '2rem'
      }}>
        <div style={{ display: 'flex', alignItems: 'center' }}>
          <img 
            src="https://cdn-icons-png.flaticon.com/512/2091/2091536.png" 
            alt="Admin Dashboard" 
            style={{ height: '32px', marginRight: '1rem' }}
          />
          <h1 style={{ fontSize: '1.5rem', margin: 0 }}>Admin Dashboard</h1>
        </div>
        
        <div>
          <button 
            onClick={() => navigate('/chat')}
            className="btn-secondary"
            style={{ marginRight: '1rem' }}
          >
            <svg xmlns="http://www.w3.org/2000/svg" width="16" height="16" fill="currentColor" viewBox="0 0 16 16" style={{ marginRight: '0.5rem' }}>
              <path d="M8 15A7 7 0 1 1 8 1a7 7 0 0 1 0 14zm0 1A8 8 0 1 0 8 0a8 8 0 0 0 0 16z"/>
              <path d="M5.255 5.786a.237.237 0 0 0 .241.247h.825c.138 0 .248-.113.266-.25.09-.656.54-1.134 1.342-1.134.686 0 1.314.343 1.314 1.168 0 .635-.374.927-.965 1.371-.673.489-1.206 1.06-1.168 1.987l.003.217a.25.25 0 0 0 .25.246h.811a.25.25 0 0 0 .25-.25v-.105c0-.718.273-.927 1.01-1.486.609-.463 1.244-.977 1.244-2.056 0-1.511-1.276-2.241-2.673-2.241-1.267 0-2.655.59-2.75 2.286zm1.557 5.763c0 .533.425.927 1.01.927.609 0 1.028-.394 1.028-.927 0-.552-.42-.94-1.029-.94-.584 0-1.009.388-1.009.94z"/>
            </svg>
            Support Chat
          </button>
          <button 
            onClick={handleLogout}
            className="btn-error"
            style={{ 
              padding: '0.5rem 1rem',
              display: 'flex',
              alignItems: 'center',
              fontSize: '1rem',
              fontWeight: '500'
            }}
          >
            <svg xmlns="http://www.w3.org/2000/svg" width="16" height="16" fill="currentColor" viewBox="0 0 16 16" style={{ marginRight: '0.5rem' }}>
              <path fillRule="evenodd" d="M10 12.5a.5.5 0 0 1-.5.5h-8a.5.5 0 0 1-.5-.5v-9a.5.5 0 0 1 .5-.5h8a.5.5 0 0 1 .5.5v2a.5.5 0 0 0 1 0v-2A1.5 1.5 0 0 0 9.5 2h-8A1.5 1.5 0 0 0 0 3.5v9A1.5 1.5 0 0 0 1.5 14h8a1.5 1.5 0 0 0 1.5-1.5v-2a.5.5 0 0 0-1 0v2z"/>
              <path fillRule="evenodd" d="M15.854 8.354a.5.5 0 0 0 0-.708l-3-3a.5.5 0 0 0-.708.708L14.293 7.5H5.5a.5.5 0 0 0 0 1h8.793l-2.147 2.146a.5.5 0 0 0 .708.708l3-3z"/>
            </svg>
            Logout
          </button>
        </div>
      </header>
      
      <div className="container">
        <div style={{ 
          display: 'flex',
          backgroundColor: 'white',
          borderRadius: 'var(--border-radius)',
          boxShadow: 'var(--box-shadow)',
          marginBottom: '2rem',
          overflow: 'hidden'
        }}>
          <TabButton 
            active={activeTab === 'analytics'} 
            onClick={() => setActiveTab('analytics')}
            icon="https://cdn-icons-png.flaticon.com/512/3688/3688532.png"
          >
            Analytics
          </TabButton>
          <TabButton 
            active={activeTab === 'users'} 
            onClick={() => setActiveTab('users')}
            icon="https://cdn-icons-png.flaticon.com/512/681/681494.png"
          >
            Users
          </TabButton>
          <TabButton 
            active={activeTab === 'knowledge'} 
            onClick={() => setActiveTab('knowledge')}
            icon="https://cdn-icons-png.flaticon.com/512/2232/2232688.png"
          >
            Knowledge Base
          </TabButton>
          <TabButton 
            active={activeTab === 'costs'} 
            onClick={() => setActiveTab('costs')}
            icon="https://cdn-icons-png.flaticon.com/512/2331/2331941.png"
          >
            Cost Estimates
          </TabButton>
        </div>
        
        <div className="card fade-in">
          {activeTab === 'analytics' && (
            <div>
              <div className="section-header">
                <h2>System Analytics</h2>
                {loading && <div className="loading-spinner-small"></div>}
              </div>
              
              {loading ? (
                <div className="loading-container">
                  <div className="loading-spinner"></div>
                  <p>Loading analytics data...</p>
                </div>
              ) : analytics ? (
                <div>
                  <div style={{ display: 'flex', flexWrap: 'wrap', gap: '1rem', marginBottom: '2rem' }}>
                    <div style={{ padding: '1rem', backgroundColor: '#f8f9fa', borderRadius: '5px', flex: 1, minWidth: '200px' }}>
                      <h3>Total Conversations</h3>
                      <p style={{ fontSize: '24px', fontWeight: 'bold' }}>{analytics.total_conversations}</p>
                    </div>
                    <div style={{ padding: '1rem', backgroundColor: '#f8f9fa', borderRadius: '5px', flex: 1, minWidth: '200px' }}>
                      <h3>Messages</h3>
                      <p style={{ fontSize: '24px', fontWeight: 'bold' }}>{analytics.total_messages}</p>
                    </div>
                    <div style={{ padding: '1rem', backgroundColor: '#f8f9fa', borderRadius: '5px', flex: 1, minWidth: '200px' }}>
                      <h3>Avg. Messages Per Conversation</h3>
                      <p style={{ fontSize: '24px', fontWeight: 'bold' }}>{analytics.avg_messages_per_conversation}</p>
                    </div>
                    <div style={{ padding: '1rem', backgroundColor: '#f8f9fa', borderRadius: '5px', flex: 1, minWidth: '200px' }}>
                      <h3>Escalation Rate</h3>
                      <p style={{ fontSize: '24px', fontWeight: 'bold' }}>{analytics.escalation_rate}%</p>
                    </div>
                  </div>
                  
                  <h3>Top Questions</h3>
                  <table style={{ width: '100%', borderCollapse: 'collapse', marginBottom: '2rem' }}>
                    <thead>
                      <tr>
                        <th style={{ textAlign: 'left', padding: '0.5rem', borderBottom: '1px solid #ddd' }}>Question</th>
                        <th style={{ textAlign: 'right', padding: '0.5rem', borderBottom: '1px solid #ddd' }}>Count</th>
                      </tr>
                    </thead>
                    <tbody>
                      {analytics.top_questions.map((item, index) => (
                        <tr key={index}>
                          <td style={{ padding: '0.5rem', borderBottom: '1px solid #eee' }}>{item.question}</td>
                          <td style={{ textAlign: 'right', padding: '0.5rem', borderBottom: '1px solid #eee' }}>{item.count}</td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                  
                  <h3>Daily Usage</h3>
                  <div style={{ height: '300px', border: '1px solid #ddd', padding: '1rem', position: 'relative' }}>
                    <div style={{ display: 'flex', height: '250px', alignItems: 'flex-end', gap: '5px' }}>
                      {analytics.daily_usage.map((day, index) => {
                        const maxCount = Math.max(...analytics.daily_usage.map(d => d.message_count));
                        const height = day.message_count > 0 ? (day.message_count / maxCount) * 100 : 0;
                        
                        return (
                          <div key={index} style={{ flex: 1, display: 'flex', flexDirection: 'column', alignItems: 'center' }}>
                            <div style={{ 
                              height: `${height}%`, 
                              width: '80%',
                              backgroundColor: '#007BFF', 
                              minHeight: day.message_count > 0 ? '5px' : '0',
                              position: 'relative'
                            }}>
                              <span style={{ position: 'absolute', top: '-20px', fontSize: '12px' }}>
                                {day.message_count}
                              </span>
                            </div>
                            <span style={{ fontSize: '10px', marginTop: '5px' }}>
                              {day.date.substring(5)}
                            </span>
                          </div>
                        );
                      })}
                    </div>
                  </div>
                </div>
              ) : (
                <div className="empty-state">
                  <img 
                    src="https://cdn-icons-png.flaticon.com/512/8027/8027825.png" 
                    alt="No data"
                    style={{ width: '100px', marginBottom: '1rem', opacity: 0.6 }}
                  />
                  <p>No analytics data available</p>
                </div>
              )}
            </div>
          )}
        </div>
      </div>
      
      <style jsx>{`
        .section-header {
          display: flex;
          justify-content: space-between;
          align-items: center;
          margin-bottom: 1.5rem;
          border-bottom: 1px solid var(--gray-200);
          padding-bottom: 1rem;
        }
        
        .section-header h2 {
          margin: 0;
          font-size: 1.5rem;
        }
        
        .loading-container {
          display: flex;
          flex-direction: column;
          align-items: center;
          justify-content: center;
          padding: 3rem 1rem;
          color: var(--gray-600);
        }
        
        .loading-spinner {
          width: 40px;
          height: 40px;
          border: 4px solid var(--gray-200);
          border-top: 4px solid var(--primary);
          border-radius: 50%;
          animation: spin 1s linear infinite;
          margin-bottom: 1rem;
        }
        
        .loading-spinner-small {
          width: 20px;
          height: 20px;
          border: 2px solid var(--gray-200);
          border-top: 2px solid var(--primary);
          border-radius: 50%;
          animation: spin 1s linear infinite;
        }
        
        @keyframes spin {
          0% { transform: rotate(0deg); }
          100% { transform: rotate(360deg); }
        }
        
        .empty-state {
          display: flex;
          flex-direction: column;
          align-items: center;
          justify-content: center;
          padding: 3rem 1rem;
          color: var(--gray-600);
        }
      `}</style>
    </div>
  );
}

function TabButton({ children, active, onClick, icon }) {
  return (
    <button 
      onClick={onClick} 
      style={{ 
        flex: 1,
        padding: '1rem',
        backgroundColor: active ? 'var(--primary)' : 'white',
        color: active ? 'white' : 'var(--gray-700)',
        border: 'none',
        borderBottom: active ? 'none' : '1px solid var(--gray-200)',
        cursor: 'pointer',
        display: 'flex',
        flexDirection: 'column',
        alignItems: 'center',
        justifyContent: 'center',
        transition: 'all 0.3s ease',
        gap: '0.5rem'
      }}
    >
      <img 
        src={icon} 
        alt={children} 
        style={{ 
          width: '24px', 
          height: '24px',
          filter: active ? 'brightness(0) invert(1)' : 'none',
          opacity: active ? 1 : 0.7
        }} 
      />
      {children}
    </button>
  );
}

export default Admin;