import React, { useState, useEffect } from 'react';
import axios from 'axios';
import { useNavigate } from 'react-router-dom';

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

  // Check if user is admin
  useEffect(() => {
    const token = localStorage.getItem('token');
    if (!token) {
      navigate('/');
      return;
    }

    // Decode JWT to check admin status (for UI only, backend will validate)
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

    // Load initial data
    fetchAnalytics();
    fetchUsers();
  }, [navigate]);

  const fetchAnalytics = async () => {
    try {
      setLoading(true);
      const token = localStorage.getItem('token');
      const response = await axios.get('http://localhost:8000/admin/analytics', {
        headers: {
          Authorization: `Bearer ${token}`,
        },
      });
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
      const response = await axios.get('http://localhost:8000/admin/users', {
        headers: {
          Authorization: `Bearer ${token}`,
        },
      });
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

      const response = await axios.post('http://localhost:8000/knowledge/upload', formData, {
        headers: {
          Authorization: `Bearer ${token}`,
          'Content-Type': 'multipart/form-data',
        },
      });
      
      setUploadStatus('File uploaded successfully');
      setUploadFile(null);
      // Reset the file input
      document.getElementById('fileInput').value = '';
    } catch (error) {
      console.error('Error uploading file:', error);
      setUploadStatus(error.response?.data?.detail || 'Error uploading file');
    }
  };

  const promoteToAdmin = async (userId) => {
    try {
      const token = localStorage.getItem('token');
      await axios.post(`http://localhost:8000/admin/users/${userId}/make-admin`, {}, {
        headers: {
          Authorization: `Bearer ${token}`,
        },
      });
      
      // Refresh users list
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
      const response = await axios.get(`http://localhost:8000/knowledge/search?query=${encodeURIComponent(searchQuery)}&top_k=5`, {
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

  return (
    <div style={{ margin: '2rem' }}>
      <h1>Admin Dashboard</h1>
      
      {/* Navigation Tabs */}
      <div style={{ marginBottom: '1.5rem', borderBottom: '1px solid #ccc' }}>
        <button 
          onClick={() => setActiveTab('analytics')} 
          style={{ 
            padding: '0.5rem 1rem',
            backgroundColor: activeTab === 'analytics' ? '#007BFF' : 'white',
            color: activeTab === 'analytics' ? 'white' : 'black',
            border: 'none',
            marginRight: '1rem',
            cursor: 'pointer'
          }}
        >
          Analytics
        </button>
        <button 
          onClick={() => setActiveTab('users')} 
          style={{ 
            padding: '0.5rem 1rem',
            backgroundColor: activeTab === 'users' ? '#007BFF' : 'white',
            color: activeTab === 'users' ? 'white' : 'black', 
            border: 'none',
            marginRight: '1rem',
            cursor: 'pointer'
          }}
        >
          Users
        </button>
        <button 
          onClick={() => setActiveTab('knowledge')} 
          style={{ 
            padding: '0.5rem 1rem',
            backgroundColor: activeTab === 'knowledge' ? '#007BFF' : 'white',
            color: activeTab === 'knowledge' ? 'white' : 'black',
            border: 'none',
            cursor: 'pointer'
          }}
        >
          Knowledge Base
        </button>
        <button 
          onClick={() => setActiveTab('costs')} 
          style={{ 
            padding: '0.5rem 1rem',
            backgroundColor: activeTab === 'costs' ? '#007BFF' : 'white',
            color: activeTab === 'costs' ? 'white' : 'black',
            border: 'none',
            cursor: 'pointer'
          }}
        >
          Cost Estimates
        </button>
      </div>
      
      {/* Analytics Tab */}
      {activeTab === 'analytics' && (
        <div>
          <h2>System Analytics</h2>
          {loading ? (
            <p>Loading analytics...</p>
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
                {/* Simple bar chart */}
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
            <p>No analytics data available</p>
          )}
        </div>
      )}
      
      {/* Users Tab */}
      {activeTab === 'users' && (
        <div>
          <h2>User Management</h2>
          <table style={{ width: '100%', borderCollapse: 'collapse' }}>
            <thead>
              <tr>
                <th style={{ textAlign: 'left', padding: '0.5rem', borderBottom: '1px solid #ddd' }}>Username</th>
                <th style={{ textAlign: 'left', padding: '0.5rem', borderBottom: '1px solid #ddd' }}>Email</th>
                <th style={{ textAlign: 'left', padding: '0.5rem', borderBottom: '1px solid #ddd' }}>Created</th>
                <th style={{ textAlign: 'center', padding: '0.5rem', borderBottom: '1px solid #ddd' }}>Admin</th>
                <th style={{ textAlign: 'right', padding: '0.5rem', borderBottom: '1px solid #ddd' }}>Actions</th>
              </tr>
            </thead>
            <tbody>
              {users.map(user => (
                <tr key={user.id}>
                  <td style={{ padding: '0.5rem', borderBottom: '1px solid #eee' }}>{user.username}</td>
                  <td style={{ padding: '0.5rem', borderBottom: '1px solid #eee' }}>{user.email || '-'}</td>
                  <td style={{ padding: '0.5rem', borderBottom: '1px solid #eee' }}>
                    {new Date(user.created_at).toLocaleDateString()}
                  </td>
                  <td style={{ textAlign: 'center', padding: '0.5rem', borderBottom: '1px solid #eee' }}>
                    {user.is_admin ? '✓' : '-'}
                  </td>
                  <td style={{ textAlign: 'right', padding: '0.5rem', borderBottom: '1px solid #eee' }}>
                    {!user.is_admin && (
                      <button 
                        onClick={() => promoteToAdmin(user.id)}
                        style={{ padding: '0.25rem 0.5rem', backgroundColor: '#28a745', color: 'white', border: 'none', borderRadius: '3px', cursor: 'pointer' }}
                      >
                        Make Admin
                      </button>
                    )}
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}
      
      {/* Knowledge Base Tab */}
      {activeTab === 'knowledge' && (
        <div>
          <h2>Knowledge Base Management</h2>
          <div style={{ marginBottom: '2rem', padding: '1rem', backgroundColor: '#f8f9fa', borderRadius: '5px' }}>
            <h3>Upload New Document</h3>
            <p>Upload PDF, Word, Excel, or text files (max 50MB)</p>
            
            <div style={{ marginTop: '1rem' }}>
              <input 
                type="file" 
                id="fileInput"
                onChange={handleFileChange} 
                style={{ marginBottom: '1rem' }} 
              />
              
              <div>
                <button 
                  onClick={handleFileUpload}
                  style={{ padding: '0.5rem 1rem', backgroundColor: '#007BFF', color: 'white', border: 'none', borderRadius: '3px', cursor: 'pointer' }}
                >
                  Upload
                </button>
                {uploadStatus && (
                  <span style={{ marginLeft: '1rem', color: uploadStatus.includes('Error') ? 'red' : 'green' }}>
                    {uploadStatus}
                  </span>
                )}
              </div>
            </div>
          </div>
          
          <div style={{ marginBottom: '2rem', padding: '1rem', backgroundColor: '#f8f9fa', borderRadius: '5px' }}>
            <h3>Knowledge Base Search</h3>
            <p>Test your knowledge base by searching for information</p>
            
            <div style={{ display: 'flex', marginTop: '1rem', marginBottom: '1rem' }}>
              <input 
                type="text" 
                placeholder="Enter search query (min 3 characters)"
                value={searchQuery}
                onChange={(e) => setSearchQuery(e.target.value)}
                style={{ flex: 1, padding: '0.5rem', marginRight: '0.5rem' }}
              />
              <button 
                onClick={handleSearch}
                style={{ 
                  padding: '0.5rem 1rem', 
                  backgroundColor: '#007BFF', 
                  color: 'white', 
                  border: 'none', 
                  borderRadius: '3px', 
                  cursor: 'pointer',
                  opacity: isSearching ? 0.7 : 1
                }}
                disabled={isSearching}
              >
                {isSearching ? 'Searching...' : 'Search'}
              </button>
            </div>
            
            {/* Search Results */}
            {searchResults.length > 0 && (
              <div>
                <h4>Search Results</h4>
                <div style={{ maxHeight: '400px', overflowY: 'auto', border: '1px solid #ddd', padding: '1rem' }}>
                  {searchResults.map((result, index) => (
                    <div key={index} style={{ marginBottom: '1.5rem', padding: '1rem', backgroundColor: 'white', borderRadius: '5px', boxShadow: '0 1px 3px rgba(0,0,0,0.1)' }}>
                      <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: '0.5rem' }}>
                        <strong>Source: {result.file_name}</strong>
                        <span>Relevance: {Math.round(result.relevance_score * 100)}%</span>
                      </div>
                      <div>{result.content}</div>
                    </div>
                  ))}
                </div>
              </div>
            )}
          </div>
        </div>
      )}
      
      {/* Cost Estimates Tab */}
      {activeTab === 'costs' && (
        <div>
          <h2>Cost Estimates</h2>
          {loading ? (
            <p>Loading cost data...</p>
          ) : analytics ? (
            <div>
              <div style={{ display: 'flex', flexWrap: 'wrap', gap: '1rem', marginBottom: '2rem' }}>
                <div style={{ padding: '1rem', backgroundColor: '#f8f9fa', borderRadius: '5px', flex: 1, minWidth: '200px' }}>
                  <h3>Estimated OpenAI Cost (Current Period)</h3>
                  <p style={{ fontSize: '24px', fontWeight: 'bold' }}>${analytics.estimated_total_cost?.toFixed(2) || '0.00'}</p>
                </div>
                <div style={{ padding: '1rem', backgroundColor: '#f8f9fa', borderRadius: '5px', flex: 1, minWidth: '200px' }}>
                  <h3>Estimated Monthly Run Rate</h3>
                  <p style={{ fontSize: '24px', fontWeight: 'bold' }}>${(analytics.estimated_total_cost ? (analytics.estimated_total_cost / analytics.days_in_period) * 30 : 0).toFixed(2)}</p>
                </div>
              </div>
              
              <h3>Token Usage Breakdown</h3>
              <table style={{ width: '100%', borderCollapse: 'collapse', marginBottom: '2rem' }}>
                <thead>
                  <tr>
                    <th style={{ textAlign: 'left', padding: '0.5rem', borderBottom: '1px solid #ddd' }}>Category</th>
                    <th style={{ textAlign: 'right', padding: '0.5rem', borderBottom: '1px solid #ddd' }}>Tokens</th>
                    <th style={{ textAlign: 'right', padding: '0.5rem', borderBottom: '1px solid #ddd' }}>Cost</th>
                  </tr>
                </thead>
                <tbody>
                  <tr>
                    <td style={{ padding: '0.5rem', borderBottom: '1px solid #eee' }}>Input (User Messages)</td>
                    <td style={{ textAlign: 'right', padding: '0.5rem', borderBottom: '1px solid #eee' }}>{analytics.estimated_user_tokens?.toLocaleString() || '0'}</td>
                    <td style={{ textAlign: 'right', padding: '0.5rem', borderBottom: '1px solid #eee' }}>${analytics.estimated_input_cost?.toFixed(2) || '0.00'}</td>
                  </tr>
                  <tr>
                    <td style={{ padding: '0.5rem', borderBottom: '1px solid #eee' }}>Output (AI Responses)</td>
                    <td style={{ textAlign: 'right', padding: '0.5rem', borderBottom: '1px solid #eee' }}>{analytics.estimated_ai_tokens?.toLocaleString() || '0'}</td>
                    <td style={{ textAlign: 'right', padding: '0.5rem', borderBottom: '1px solid #eee' }}>${analytics.estimated_output_cost?.toFixed(2) || '0.00'}</td>
                  </tr>
                  <tr style={{ fontWeight: 'bold' }}>
                    <td style={{ padding: '0.5rem' }}>Total</td>
                    <td style={{ textAlign: 'right', padding: '0.5rem' }}>{(analytics.estimated_user_tokens + analytics.estimated_ai_tokens)?.toLocaleString() || '0'}</td>
                    <td style={{ textAlign: 'right', padding: '0.5rem' }}>${analytics.estimated_total_cost?.toFixed(2) || '0.00'}</td>
                  </tr>
                </tbody>
              </table>
              
              <h3>Cost Optimization Tips</h3>
              <ul style={{ lineHeight: '1.5' }}>
                <li>Consider implementing caching for common questions to reduce API calls</li>
                <li>Adjust conversation history length to optimize token usage</li>
                <li>Set up usage limits and alerts to avoid unexpected charges</li>
                <li>See the COST_ESTIMATE.md file for detailed cost projections</li>
              </ul>
              
              <div style={{ marginTop: '2rem', padding: '1rem', backgroundColor: '#f8f9fa', borderRadius: '5px' }}>
                <h3>AWS Infrastructure Estimate</h3>
                <p>Based on your current usage patterns:</p>
                <table style={{ width: '100%', borderCollapse: 'collapse' }}>
                  <thead>
                    <tr>
                      <th style={{ textAlign: 'left', padding: '0.5rem', borderBottom: '1px solid #ddd' }}>Component</th>
                      <th style={{ textAlign: 'right', padding: '0.5rem', borderBottom: '1px solid #ddd' }}>Monthly Estimate</th>
                    </tr>
                  </thead>
                  <tbody>
                    <tr>
                      <td style={{ padding: '0.5rem', borderBottom: '1px solid #eee' }}>EC2/Elastic Beanstalk (t3.small)</td>
                      <td style={{ textAlign: 'right', padding: '0.5rem', borderBottom: '1px solid #eee' }}>$16.64</td>
                    </tr>
                    <tr>
                      <td style={{ padding: '0.5rem', borderBottom: '1px solid #eee' }}>S3 Storage (10GB)</td>
                      <td style={{ textAlign: 'right', padding: '0.5rem', borderBottom: '1px solid #eee' }}>$0.46</td>
                    </tr>
                    <tr>
                      <td style={{ padding: '0.5rem', borderBottom: '1px solid #eee' }}>RDS Database (if used)</td>
                      <td style={{ textAlign: 'right', padding: '0.5rem', borderBottom: '1px solid #eee' }}>$12.70</td>
                    </tr>
                    <tr>
                      <td style={{ padding: '0.5rem', borderBottom: '1px solid #eee' }}>Data Transfer (20GB)</td>
                      <td style={{ textAlign: 'right', padding: '0.5rem', borderBottom: '1px solid #eee' }}>$1.80</td>
                    </tr>
                    <tr style={{ fontWeight: 'bold' }}>
                      <td style={{ padding: '0.5rem' }}>AWS Infrastructure Total</td>
                      <td style={{ textAlign: 'right', padding: '0.5rem' }}>$31.60</td>
                    </tr>
                    <tr style={{ fontWeight: 'bold', backgroundColor: '#e9ecef' }}>
                      <td style={{ padding: '0.5rem' }}>Combined Total (AWS + OpenAI)</td>
                      <td style={{ textAlign: 'right', padding: '0.5rem' }}>${(31.60 + (analytics?.estimated_total_cost || 0)).toFixed(2)}</td>
                    </tr>
                  </tbody>
                </table>
              </div>
            </div>
          ) : (
            <p>No cost data available</p>
          )}
        </div>
      )}
    </div>
  );
}

export default Admin;