import React, { useState, useEffect, useRef } from 'react';
import axios from 'axios';
import { useNavigate } from 'react-router-dom';

function Chat() {
  const [question, setQuestion] = useState('');
  const [conversation, setConversation] = useState([]);
  const [isLoading, setIsLoading] = useState(false);
  const [isAdmin, setIsAdmin] = useState(false);
  const [activeConversationId, setActiveConversationId] = useState(null);
  const [isLoadingHistory, setIsLoadingHistory] = useState(true);
  const chatContainerRef = useRef(null);
  const navigate = useNavigate();

  // Define API base URL
  const API_BASE_URL = ''; // Empty for relative paths with Nginx proxy

  useEffect(() => {
    const token = localStorage.getItem('token');
    if (!token) {
      alert('You must be logged in to access the chat.');
      navigate('/');
      return;
    }

    try {
      const base64Url = token.split('.')[1];
      const base64 = base64Url.replace(/-/g, '+').replace(/_/g, '/');
      const payload = JSON.parse(window.atob(base64));
      setIsAdmin(payload.is_admin || false);
    } catch (error) {
      console.error('Error parsing token:', error);
    }

    fetchConversationHistory();
  }, [navigate]);

  const fetchConversationHistory = async () => {
    setIsLoadingHistory(true);
    const token = localStorage.getItem('token');
    
    try {
      console.log("Fetching conversation history...");
      const response = await axios.get(
        `${API_BASE_URL}/conversation/history`,
        {
          headers: {
            Authorization: `Bearer ${token}`,
          },
        }
      );
      
      console.log("Conversation history received:", response.data);
      
      if (response.data.active_conversation_id) {
        setActiveConversationId(response.data.active_conversation_id);
      }
      
      if (response.data.messages && response.data.messages.length > 0) {
        const formattedMessages = response.data.messages.map(msg => ({
          role: msg.role,
          content: msg.content,
          confidence: msg.confidence_score,
          escalate: msg.was_escalated
        }));
        
        setConversation(formattedMessages);
      }
    } catch (error) {
      console.error('Error fetching conversation history:', error);
      if (error.response && error.response.status === 401) {
        alert('Your session has expired. Please login again.');
        navigate('/');
      } else {
        console.error('Failed to load conversation history');
      }
    } finally {
      setIsLoadingHistory(false);
    }
  };

  useEffect(() => {
    if (chatContainerRef.current) {
      chatContainerRef.current.scrollTop = chatContainerRef.current.scrollHeight;
    }
  }, [conversation]);

  const handleSend = async () => {
    if (!question.trim()) return;
    
    const token = localStorage.getItem('token');
    if (!token) {
      alert('No token found. Please login again.');
      navigate('/');
      return;
    }

    setIsLoading(true);

    const userMessage = { role: 'user', content: question };
    setConversation(prev => [...prev, userMessage]);
    
    try {
      console.log("Sending chat message:", question);
      const response = await axios.post(
        `${API_BASE_URL}/chat`,
        { question },
        {
          headers: {
            Authorization: `Bearer ${token}`,
          },
        }
      );
      
      console.log("Chat response received:", response.data);
      
      const { answer, confidence, escalate, conversation_id } = response.data;
      
      if (!activeConversationId && conversation_id) {
        setActiveConversationId(conversation_id);
      }
      
      const aiMessage = { 
        role: 'assistant', 
        content: answer,
        confidence,
        escalate
      };
      
      setConversation(prev => [...prev, aiMessage]);
      setQuestion('');

      if (escalate) {
        handleEscalate();
      }
    } catch (error) {
      console.error("Chat error:", error);
      if (error.response && error.response.status === 401) {
        alert('Your session has expired. Please login again.');
        navigate('/');
      } else {
        alert('Error sending message. Please try again.');
        setConversation(prev => prev.slice(0, -1));
      }
    } finally {
      setIsLoading(false);
    }
  };

  const handleEscalate = async () => {
    const token = localStorage.getItem('token');
    try {
      await axios.post(
        `${API_BASE_URL}/escalate`,
        {},
        {
          headers: {
            Authorization: `Bearer ${token}`,
          },
        }
      );
      alert('Your conversation has been escalated to human support. Someone will contact you shortly.');
    } catch (error) {
      console.error('Error escalating conversation:', error);
    }
  };

  const handleKeyPress = (e) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      handleSend();
    }
  };

  const handleLogout = () => {
    localStorage.removeItem('token');
    navigate('/');
  };

  return (
    <div style={{ 
      display: 'flex', 
      flexDirection: 'column', 
      height: '100vh',
      backgroundColor: '#f5f7fb' 
    }}>
      <header style={{ 
        padding: '1rem 2rem',
        backgroundColor: 'white',
        boxShadow: '0 2px 4px rgba(0,0,0,0.08)',
        display: 'flex',
        justifyContent: 'space-between',
        alignItems: 'center'
      }}>
        <div style={{ display: 'flex', alignItems: 'center' }}>
          <img 
            src="https://cdn-icons-png.flaticon.com/512/4233/4233830.png" 
            alt="AI Support" 
            style={{ height: '32px', marginRight: '1rem' }}
          />
          <h1 style={{ fontSize: '1.5rem', margin: 0 }}>AI Customer Support</h1>
        </div>
        
        <div>
          {isAdmin && (
            <button 
              onClick={() => navigate('/admin')}
              className="btn-secondary"
              style={{ marginRight: '1rem' }}
            >
              Admin Dashboard
            </button>
          )}
          <button 
            onClick={handleLogout}
            className="btn-error"
          >
            Logout
          </button>
        </div>
      </header>
      
      <div style={{ 
        flex: 1, 
        padding: '1.5rem 2rem',
        display: 'flex',
        flexDirection: 'column'
      }}>
        <div
          ref={chatContainerRef}
          style={{
            flex: 1,
            borderRadius: 'var(--border-radius)',
            padding: '1.5rem',
            overflowY: 'auto',
            backgroundColor: 'white',
            boxShadow: 'var(--box-shadow)',
            marginBottom: '1.5rem',
            backgroundImage: 'url("https://www.transparenttextures.com/patterns/cubes.png")',
            backgroundBlendMode: 'overlay'
          }}
        >
          {isLoadingHistory ? (
            <div style={{ 
              display: 'flex', 
              flexDirection: 'column',
              alignItems: 'center', 
              justifyContent: 'center', 
              height: '100%' 
            }}>
              <div className="loading-spinner"></div>
              <p style={{ marginTop: '1rem', color: 'var(--gray-600)' }}>Loading conversation history...</p>
            </div>
          ) : conversation.length === 0 ? (
            <div style={{ 
              textAlign: 'center', 
              marginTop: '4rem',
              padding: '2rem',
              color: 'var(--gray-600)',
              display: 'flex',
              flexDirection: 'column',
              alignItems: 'center'
            }}>
              <img 
                src="https://cdn-icons-png.flaticon.com/512/1041/1041916.png"
                alt="Empty chat"
                style={{ width: '120px', marginBottom: '1.5rem', opacity: 0.6 }}
              />
              <h2 style={{ fontWeight: '500', marginBottom: '0.5rem' }}>How can I help you today?</h2>
              <p>Ask me anything about our products or services!</p>
            </div>
          ) : (
            conversation.map((msg, index) => (
              <div 
                key={index} 
                className="fade-in"
                style={{ 
                  marginBottom: '1.5rem',
                  display: 'flex',
                  flexDirection: 'column',
                  alignItems: msg.role === 'user' ? 'flex-end' : 'flex-start'
                }}
              >
                {msg.role === 'assistant' && (
                  <div style={{ 
                    display: 'flex', 
                    alignItems: 'center', 
                    marginBottom: '0.35rem',
                    color: 'var(--gray-700)',
                    fontSize: '0.9rem'
                  }}>
                    <img 
                      src="https://cdn-icons-png.flaticon.com/512/4233/4233830.png"
                      alt="AI"
                      style={{ width: '20px', height: '20px', marginRight: '0.5rem' }}
                    />
                    <span>AI Assistant</span>
                  </div>
                )}
                
                <div 
                  style={{
                    backgroundColor: msg.role === 'user' ? 'var(--primary)' : 'white',
                    color: msg.role === 'user' ? 'white' : 'var(--gray-800)',
                    padding: '1rem',
                    borderRadius: msg.role === 'user' ? '18px 18px 0 18px' : '18px 18px 18px 0',
                    maxWidth: '75%',
                    boxShadow: '0 1px 2px rgba(0, 0, 0, 0.1)',
                    border: msg.role === 'assistant' ? '1px solid var(--gray-200)' : 'none',
                    position: 'relative',
                    lineHeight: '1.5'
                  }}
                >
                  {msg.content}
                  
                  {msg.role === 'assistant' && msg.confidence !== undefined && (
                    <div style={{ 
                      fontSize: '0.8rem', 
                      marginTop: '0.5rem',
                      color: msg.confidence < 70 ? 'var(--error)' : 'var(--success)',
                      display: 'flex',
                      alignItems: 'center'
                    }}>
                      <span 
                        style={{ 
                          width: '8px', 
                          height: '8px', 
                          borderRadius: '50%', 
                          backgroundColor: msg.confidence < 70 ? 'var(--error)' : 'var(--success)',
                          display: 'inline-block',
                          marginRight: '0.5rem'
                        }}
                      />
                      Confidence: {Math.round(msg.confidence)}%
                      {msg.escalate && <span style={{ marginLeft: '0.5rem' }}>• Escalated to human support</span>}
                    </div>
                  )}
                </div>
              </div>
            ))
          )}
          {isLoading && (
            <div style={{ padding: '0.75rem', color: 'var(--gray-600)', textAlign: 'center' }}>
              <div className="typing-indicator">
                <span></span>
                <span></span>
                <span></span>
              </div>
            </div>
          )}
        </div>
        
        <div style={{ 
          backgroundColor: 'white',
          borderRadius: 'var(--border-radius)',
          boxShadow: 'var(--box-shadow)',
          padding: '1.25rem',
        }}>
          <div style={{ display: 'flex', marginBottom: '1rem' }}>
            <textarea
              value={question}
              onChange={(e) => setQuestion(e.target.value)}
              onKeyPress={handleKeyPress}
              placeholder="Type your message..."
              style={{ 
                flex: 1,
                padding: '1rem',
                borderRadius: 'var(--border-radius)',
                border: '1px solid var(--gray-300)',
                resize: 'none',
                minHeight: '60px',
                fontFamily: 'inherit',
                fontSize: '1rem'
              }}
              disabled={isLoading || isLoadingHistory}
            />
            <button
              onClick={handleSend}
              className="btn-primary"
              style={{ 
                marginLeft: '0.75rem',
                padding: '0 1.5rem',
                display: 'flex',
                alignItems: 'center',
                alignSelf: 'flex-end',
                height: '45px',
                opacity: (isLoading || isLoadingHistory) ? 0.7 : 1
              }}
              disabled={isLoading || isLoadingHistory}
            >
              <span>Send</span>
              <svg xmlns="http://www.w3.org/2000/svg" width="16" height="16" fill="currentColor" viewBox="0 0 16 16" style={{ marginLeft: '0.5rem' }}>
                <path d="M15.964.686a.5.5 0 0 0-.65-.65L.767 5.855H.766l-.452.18a.5.5 0 0 0-.082.887l.41.26.001.002 4.995 3.178 3.178 4.995.002.002.26.41a.5.5 0 0 0 .886-.083l6-15Zm-1.833 1.89L6.637 10.07l-.215-.338a.5.5 0 0 0-.154-.154l-.338-.215 7.494-7.494 1.178-.471-.47 1.178Z"/>
              </svg>
            </button>
          </div>
          
          <div style={{ textAlign: 'center' }}>
            <button
              onClick={handleEscalate}
              style={{ 
                padding: '0.5rem 1rem',
                backgroundColor: 'transparent',
                color: 'var(--error)',
                border: '1px solid var(--error)',
                borderRadius: 'var(--border-radius)',
                cursor: 'pointer',
                opacity: isLoadingHistory ? 0.7 : 1,
                transition: 'all 0.2s ease'
              }}
              onMouseOver={(e) => e.target.style.backgroundColor = 'rgba(244, 67, 54, 0.05)'}
              onMouseOut={(e) => e.target.style.backgroundColor = 'transparent'}
              disabled={isLoadingHistory}
            >
              Escalate to Human Support
            </button>
          </div>
        </div>
      </div>
      
      <style jsx>{`
        .loading-spinner {
          width: 30px;
          height: 30px;
          border: 4px solid var(--gray-200);
          border-top: 4px solid var(--primary);
          border-radius: 50%;
          animation: spin 1s linear infinite;
        }

        @keyframes spin {
          0% { transform: rotate(0deg); }
          100% { transform: rotate(360deg); }
        }

        .typing-indicator {
          display: flex;
          justify-content: center;
        }

        .typing-indicator span {
          width: 8px;
          height: 8px;
          margin: 0 2px;
          background-color: var(--gray-400);
          border-radius: 50%;
          display: inline-block;
          animation: bounce 1.5s infinite ease-in-out;
        }

        .typing-indicator span:nth-child(2) {
          animation-delay: 0.2s;
        }

        .typing-indicator span:nth-child(3) {
          animation-delay: 0.4s;
        }

        @keyframes bounce {
          0%, 80%, 100% { transform: translateY(0); }
          40% { transform: translateY(-8px); }
        }
      `}</style>
    </div>
  );
}

export default Chat;
