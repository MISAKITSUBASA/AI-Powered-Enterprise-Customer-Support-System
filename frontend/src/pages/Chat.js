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

  // Check auth token on component mount
  useEffect(() => {
    const token = localStorage.getItem('token');
    if (!token) {
      alert('You must be logged in to access the chat.');
      navigate('/');
      return;
    }

    // Check if user is admin (for UI purposes)
    try {
      const base64Url = token.split('.')[1];
      const base64 = base64Url.replace(/-/g, '+').replace(/_/g, '/');
      const payload = JSON.parse(window.atob(base64));
      setIsAdmin(payload.is_admin || false);
    } catch (error) {
      console.error('Error parsing token:', error);
    }

    // Load conversation history when component mounts
    fetchConversationHistory();
  }, [navigate]);

  // Function to fetch conversation history from the backend
  const fetchConversationHistory = async () => {
    setIsLoadingHistory(true);
    const token = localStorage.getItem('token');
    
    try {
      const response = await axios.get(
        'http://localhost:8000/conversation/history',
        {
          headers: {
            Authorization: `Bearer ${token}`,
          },
        }
      );
      
      // Set active conversation ID and load messages
      if (response.data.active_conversation_id) {
        setActiveConversationId(response.data.active_conversation_id);
      }
      
      // Format and set the conversation messages
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

  // Scroll to bottom of chat when new messages arrive
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

    // Add user message to conversation immediately for better UX
    const userMessage = { role: 'user', content: question };
    setConversation(prev => [...prev, userMessage]);
    
    try {
      const response = await axios.post(
        'http://localhost:8000/chat',
        { question },
        {
          headers: {
            Authorization: `Bearer ${token}`,
          },
        }
      );
      
      const { answer, confidence, escalate, conversation_id } = response.data;
      
      // Save conversation ID if this is a new conversation
      if (!activeConversationId && conversation_id) {
        setActiveConversationId(conversation_id);
      }
      
      // Add AI response to conversation
      const aiMessage = { 
        role: 'assistant', 
        content: answer,
        confidence,
        escalate
      };
      
      setConversation(prev => [...prev, aiMessage]);
      setQuestion('');

      if (escalate) {
        // Auto-escalate when confidence is low
        handleEscalate();
      }
    } catch (error) {
      console.error(error);
      if (error.response && error.response.status === 401) {
        alert('Your session has expired. Please login again.');
        navigate('/');
      } else {
        alert('Error sending message. Please try again.');
        // Remove the user message we added earlier
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
        'http://localhost:8000/escalate',
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
    <div style={{ margin: '2rem', maxWidth: '800px', marginLeft: 'auto', marginRight: 'auto' }}>
      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '1rem' }}>
        <h1>AI Customer Support</h1>
        <div>
          {isAdmin && (
            <button 
              onClick={() => navigate('/admin')}
              style={{ 
                marginRight: '1rem',
                padding: '0.5rem 1rem',
                backgroundColor: '#28a745',
                color: 'white',
                border: 'none',
                borderRadius: '3px',
                cursor: 'pointer'
              }}
            >
              Admin Dashboard
            </button>
          )}
          <button 
            onClick={handleLogout}
            style={{ 
              padding: '0.5rem 1rem',
              backgroundColor: '#6c757d',
              color: 'white',
              border: 'none',
              borderRadius: '3px',
              cursor: 'pointer'
            }}
          >
            Logout
          </button>
        </div>
      </div>
      
      {/* Chat Container */}
      <div
        ref={chatContainerRef}
        style={{
          border: '1px solid #ccc',
          borderRadius: '5px',
          padding: '1rem',
          height: '400px',
          overflowY: 'auto',
          backgroundColor: '#f8f9fa',
          marginBottom: '1rem'
        }}
      >
        {isLoadingHistory ? (
          <div style={{ textAlign: 'center', color: '#6c757d', marginTop: '150px' }}>
            <p>Loading conversation history...</p>
          </div>
        ) : conversation.length === 0 ? (
          <div style={{ textAlign: 'center', color: '#6c757d', marginTop: '150px' }}>
            <p>Ask me anything about our products or services!</p>
          </div>
        ) : (
          conversation.map((msg, index) => (
            <div 
              key={index} 
              style={{ 
                marginBottom: '1rem',
                display: 'flex',
                flexDirection: 'column',
                alignItems: msg.role === 'user' ? 'flex-end' : 'flex-start'
              }}
            >
              <div 
                style={{
                  backgroundColor: msg.role === 'user' ? '#007BFF' : 'white',
                  color: msg.role === 'user' ? 'white' : 'black',
                  padding: '0.75rem',
                  borderRadius: '1rem',
                  maxWidth: '70%',
                  boxShadow: '0 1px 2px rgba(0, 0, 0, 0.1)',
                  border: msg.role === 'assistant' ? '1px solid #ddd' : 'none',
                  position: 'relative'
                }}
              >
                {msg.content}
                
                {/* Show confidence score for AI responses */}
                {msg.role === 'assistant' && msg.confidence !== undefined && (
                  <div style={{ 
                    fontSize: '12px', 
                    marginTop: '0.5rem',
                    color: msg.confidence < 70 ? '#dc3545' : '#28a745'
                  }}>
                    Confidence: {Math.round(msg.confidence)}%
                    {msg.escalate && <span> • Escalated to human support</span>}
                  </div>
                )}
              </div>
            </div>
          ))
        )}
        {isLoading && (
          <div style={{ marginTop: '1rem', color: '#6c757d', textAlign: 'center' }}>
            AI is thinking...
          </div>
        )}
      </div>
      
      {/* Input Area */}
      <div style={{ display: 'flex', marginBottom: '1rem' }}>
        <textarea
          value={question}
          onChange={(e) => setQuestion(e.target.value)}
          onKeyPress={handleKeyPress}
          placeholder="Type your message..."
          style={{ 
            flex: 1,
            padding: '0.75rem',
            borderRadius: '5px',
            border: '1px solid #ccc',
            resize: 'none',
            height: '50px'
          }}
          disabled={isLoading || isLoadingHistory}
        />
        <button
          onClick={handleSend}
          style={{ 
            marginLeft: '0.5rem',
            padding: '0 1.5rem',
            backgroundColor: '#007BFF',
            color: 'white',
            border: 'none',
            borderRadius: '5px',
            cursor: 'pointer',
            opacity: (isLoading || isLoadingHistory) ? 0.7 : 1
          }}
          disabled={isLoading || isLoadingHistory}
        >
          Send
        </button>
      </div>
      
      {/* Manual Escalation Button */}
      <div style={{ textAlign: 'center' }}>
        <button
          onClick={handleEscalate}
          style={{ 
            padding: '0.5rem 1rem',
            backgroundColor: '#dc3545',
            color: 'white',
            border: 'none',
            borderRadius: '3px',
            cursor: 'pointer',
            opacity: isLoadingHistory ? 0.7 : 1
          }}
          disabled={isLoadingHistory}
        >
          Escalate to Human Support
        </button>
      </div>
    </div>
  );
}

export default Chat;
