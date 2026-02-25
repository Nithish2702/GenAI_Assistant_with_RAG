import { useState, useEffect, useRef } from 'react'
import ReactMarkdown from 'react-markdown'
import './App.css'

const API_URL = 'http://localhost:8000'

function App() {
  const [messages, setMessages] = useState([])
  const [input, setInput] = useState('')
  const [loading, setLoading] = useState(false)
  const [sessionId, setSessionId] = useState('')
  const [chatHistory, setChatHistory] = useState([])
  const messagesEndRef = useRef(null)

  useEffect(() => {
    const history = JSON.parse(localStorage.getItem('chatHistory') || '[]')
    setChatHistory(history)
    
    let sid = localStorage.getItem('currentSessionId')
    if (!sid) {
      sid = 'session_' + Date.now() + '_' + Math.random().toString(36).substr(2, 9)
      localStorage.setItem('currentSessionId', sid)
    }
    setSessionId(sid)
    
    const savedMessages = JSON.parse(localStorage.getItem(`chat_${sid}`) || '[]')
    setMessages(savedMessages)
  }, [])

  useEffect(() => {
    scrollToBottom()
  }, [messages])

  useEffect(() => {
    if (sessionId && messages.length > 0) {
      localStorage.setItem(`chat_${sessionId}`, JSON.stringify(messages))
      updateChatHistory()
    }
  }, [messages, sessionId])

  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' })
  }

  const updateChatHistory = () => {
    const history = JSON.parse(localStorage.getItem('chatHistory') || '[]')
    const existingIndex = history.findIndex(h => h.sessionId === sessionId)
    
    const chatItem = {
      sessionId,
      title: messages[0]?.content.substring(0, 40) || 'New Chat',
      timestamp: new Date().toISOString(),
      messageCount: messages.length
    }
    
    if (existingIndex >= 0) {
      history[existingIndex] = chatItem
    } else {
      history.unshift(chatItem)
    }
    
    const trimmedHistory = history.slice(0, 20)
    localStorage.setItem('chatHistory', JSON.stringify(trimmedHistory))
    setChatHistory(trimmedHistory)
  }

  const sendMessage = async (e) => {
    e.preventDefault()
    if (!input.trim() || loading) return

    const userMessage = input.trim()
    setInput('')
    setMessages(prev => [...prev, { role: 'user', content: userMessage, timestamp: new Date() }])
    setLoading(true)

    try {
      const response = await fetch(`${API_URL}/api/chat`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          sessionId: sessionId,
          message: userMessage
        })
      })

      if (!response.ok) throw new Error('API request failed')

      const data = await response.json()
      setMessages(prev => [...prev, {
        role: 'assistant',
        content: data.reply,
        timestamp: new Date(),
        metadata: {
          tokensUsed: data.tokensUsed,
          retrievedChunks: data.retrievedChunks,
          similarityScores: data.similarityScores
        }
      }])
    } catch (error) {
      setMessages(prev => [...prev, {
        role: 'assistant',
        content: 'Sorry, I encountered an error. Please try again.',
        timestamp: new Date(),
        error: true
      }])
    } finally {
      setLoading(false)
    }
  }

  const newChat = () => {
    const sid = 'session_' + Date.now() + '_' + Math.random().toString(36).substr(2, 9)
    localStorage.setItem('currentSessionId', sid)
    setSessionId(sid)
    setMessages([])
  }

  const loadChat = (sid) => {
    const savedMessages = JSON.parse(localStorage.getItem(`chat_${sid}`) || '[]')
    setMessages(savedMessages.map(msg => ({
      ...msg,
      timestamp: new Date(msg.timestamp)
    })))
    setSessionId(sid)
    localStorage.setItem('currentSessionId', sid)
  }

  const deleteChat = (sid, e) => {
    e.stopPropagation()
    localStorage.removeItem(`chat_${sid}`)
    const history = chatHistory.filter(h => h.sessionId !== sid)
    localStorage.setItem('chatHistory', JSON.stringify(history))
    setChatHistory(history)
    
    if (sid === sessionId) {
      newChat()
    }
  }

  return (
    <div className="app">
      <div className="sidebar">
        <div className="sidebar-header">
          <button onClick={newChat} className="new-chat-btn">
            + New chat
          </button>
        </div>
        <div className="chat-list">
          {chatHistory.length === 0 ? (
            <div className="no-history">No previous chats</div>
          ) : (
            chatHistory.map((chat) => (
              <div
                key={chat.sessionId}
                className={`chat-item ${chat.sessionId === sessionId ? 'active' : ''}`}
                onClick={() => loadChat(chat.sessionId)}
              >
                <div className="chat-item-content">
                  <div className="chat-item-title">{chat.title}</div>
                  <div className="chat-item-meta">
                    {new Date(chat.timestamp).toLocaleDateString()}
                  </div>
                </div>
                <button
                  className="delete-chat"
                  onClick={(e) => deleteChat(chat.sessionId, e)}
                  title="Delete chat"
                >
                  🗑️
                </button>
              </div>
            ))
          )}
        </div>
      </div>

      <div className="main-content">
        <div className="header">
          <h1>RAG Chat Assistant</h1>
        </div>

        <div className="chat-container">
          <div className="messages">
            {messages.length === 0 && (
              <div className="welcome">
                <h2>How can I help you today?</h2>
                <div className="welcome-topics">
                  <div className="topic-card">🔐 Account Management</div>
                  <div className="topic-card">🔑 Password Reset</div>
                  <div className="topic-card">💳 Payment Methods</div>
                  <div className="topic-card">📦 Subscription Plans</div>
                  <div className="topic-card">🔒 Security Features</div>
                  <div className="topic-card">🔧 API Access</div>
                  <div className="topic-card">📊 Data Export</div>
                  <div className="topic-card">💬 Customer Support</div>
                </div>
              </div>
            )}

            {messages.map((msg, idx) => (
              <div key={idx} className={`message ${msg.role}`}>
                <div className="message-avatar">
                  {msg.role === 'user' ? '👤' : '🤖'}
                </div>
                <div className="message-body">
                  <div className="message-content">
                    {msg.role === 'assistant' ? (
                      <ReactMarkdown>{msg.content}</ReactMarkdown>
                    ) : (
                      <p>{msg.content}</p>
                    )}
                  </div>
                  {msg.metadata && (
                    <div className="metadata">
                      <span>Chunks: {msg.metadata.retrievedChunks}</span>
                      <span>Tokens: {msg.metadata.tokensUsed}</span>
                      {msg.metadata.similarityScores && msg.metadata.similarityScores.length > 0 && (
                        <span>Score: {msg.metadata.similarityScores[0].toFixed(3)}</span>
                      )}
                    </div>
                  )}
                  <div className="timestamp">
                    {msg.timestamp.toLocaleTimeString()}
                  </div>
                </div>
              </div>
            ))}

            {loading && (
              <div className="message assistant">
                <div className="message-avatar">🤖</div>
                <div className="message-body">
                  <div className="typing-indicator">
                    <span></span>
                    <span></span>
                    <span></span>
                  </div>
                </div>
              </div>
            )}

            <div ref={messagesEndRef} />
          </div>

          <div className="input-container">
            <form onSubmit={sendMessage} className="input-form">
              <input
                type="text"
                value={input}
                onChange={(e) => setInput(e.target.value)}
                placeholder="Send a message..."
                disabled={loading}
                className="input-field"
              />
              <button type="submit" disabled={loading || !input.trim()} className="send-btn">
                ↑
              </button>
            </form>
          </div>
        </div>
      </div>
    </div>
  )
}

export default App
