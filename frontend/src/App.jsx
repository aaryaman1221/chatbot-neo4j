import React, { useState, useEffect } from 'react';
import ReactMarkdown from 'react-markdown';
import remarkGfm from 'remark-gfm';
import './index.css';

const API_BASE = 'http://localhost:8000/api';

function App() {
  const [uri, setUri] = useState('neo4j://localhost:7687');
  const [user, setUser] = useState('neo4j');
  const [password, setPassword] = useState('');
  
  const [connected, setConnected] = useState(false);
  const [stats, setStats] = useState(null);
  const [repos, setRepos] = useState([]);
  const [selectedRepos, setSelectedRepos] = useState([]);
  const [topK, setTopK] = useState(5);
  
  const [messages, setMessages] = useState([]);
  const [input, setInput] = useState('');
  const [loading, setLoading] = useState(false);

  const getHeaders = () => ({
    'Content-Type': 'application/json',
    'x-neo4j-uri': uri,
    'x-neo4j-user': user,
    'x-neo4j-password': password
  });

  const checkConnection = async () => {
    try {
      const res = await fetch(`${API_BASE}/check-connection`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ uri, user, password })
      });
      if (res.ok) {
        setConnected(true);
        fetchStats();
        fetchRepos();
      } else {
        setConnected(false);
        alert('Connection failed');
      }
    } catch (e) {
      setConnected(false);
      alert('Error connecting to backend');
    }
  };

  const fetchStats = async () => {
    try {
      const res = await fetch(`${API_BASE}/stats`, { headers: getHeaders() });
      if (res.ok) {
        const data = await res.json();
        setStats(data);
      }
    } catch (e) {
      console.error(e);
    }
  };

  const fetchRepos = async () => {
    try {
      const res = await fetch(`${API_BASE}/repos`, { headers: getHeaders() });
      if (res.ok) {
        const data = await res.json();
        setRepos(data.repos);
      }
    } catch (e) {
      console.error(e);
    }
  };

  const sendMessage = async (msgText) => {
    if (!msgText.trim()) return;
    
    const newMsg = { role: 'user', content: msgText };
    setMessages((prev) => [...prev, newMsg]);
    setInput('');
    setLoading(true);

    try {
      const res = await fetch(`${API_BASE}/chat`, {
        method: 'POST',
        headers: getHeaders(),
        body: JSON.stringify({
          query: msgText,
          selected_repos: selectedRepos.length > 0 ? selectedRepos : null,
          top_k: topK
        })
      });
      const data = await res.json();
      if (res.ok) {
        setMessages((prev) => [...prev, { role: 'assistant', content: data.answer }]);
      } else {
        setMessages((prev) => [...prev, { role: 'assistant', content: `❌ Error: ${data.detail}` }]);
      }
    } catch (e) {
      setMessages((prev) => [...prev, { role: 'assistant', content: `❌ Request failed: ${e.message}` }]);
    } finally {
      setLoading(false);
    }
  };

  const suggestions = [
    "What are the most critical open issues?",
    "Summarise recent bug reports",
    "What files were changed most frequently?",
    "Which commits touched authentication code?",
    "Which parent-repo files import from the helper repo?",
    "What functions in the parent repo call helper repo functions?",
  ];

  return (
    <div className="app-container">
      {/* Sidebar */}
      <div className="sidebar">
        <div style={{ textAlign: 'center', padding: '0.5rem 0 1rem 0' }}>
          <div style={{ fontSize: '2.4rem' }}>🕸️</div>
          <div style={{ fontSize: '1.1rem', fontWeight: 700, color: '#00d4ff', letterSpacing: '0.5px' }}>GraphRAG Chat</div>
          <div style={{ fontSize: '0.72rem', color: 'rgba(255,255,255,0.4)', marginTop: '2px' }}>Neo4j · Gemini · Query-Only</div>
        </div>

        <div className="section-header">🗄️ Neo4j Connection</div>
        <div className="input-group">
          <label className="input-label">Neo4j URI</label>
          <input className="text-input" value={uri} onChange={e => setUri(e.target.value)} />
        </div>
        <div className="input-group">
          <label className="input-label">Username</label>
          <input className="text-input" value={user} onChange={e => setUser(e.target.value)} />
        </div>
        <div className="input-group">
          <label className="input-label">Password</label>
          <input className="text-input" type="password" value={password} onChange={e => setPassword(e.target.value)} />
        </div>
        <button className="btn" onClick={checkConnection} style={{ marginTop: '0.5rem' }}>Connect</button>

        {connected && stats && (
          <>
            <div className="section-header">📊 Graph Metrics</div>
            <div className="stats-grid">
              <div className="stat-item"><span className="stat-value">{stats.nodes?.toLocaleString()}</span><span className="stat-label">Nodes</span></div>
              <div className="stat-item"><span className="stat-value">{stats.relationships?.toLocaleString()}</span><span className="stat-label">Edges</span></div>
              <div className="stat-item"><span className="stat-value">{stats.commits?.toLocaleString()}</span><span className="stat-label">Commits</span></div>
              <div className="stat-item"><span className="stat-value">{stats.files?.toLocaleString()}</span><span className="stat-label">Files</span></div>
              <div className="stat-item"><span className="stat-value">{stats.issues?.toLocaleString()}</span><span className="stat-label">Issues</span></div>
              <div className="stat-item"><span className="stat-value">{stats.modules?.toLocaleString()}</span><span className="stat-label">Modules</span></div>
            </div>
          </>
        )}

        <div className="section-header">⚙️ RAG Settings</div>
        <div className="input-group">
          <label className="input-label">Top-K retrieval results: {topK}</label>
          <input type="range" min="1" max="15" value={topK} onChange={e => setTopK(parseInt(e.target.value))} />
        </div>
      </div>

      {/* Main Content */}
      <div className="main-content">
        <h1 className="hero-title">🕸️ GraphRAG Chat</h1>
        <p className="hero-sub">Query your Neo4j knowledge graph with natural language, powered by Google Gemini</p>

        {/* Pipeline Overview */}
        <div className="metrics-row">
          <div className="metric-card">
            <div className="num">1️⃣</div><div className="title">Neo4j Graph</div><div className="desc">Pre-built by backend</div>
          </div>
          <div className="metric-card">
            <div className="num">2️⃣</div><div className="title">VectorRetriever</div><div className="desc">Cosine search</div>
          </div>
          <div className="metric-card">
            <div className="num">3️⃣</div><div className="title">Fulltext Search</div><div className="desc">Commit summaries</div>
          </div>
          <div className="metric-card">
            <div className="num">4️⃣</div><div className="title">Context Merge</div><div className="desc">Fused into prompt</div>
          </div>
          <div className="metric-card">
            <div className="num">5️⃣</div><div className="title">Gemini 2.5 Flash</div><div className="desc">Generates answer</div>
          </div>
        </div>

        {connected && stats?.nodes > 0 && (
          <div className="ready-banner">
            ✅ Knowledge graph ready &nbsp;·&nbsp; <strong>{stats.nodes.toLocaleString()} nodes</strong> &nbsp;·&nbsp; <strong>{stats.relationships.toLocaleString()} relationships</strong> &nbsp;·&nbsp; Ask anything below ↓
          </div>
        )}

        {/* Chat Area */}
        <div className="chat-container">
          {messages.length === 0 && connected && (
            <div style={{ textAlign: 'center', margin: '2rem 0' }}>
              <div style={{ fontSize: '0.78rem', color: 'rgba(255,255,255,0.4)', marginBottom: '8px' }}>💡 Try asking…</div>
              <div style={{ display: 'flex', flexWrap: 'wrap', gap: '0.5rem', justifyContent: 'center' }}>
                {suggestions.map((sugg, i) => (
                  <span key={i} className="suggestion-chip" onClick={() => sendMessage(sugg)}>{sugg}</span>
                ))}
              </div>
            </div>
          )}

          {messages.map((m, i) => (
            <div key={i} className="chat-message">
              <div className="chat-avatar">{m.role === 'user' ? '🧑‍💻' : '🤖'}</div>
              <div className="chat-content">
                <ReactMarkdown remarkPlugins={[remarkGfm]}>{m.content}</ReactMarkdown>
              </div>
            </div>
          ))}
          {loading && (
            <div className="chat-message">
              <div className="chat-avatar">🤖</div>
              <div className="chat-content" style={{ opacity: 0.7 }}>🔍 Searching knowledge graph...</div>
            </div>
          )}
        </div>

        {/* Input Area */}
        <div className="chat-input-wrapper">
          <input 
            className="text-input" 
            placeholder={connected ? "Ask a question about the repository issues or codebase…" : "Configure Neo4j credentials in the sidebar first…"}
            value={input}
            onChange={(e) => setInput(e.target.value)}
            onKeyDown={(e) => e.key === 'Enter' && sendMessage(input)}
            disabled={!connected || loading}
          />
          <button className="btn" onClick={() => sendMessage(input)} disabled={!connected || loading || !input.trim()}>
            Send
          </button>
        </div>

        <div style={{ textAlign: 'center', marginTop: '2rem', paddingTop: '1rem', borderTop: '1px solid rgba(255,255,255,0.06)', fontSize: '0.7rem', color: 'rgba(255,255,255,0.2)', letterSpacing: '0.5px' }}>
          GraphRAG Chat &nbsp;·&nbsp; Neo4j + Google Gemini
        </div>
      </div>
    </div>
  );
}

export default App;
