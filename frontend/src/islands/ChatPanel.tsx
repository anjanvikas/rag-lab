import { useState, useRef, useEffect } from 'react';
import ReactMarkdown from 'react-markdown';
import remarkGfm from 'remark-gfm';
import remarkBreaks from 'remark-breaks';
import { useChatStream } from '../hooks/useChatStream';
import { useAppStore } from '../stores/appStore';
import { useAuthStore } from '../stores/authStore';
import { useChatStore } from '../stores/chatStore';
import type { PipelineStep } from '../stores/chatStore';

function ThinkingProgress({ active, completed }: { active: PipelineStep | null, completed: PipelineStep[] }) {
  if (!active && completed.length === 0) return null;
  
  return (
    <div className="step-progress-list">
      {completed.map((s, i) => (
        <div key={i} className="step-progress-item done">
          <div className="step-dot" />
          <div className="step-text">✅ {s.desc}</div>
        </div>
      ))}
      {active && (
        <div className="step-progress-item active">
          <div className="step-dot" />
          <div className="step-text">🚀 {active.desc}</div>
        </div>
      )}
    </div>
  );
}

export default function ChatPanel() {
  const [input, setInput] = useState('');
  const textareaRef = useRef<HTMLTextAreaElement>(null);
  const messagesEndRef = useRef<HTMLDivElement>(null);
  
  const { mode, setMode, selectedIds, showViz, setShowViz } = useAppStore();
  const { user } = useAuthStore();
  const { messages, activeStep, completedSteps, error, clearChat } = useChatStore();
  const { isStreaming, sendMessage, abortStream } = useChatStream();

  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [messages, activeStep, completedSteps]);

  const handleSend = () => {
    if (!input.trim() || isStreaming) return;
    sendMessage(input, mode, Array.from(selectedIds));
    setInput('');
  };

  const handleKey = (e: React.KeyboardEvent) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      handleSend();
    }
  };

  return (
    <>
      <div className="chat-header">
        <div className="chat-icon">🧠</div>
        <div style={{ flex: 1 }}>
          <div className="chat-title">Research Assistant</div>
          <div className="chat-sub">
            {mode === 'kg' ? 'Knowledge Graph Explorer' : 'Hybrid Vector RAG'}
          </div>
        </div>
        
        <div className="header-right" style={{display:'flex', gap:10, alignItems:'center'}}>
          <div className="mode-toggle">
            <button 
              className={`mode-btn ${mode === 'vector' ? 'active' : ''}`}
              onClick={() => setMode('vector')}
            >⚡ Vector</button>
            <button 
              className={`mode-btn ${mode === 'kg' ? 'active kg' : ''}`}
              onClick={() => setMode('kg')}
            >🕸️ Graph</button>
          </div>

          <button 
            className={`viz-toggle ${showViz ? 'active' : ''}`}
            onClick={() => setShowViz(!showViz)}
          >
            <span className="dot" />
            Pipeline
          </button>

          {user && (
            <div className="user-area">
              <img 
                className="user-avatar" 
                src={user.picture || 'https://via.placeholder.com/28'} 
                alt={user.name} 
              />
              <span className="user-name">{user.name.split(' ')[0]}</span>
              <div className={`user-key-dot ${user.has_api_key ? 'has-key' : ''}`} />
            </div>
          )}
        </div>
      </div>

      <div className="chat-messages">
        {messages.length === 0 ? (
          <div className="empty-state">
            <div className="empty-icon">🔭</div>
            <div className="empty-title">ArXiv Intel v1.2</div>
            <p className="empty-sub">Ask a question about the 100 seminal AI/ML papers. Our RAG pipeline will trace the research for you.</p>
          </div>
        ) : (
          messages.map((m, i) => (
            <div key={i} className={`message ${m.role} msg-fade`}>
              <div className="msg-label">{m.role === 'user' ? 'You' : 'ArXiv Assistant'}</div>
              <div className="bubble">
                {m.role === 'user' ? (
                  <div style={{whiteSpace:'pre-wrap'}}>{m.content}</div>
                ) : (
                  <div className="md-content">
                    <ReactMarkdown 
                      remarkPlugins={[remarkGfm, remarkBreaks]}
                    >
                      {m.content}
                    </ReactMarkdown>
                  </div>
                )}
                {m.streaming && <span className="streaming-dot" style={{display:'inline-block', width:8, height:8, background:'var(--accent2)', borderRadius:'50%', marginLeft:6, animation:'pulse-dot 1s infinite'}} />}
              </div>
            </div>
          ))
        )}
        
        {/* Intermediate Pipeline Feedback */}
        {isStreaming && messages[messages.length - 1]?.role === 'user' && (
           <div className="message assistant msg-fade">
             <div className="msg-label">Thinking...</div>
             <ThinkingProgress active={activeStep} completed={completedSteps} />
           </div>
        )}
        
        {/* Status indicator while message is streaming */}
        {isStreaming && activeStep && messages[messages.length - 1]?.role === 'assistant' && (
           <div className="typing-bubble" style={{marginLeft: 28}}>
              <div className="typing-dots">
                <span /> <span /> <span />
              </div>
              <span style={{fontSize: 11, marginLeft: 8, color: 'var(--td)'}}>
                {activeStep.desc}
              </span>
           </div>
        )}

        {error && <div className="error-msg">⚠️ {error}</div>}
        <div ref={messagesEndRef} />
      </div>

      <div className="input-bar">
        <div className="input-row">
          <div className="input-wrap">
            <textarea 
              ref={textareaRef}
              placeholder="Deep dive into a paper or topic..." 
              value={input}
              onChange={e => setInput(e.target.value)}
              onKeyDown={handleKey}
              rows={1}
            />
            {isStreaming ? (
              <button className="send-btn" onClick={abortStream} title="Stop generation">⏹</button>
            ) : (
              <button 
                className="send-btn" 
                onClick={handleSend}
                disabled={!input.trim()}
                title="Send message"
              >➤</button>
            )}
          </div>
        </div>
        <div className="input-hint" style={{display:'flex',justifyContent:'space-between'}}>
          <span>{selectedIds.size > 0 ? `🎯 Searching ${selectedIds.size} papers` : '📚 Searching all papers'}</span>
          <button className="clear-btn" onClick={() => clearChat()} style={{padding:'2px 8px',fontSize:10}}>Reset chat</button>
        </div>
      </div>
    </>
  );
}
