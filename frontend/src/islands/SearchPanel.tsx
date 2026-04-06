import { useEffect, useState, useMemo } from 'react';
import { useAppStore } from '../stores/appStore';
import type { Paper } from '../types/api';

export default function SearchPanel() {
  const [papers, setPapers] = useState<Paper[]>([]);
  const [loading, setLoading] = useState(true);
  const [filter, setFilter] = useState('');
  
  const { mode, selectedIds, togglePaper, clearSelected } = useAppStore();

  useEffect(() => {
    fetch('/api/papers')
      .then(r => r.json())
      .then(d => {
        setPapers(d.papers || []);
        setLoading(false);
      })
      .catch(() => setLoading(false));
  }, []);

  const filtered = useMemo(() => {
    return papers.filter(p => 
      !filter || 
      p.title.toLowerCase().includes(filter.toLowerCase()) ||
      p.arxiv_id.includes(filter)
    );
  }, [papers, filter]);

  return (
    <>
      <div className="sidebar-header">
        <div className="brand">
          <div className="brand-icon">🧠</div>
          <div>
            <div className="brand-name">ArXiv RAG</div>
            <div className="brand-sub">AI Research Assistant</div>
          </div>
        </div>
        <div className="pipeline-pills">
          {mode === 'vector' ? (
            <>
              <span className="pill on">⚡ Dense</span>
              <span className="pill on">📝 BM25</span>
              <span className="pill on">🔀 RRF</span>
              <span className="pill on">🎯 Rerank</span>
            </>
          ) : (
            <>
              <span className="pill on">⚡ Dense</span>
              <span className="pill on">🕸️ Graph</span>
              <span className="pill on">🎯 Rerank</span>
              <span className="pill off">📝 BM25 ✗</span>
            </>
          )}
        </div>
        <div className="search-box">
          <span className="search-icon">🔍</span>
          <input 
            placeholder="Filter papers…" 
            value={filter} 
            onChange={e => setFilter(e.target.value)} 
          />
        </div>
      </div>

      <div className="sidebar-count">
        Papers <span className="count-badge">{filtered.length}</span>
      </div>

      <div className="papers-list">
        {loading ? (
          <div className="no-papers">Loading…</div>
        ) : filtered.length === 0 ? (
          <div className="no-papers">No papers indexed yet</div>
        ) : (
          filtered.map(p => (
            <div 
              key={p.arxiv_id} 
              className={`paper-item ${selectedIds.has(p.arxiv_id) ? 'selected' : ''}`}
              onClick={() => togglePaper(p.arxiv_id)}
            >
              <div className="paper-title">{p.title}</div>
              <div className="paper-meta">
                <span className="paper-year">{p.arxiv_id.split('.')[0]}</span>
                <span className="paper-category">{p.status}</span>
              </div>
              {selectedIds.has(p.arxiv_id) && <span className="check-mark">✓</span>}
            </div>
          ))
        )}
      </div>

      <div className="sidebar-footer">
        <div className="scope-indicator" style={{marginBottom: 8}}>
          <div className={`scope-dot ${selectedIds.size === 0 ? 'all' : ''}`} />
          {selectedIds.size === 0 ? 'All papers' : `Selected: ${selectedIds.size}`}
          {selectedIds.size > 0 && (
            <button className="clear-btn" onClick={clearSelected}>Clear</button>
          )}
        </div>

        {/* Ingestion Monitor Widget */}
        <IngestionMonitor />

        <button className="new-chat-btn" onClick={() => window.dispatchEvent(new CustomEvent('new-chat'))}>
          + New conversation
        </button>
      </div>
    </>
  );
}

function IngestionMonitor() {
  const [status, setStatus] = useState<any>(null);

  useEffect(() => {
    const poll = async () => {
      try {
        const r = await fetch('/api/ingestion/status');
        const d = await r.json();
        setStatus(d);
      } catch (e) {}
    };
    poll();
    const timer = setInterval(poll, 8000);
    return () => clearInterval(timer);
  }, []);

  if (!status || status.total === 0) return null;

  const pct = Math.round((status.done / status.total) * 100);

  return (
    <div style={{
      padding: '10px',
      background: 'rgba(255,255,255,0.03)',
      border: '1px solid var(--border)',
      borderRadius: '8px',
      marginBottom: '10px'
    }}>
      <div style={{display: 'flex', justifyContent: 'space-between', fontSize: '10px', marginBottom: '6px', color: 'var(--td)'}}>
        <span>Indexing Progress</span>
        <span>{pct}%</span>
      </div>
      <div style={{height: '4px', background: 'rgba(255,255,255,0.05)', borderRadius: '2px', overflow: 'hidden'}}>
        <div style={{width: `${pct}%`, height: '100%', background: 'var(--accent)', transition: 'width 1s ease'}} />
      </div>
      <div style={{fontSize: '9px', marginTop: '6px', color: 'var(--tdm)', display: 'flex', gap: '8px'}}>
        <span>Total: {status.total}</span>
        <span style={{color: 'var(--success)'}}>Ready: {status.done}</span>
      </div>
    </div>
  );
}
