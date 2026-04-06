import { useEffect, useRef } from 'react';
import { useAppStore } from '../stores/appStore';
import { useChatStore } from '../stores/chatStore';
import Graph from 'graphology';
import { Sigma } from 'sigma';
// @ts-ignore
import { forceAtlas2 } from 'graphology-layout-forceatlas2';

export default function KGExplorer() {
  const { mode, showViz } = useAppStore();
  const { trace, activeStep } = useChatStore();
  const sigmaContainerRef = useRef<HTMLDivElement>(null);
  const sigmaInstance = useRef<Sigma | null>(null);

  useEffect(() => {
    if (mode === 'kg' && showViz && sigmaContainerRef.current) {
      const graph = new Graph();
      
      const nodes = (trace.retrieval as any)?.graph_nodes || [
        { id: '1', label: 'RAG Architecture', size: 15, color: '#63b3ed' },
        { id: '2', label: 'Vector DB', size: 10, color: '#9f7aea' },
        { id: '3', label: 'Graph DB', size: 10, color: '#9f7aea' }
      ];
      
      const edges = (trace.retrieval as any)?.graph_edges || [
        { source: '1', target: '2' },
        { source: '1', target: '3' }
      ];

      nodes.forEach((n: any) => {
        if (!graph.hasNode(n.id)) {
          graph.addNode(n.id, { 
            label: n.label, 
            size: n.size || 10, 
            color: n.color || '#63b3ed', 
            x: Math.random(), 
            y: Math.random() 
          });
        }
      });

      edges.forEach((e: any, i: number) => {
        if (!graph.hasEdge(e.source, e.target)) {
          graph.addEdgeWithKey(`e${i}`, e.source, e.target, { size: 2, color: '#4a5568' });
        }
      });

      forceAtlas2.assign(graph, { iterations: 50, settings: { gravity: 1 } });

      if (sigmaInstance.current) sigmaInstance.current.kill();
      
      sigmaInstance.current = new Sigma(graph, sigmaContainerRef.current, {
        renderEdgeLabels: true,
        labelFont: 'Inter',
        labelWeight: '600'
      });

      return () => {
        sigmaInstance.current?.kill();
        sigmaInstance.current = null;
      };
    }
  }, [mode, showViz, trace]);

  if (!showViz) return null;

  return (
    <div className={`viz-drawer ${showViz ? '' : 'hidden'}`}>
      <div className="viz-header">
        {activeStep ? <span className="live-dot" /> : <span>{mode === 'kg' ? '🕸️' : '🔍'}</span>}
        <span style={{flex: 1, marginLeft: 8}}>{mode === 'kg' ? 'Graph Explorer' : 'Pipeline Trace'}</span>
      </div>
      
      <div className="viz-scroll" style={{ display: 'flex', flexDirection: 'column' }}>
        {mode === 'vector' ? (
          <div className="pipeline-steps">
            <TraceStep num={1} title="Query Rewriting" active={activeStep === 'rewrite'} data={trace.rewrite} theme="rewrite" />
            <TraceStep num={2} title="Hybrid Retrieval" active={activeStep === 'retrieve'} data={trace.retrieval} theme="retrieve" />
            <TraceStep num={3} title="Cross-Encoder Rerank" active={activeStep === 'rerank'} data={trace.rerank} theme="rerank" />
            <TraceStep num={4} title="Model Selection" active={activeStep === 'generate'} data={trace.model} theme="model" />
          </div>
        ) : (
          <div style={{ flex: 1, position: 'relative', minHeight: '400px' }}>
             <div ref={sigmaContainerRef} style={{ position: 'absolute', inset: 0, background: '#090d1a', borderRadius: '12px' }} />
             <div style={{ position: 'absolute', bottom: 12, left: 12, pointerEvents: 'none', fontSize: 10, color: 'var(--tdm)' }}>
                Powered by Sigma.js + Neo4j
             </div>
          </div>
        )}
      </div>
    </div>
  );
}

function TraceStep({ num, title, active, data, theme }: any) {
  if (!data && !active) return null;
  return (
    <div className={`vcard ${theme}`} style={{ marginBottom: 10 }}>
      <div className="vcard-head">
        <div className="step-num">{num}</div>
        <span>{title}</span>
      </div>
      <div className="vcard-body">
        {active && <div className="loading-dots"><span>.</span><span>.</span><span>.</span></div>}
        {data && (
          <div style={{fontSize: 11, color: 'var(--td)'}}>
             {num === 1 && <div><strong>Rewritten:</strong> {data.rewritten}</div>}
             {num === 2 && <div><strong>Fused:</strong> {data.fused_count} chunks</div>}
             {num === 3 && <div><strong>Top Score:</strong> {data.chunks?.[0]?.score?.toFixed(3)}</div>}
             {num === 4 && <div><strong>LLM:</strong> Claude {data.model}</div>}
          </div>
        )}
      </div>
    </div>
  );
}
