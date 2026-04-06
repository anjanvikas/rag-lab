import { useCallback, useRef } from 'react';
import type { ChatMessage } from '../types/api';
import { useChatStore } from '../stores/chatStore';

export function useChatStream() {
  const { messages, setMessages, setTrace, setActiveStep, setError, setIsStreaming, isStreaming } = useChatStore();
  const abortCtrl = useRef<AbortController | null>(null);

  const sendMessage = useCallback(async (question: string, mode: string = 'vector', selectedIds: string[] = []) => {
    if (!question.trim() || isStreaming) return;
    
    setError(null);
    setTrace(() => ({}));
    setActiveStep(null);
    
    const userMsg: ChatMessage = { role: 'user', content: question };
    setMessages(prev => [...prev, userMsg]);
    setIsStreaming(true);

    abortCtrl.current = new AbortController();

    try {
      const startRes = await fetch('/api/chat/start', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ message: question, mode, selected_ids: selectedIds }),
        signal: abortCtrl.current.signal
      });

      if (!startRes.ok) throw new Error(`Failed to start chat: ${startRes.status}`);
      const { stream_id } = await startRes.json();

      const es = new EventSource(`/api/chat/stream?stream_id=${stream_id}`);
      
      es.addEventListener('step', (e: any) => {
        try {
          const data = JSON.parse(e.data);
          const { step, desc } = data;
          
          useChatStore.getState().setMessages(prev => {
            // Re-use current messages, but we use the store to update steps
            return prev;
          });

          const currentActive = useChatStore.getState().activeStep;
          if (currentActive && currentActive.name !== step) {
            useChatStore.getState().addCompletedStep(currentActive);
          }
          
          useChatStore.getState().setActiveStep({
            name: step,
            desc: desc || step,
            status: 'active'
          });
        } catch (err) {
          console.error("Step parse error", err);
        }
      });

      es.addEventListener('trace', (e: any) => {
        try {
          const data = JSON.parse(e.data);
          setTrace(prev => ({ ...prev, ...data }));
        } catch {}
      });

      es.addEventListener('token', (e: any) => {
        try {
          const data = JSON.parse(e.data);
          setMessages(prev => {
            const next = [...prev];
            const last = next[next.length - 1];
            
            if (last && last.role === 'assistant') {
              next[next.length - 1] = { ...last, content: last.content + data.token };
              return next;
            } else {
              // Create the assistant message when first token arrives
              return [...next, { role: 'assistant', content: data.token, streaming: true, traceId: 'noop' }];
            }
          });
        } catch {}
      });

      es.addEventListener('done', (e: any) => {
        es.close();
        try {
          const data = JSON.parse(e.data);
          setMessages(prev => {
            const next = [...prev];
            next[next.length - 1] = { ...next[next.length - 1], streaming: false, traceId: data.trace_id };
            return next;
          });
          setActiveStep(null);
          setIsStreaming(false);
        } catch {}
      });

      es.onerror = () => {
        es.close();
        setError('Stream connection lost.');
        setIsStreaming(false);
        setActiveStep(null);
      };

    } catch (e: any) {
      if (e.name !== 'AbortError') {
        setError(e.message);
        setMessages(prev => prev.slice(0, -1));
      }
      setIsStreaming(false);
      setActiveStep(null);
    }
  }, [isStreaming, setMessages, setTrace, setActiveStep, setError, setIsStreaming]);

  const abortStream = useCallback(() => {
    if (abortCtrl.current) {
      abortCtrl.current.abort();
      setIsStreaming(false);
      setActiveStep(null);
    }
  }, [setIsStreaming, setActiveStep]);

  return { messages, isStreaming, sendMessage, abortStream };
}
