import { create } from 'zustand';
import type { ChatMessage, TraceData } from '../types/api';

export interface PipelineStep {
  name: string;
  desc: string;
  status: 'active' | 'done' | 'failed';
}

interface ChatState {
  messages: ChatMessage[];
  trace: TraceData;
  activeStep: PipelineStep | null;
  completedSteps: PipelineStep[];
  isStreaming: boolean;
  error: string | null;
  
  setMessages: (fn: (prev: ChatMessage[]) => ChatMessage[]) => void;
  setTrace: (fn: (prev: TraceData) => TraceData) => void;
  addCompletedStep: (step: PipelineStep) => void;
  setActiveStep: (step: PipelineStep | null) => void;
  setIsStreaming: (streaming: boolean) => void;
  setError: (error: string | null) => void;
  clearChat: () => void;
}

export const useChatStore = create<ChatState>((set) => ({
  messages: [],
  trace: {},
  activeStep: null,
  completedSteps: [],
  isStreaming: false,
  error: null,
  
  setMessages: (fn) => set((s) => ({ messages: fn(s.messages) })),
  setTrace: (fn) => set((s) => ({ trace: fn(s.trace) })),
  addCompletedStep: (step) => set((s) => ({ completedSteps: [...s.completedSteps, { ...step, status: 'done' }] })),
  setActiveStep: (activeStep) => set({ activeStep }),
  setIsStreaming: (isStreaming) => set({ isStreaming }),
  setError: (error) => set({ error }),
  clearChat: () => set({ messages: [], trace: {}, activeStep: null, completedSteps: [], error: null, isStreaming: false }),
}));
