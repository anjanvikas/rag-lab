import { create } from 'zustand';

interface AppState {
  mode: 'vector' | 'kg';
  selectedIds: Set<string>;
  showViz: boolean;
  setMode: (mode: 'vector' | 'kg') => void;
  togglePaper: (id: string) => void;
  clearSelected: () => void;
  setShowViz: (show: boolean) => void;
}

export const useAppStore = create<AppState>((set) => ({
  mode: 'vector',
  selectedIds: new Set(),
  showViz: false,
  setMode: (mode) => set({ mode }),
  setShowViz: (show) => set({ showViz: show }),
  togglePaper: (id) => set((state) => {
    const next = new Set(state.selectedIds);
    if (next.has(id)) next.delete(id);
    else next.add(id);
    return { selectedIds: next };
  }),
  clearSelected: () => set({ selectedIds: new Set() }),
}));
