import { create } from 'zustand';
import type { AuthUser } from '../types/api';

interface AuthState {
  user: AuthUser | null;
  loading: boolean;
  setUser: (user: AuthUser | null) => void;
  checkAuth: () => Promise<void>;
  logout: () => Promise<void>;
}

export const useAuthStore = create<AuthState>((set) => ({
  user: null,
  loading: true,
  setUser: (user) => set({ user }),
  checkAuth: async () => {
    try {
      const res = await fetch('/auth/me');
      if (res.ok) {
        const data = await res.json();
        set({ user: data, loading: false });
      } else {
        set({ user: null, loading: false });
      }
    } catch {
      set({ user: null, loading: false });
    }
  },
  logout: async () => {
    // Basic reload for now, the FastAPI handles cookie clearing
    window.location.href = '/login';
  }
}));
