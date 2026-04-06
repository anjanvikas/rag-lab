export interface Paper {
  arxiv_id: string;
  title: string;
  tier: number;
  status: string;
  chunks: number;
}

export interface TraceData {
  rewrite?: any;
  retrieval?: {
    mode: string;
    fused_count: number;
    dense_count: number;
    bm25_count: number;
    fused_top: any[];
  };
  rerank?: {
    input_count: number;
    output_count: number;
    chunks: any[];
  };
  model?: any;
}

export interface ChatMessage {
  role: 'user' | 'assistant';
  content: string;
  streaming?: boolean;
  traceId?: string;
}

export interface AuthUser {
  id: string;
  email: string;
  name: string;
  picture: string;
  has_api_key: boolean;
}
