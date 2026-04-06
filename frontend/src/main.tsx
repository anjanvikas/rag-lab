import React from 'react'
import { createRoot } from 'react-dom/client'

// Island Components
import ChatPanel from './islands/ChatPanel'
import SearchPanel from './islands/SearchPanel'
import KGExplorer from './islands/KGExplorer'

// Global styles injected here
import './index.css'

function mountIfPresent(id: string, Component: React.ComponentType) {
  const el = document.getElementById(id)
  if (el) {
    createRoot(el).render(
      <React.StrictMode>
        <Component />
      </React.StrictMode>
    )
  }
}

// Map DOM IDs to React Islands
mountIfPresent('chat-root', ChatPanel)
mountIfPresent('search-root', SearchPanel)
mountIfPresent('viz-root', KGExplorer)
