import { StrictMode } from 'react';
import { createRoot } from 'react-dom/client';
import { App } from './app/App';
import './app/theme/theme.css';
import './app/app.css';

// Set the theme attribute before first paint to avoid a flash.
try {
  const t = localStorage.getItem('nervegear.theme');
  document.documentElement.dataset.theme = t === 'light' ? 'light' : 'dark';
} catch {
  document.documentElement.dataset.theme = 'dark';
}

createRoot(document.getElementById('root')!).render(
  <StrictMode>
    <App />
  </StrictMode>,
);
