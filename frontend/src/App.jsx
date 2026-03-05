import { BrowserRouter, Routes, Route, Link, useLocation } from 'react-router-dom';
import { Shield, FileText, UploadCloud, MessageSquare, Briefcase, FileSearch } from 'lucide-react';
import React from 'react';

// Pages
import Dashboard from './pages/Dashboard';
import ChatInterface from './pages/ChatInterface';
import PolicyUpload from './pages/PolicyUpload';
import BillScanner from './pages/BillScanner';
import ClaimChecklist from './pages/ClaimChecklist';
import PolicyComparator from './pages/PolicyComparator';

function Sidebar() {
  const location = useLocation();
  const currentPath = location.pathname;

  const links = [
    { to: '/', label: 'Overview', icon: <Shield size={20} /> },
    { to: '/chat', label: 'Insurance Assistant', icon: <MessageSquare size={20} /> },
    { to: '/comparator', label: 'Compare Policies', icon: <Briefcase size={20} /> },
    { to: '/checklist', label: 'Claim Checklist', icon: <FileText size={20} /> },
    { to: '/bills', label: 'Bill Scanner', icon: <FileSearch size={20} /> },
    { to: '/policies', label: 'Upload Policy', icon: <UploadCloud size={20} /> },
  ];

  return (
    <aside className="sidebar glass-panel" style={{ width: 'var(--sidebar-width)', margin: '16px', display: 'flex', flexDirection: 'column' }}>
      <div className="sidebar-header" style={{ padding: '24px', borderBottom: '1px solid var(--glass-border)' }}>
        <h2 className="gradient-text" style={{ fontSize: '1.5rem', display: 'flex', alignItems: 'center', gap: '8px' }}>
          <Shield color="var(--accent-purple)" />
          InsureAI
        </h2>
        <p style={{ color: 'var(--text-muted)', fontSize: '0.85rem', marginTop: '4px' }}>Smart Claims Assistant</p>
      </div>

      <nav style={{ padding: '16px 0', flex: 1 }}>
        {links.map(link => (
          <Link
            key={link.to}
            to={link.to}
            style={{
              display: 'flex',
              alignItems: 'center',
              gap: '12px',
              padding: '12px 24px',
              color: currentPath === link.to ? 'white' : 'var(--text-secondary)',
              textDecoration: 'none',
              background: currentPath === link.to ? 'var(--bg-tertiary)' : 'transparent',
              borderRight: currentPath === link.to ? '3px solid var(--accent-blue)' : '3px solid transparent',
              transition: 'all 0.2s',
              fontWeight: currentPath === link.to ? 500 : 400
            }}
          >
            {React.cloneElement(link.icon, {
              color: currentPath === link.to ? 'var(--accent-blue)' : 'currentColor'
            })}
            {link.label}
          </Link>
        ))}
      </nav>

      <div style={{ padding: '24px', fontSize: '0.8rem', color: 'var(--text-muted)' }}>
        <p>Enterprise RAG System v2.0</p>
        <p>Powered by Groq & Ollama</p>
      </div>
    </aside>
  );
}

function App() {
  return (
    <BrowserRouter>
      <div className="app-container">
        <div className="bg-glow-orb" style={{ top: '-20%', left: '-10%', width: '600px', height: '600px', background: 'var(--accent-blue)' }}></div>
        <div className="bg-glow-orb" style={{ bottom: '-20%', right: '-10%', width: '500px', height: '500px', background: 'var(--accent-purple)' }}></div>

        <Sidebar />

        <main className="main-content" style={{ padding: '16px 16px 16px 0' }}>
          <div className="glass-panel fade-in" style={{ flex: 1, padding: '32px', overflowY: 'auto' }}>
            <Routes>
              <Route path="/" element={<Dashboard />} />
              <Route path="/chat" element={<ChatInterface />} />
              <Route path="/comparator" element={<PolicyComparator />} />
              <Route path="/checklist" element={<ClaimChecklist />} />
              <Route path="/bills" element={<BillScanner />} />
              <Route path="/policies" element={<PolicyUpload />} />
            </Routes>
          </div>
        </main>
      </div>
    </BrowserRouter>
  );
}

export default App;
