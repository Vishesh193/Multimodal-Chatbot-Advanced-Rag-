import React from 'react';
import { BrowserRouter, Routes, Route, Link, useLocation, Outlet } from 'react-router-dom';
import { Shield, FileText, UploadCloud, MessageSquare, Briefcase, FileSearch, PieChart } from 'lucide-react';

// Common Components
import Navbar from './components/Navbar';
import Footer from './components/Footer';

// Pages
import LandingPage from './pages/LandingPage';
import Dashboard from './pages/Dashboard';
import ChatInterface from './pages/ChatInterface';
import PolicyUpload from './pages/PolicyUpload';
import BillScanner from './pages/BillScanner';
import ClaimChecklist from './pages/ClaimChecklist';
import PolicyComparator from './pages/PolicyComparator';

// ── 1. Website Layout (Navbar + Footer) ──
function WebsiteLayout() {
  return (
    <div style={{ display: 'flex', flexDirection: 'column', minHeight: '100vh', width: '100vw' }}>
      <Navbar />
      <main style={{ flex: 1 }}>
        <Outlet />
      </main>
      <Footer />
    </div>
  );
}

// ── 2. App Sidebar ──
function AppSidebar() {
  const location = useLocation();
  const currentPath = location.pathname;

  const links = [
    { to: '/app/overview', label: 'Overview', icon: <PieChart size={20} /> },
    { to: '/app/chat', label: 'Insurance Assistant', icon: <MessageSquare size={20} /> },
    { to: '/app/comparator', label: 'Compare Policies', icon: <Briefcase size={20} /> },
    { to: '/app/checklist', label: 'Claim Checklist', icon: <FileText size={20} /> },
    { to: '/app/bills', label: 'Bill Scanner', icon: <FileSearch size={20} /> },
    { to: '/app/policies', label: 'Upload Policy', icon: <UploadCloud size={20} /> },
  ];

  return (
    <aside className="glass-panel" style={{ width: 'var(--sidebar-width)', margin: '16px', display: 'flex', flexDirection: 'column', background: 'white' }}>
      <div style={{ padding: '24px', borderBottom: '1px solid var(--glass-border)' }}>
        <h2 className="gradient-text" style={{ fontSize: '1.2rem', display: 'flex', alignItems: 'center', gap: '8px', margin: 0 }}>
          <Shield color="var(--accent-blue)" size={20} />
          InsureAI
        </h2>
        <p style={{ color: 'var(--text-muted)', fontSize: '0.85rem', marginTop: '4px' }}>Dashboard</p>
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
              color: currentPath === link.to ? 'var(--accent-deep)' : 'var(--text-secondary)',
              textDecoration: 'none',
              background: currentPath === link.to ? 'var(--accent-light)' : 'transparent',
              borderRight: currentPath === link.to ? '3px solid var(--accent-blue)' : '3px solid transparent',
              transition: 'all 0.2s',
              fontWeight: currentPath === link.to ? 600 : 500
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
        <p>InsureAI Platform v2.0</p>
        <Link to="/" style={{ color: 'var(--accent-blue)', textDecoration: 'none', marginTop: '8px', display: 'inline-block' }}>&larr; Back to Website</Link>
      </div>
    </aside>
  );
}

// ── 3. App Layout (Navbar + Sidebar + Content) ──
function AppLayout() {
  return (
    <div style={{ display: 'flex', flexDirection: 'column', height: '100vh', width: '100vw' }}>
      <Navbar />
      <div className="app-container" style={{ height: 'calc(100vh - var(--navbar-height))' }}>
        <AppSidebar />
        <main className="main-content" style={{ padding: '16px 16px 16px 0' }}>
          <div className="glass-panel fade-in" style={{ flex: 1, padding: '32px', overflowY: 'auto', background: 'white' }}>
            {/* Header banner explaining assistant */}
            <div style={{ marginBottom: '24px', padding: '16px 24px', background: 'var(--accent-light)', border: '1px solid #DBEAFE', borderRadius: '8px', display: 'flex', alignItems: 'center', gap: '12px', color: 'var(--accent-deep)' }}>
              <Shield size={24} />
              <div>
                <h4 style={{ margin: 0, fontSize: '1rem' }}>AI Insurance Assistant Active</h4>
                <p style={{ margin: 0, fontSize: '0.9rem', color: 'var(--text-secondary)' }}>Your completely private LLM environment is ready to securely analyze your uploaded documents.</p>
              </div>
            </div>

            <Outlet />
          </div>
        </main>
      </div>
    </div>
  );
}

// ── 4. Main Router ──
function App() {
  return (
    <BrowserRouter>
      <Routes>
        {/* Public Website Routes */}
        <Route element={<WebsiteLayout />}>
          <Route path="/" element={<LandingPage />} />
        </Route>

        {/* Private App Routes */}
        <Route path="/app" element={<AppLayout />}>
          <Route path="overview" element={<Dashboard />} />
          <Route path="chat" element={<ChatInterface />} />
          <Route path="comparator" element={<PolicyComparator />} />
          <Route path="checklist" element={<ClaimChecklist />} />
          <Route path="bills" element={<BillScanner />} />
          <Route path="policies" element={<PolicyUpload />} />
        </Route>
      </Routes>
    </BrowserRouter>
  );
}

export default App;
