import React from 'react';
import { Shield } from 'lucide-react';
import { Link } from 'react-router-dom';

export default function Footer() {
    return (
        <footer style={{ background: '#0F172A', color: 'white', padding: '64px 5% 32px 5%' }}>
            <div style={{ display: 'flex', flexWrap: 'wrap', gap: '64px', justifyContent: 'space-between', marginBottom: '64px' }}>

                <div style={{ maxWidth: '300px' }}>
                    <div style={{ display: 'flex', alignItems: 'center', gap: '8px', marginBottom: '16px' }}>
                        <Shield size={28} color="var(--accent-blue)" />
                        <span style={{ fontSize: '1.5rem', fontWeight: 700, fontFamily: 'Poppins, sans-serif' }}>
                            InsureAI
                        </span>
                    </div>
                    <p style={{ color: '#94A3B8', lineHeight: 1.6, fontSize: '0.95rem' }}>
                        AI-powered insurance policy intelligence. Understand coverage, claim smarter, and resolve complex policy comparisons in seconds.
                    </p>
                </div>

                <div style={{ display: 'flex', gap: '64px', flexWrap: 'wrap' }}>
                    <div>
                        <h4 style={{ color: 'white', marginBottom: '24px', fontSize: '1.1rem' }}>Product</h4>
                        <ul style={{ listStyle: 'none', display: 'flex', flexDirection: 'column', gap: '12px', padding: 0 }}>
                            <li><Link to="/app/chat" style={{ color: '#94A3B8', textDecoration: 'none' }}>Insurance Assistant</Link></li>
                            <li><Link to="/app/comparator" style={{ color: '#94A3B8', textDecoration: 'none' }}>Policy Comparison</Link></li>
                            <li><Link to="/app/checklist" style={{ color: '#94A3B8', textDecoration: 'none' }}>Claim Guide</Link></li>
                        </ul>
                    </div>

                    <div>
                        <h4 style={{ color: 'white', marginBottom: '24px', fontSize: '1.1rem' }}>Resources</h4>
                        <ul style={{ listStyle: 'none', display: 'flex', flexDirection: 'column', gap: '12px', padding: 0 }}>
                            <li><a href="#faq" style={{ color: '#94A3B8', textDecoration: 'none' }}>FAQ</a></li>
                            <li><Link to="#" style={{ color: '#94A3B8', textDecoration: 'none' }}>Documentation</Link></li>
                            <li><Link to="#" style={{ color: '#94A3B8', textDecoration: 'none' }}>Privacy Policy</Link></li>
                        </ul>
                    </div>

                    <div>
                        <h4 style={{ color: 'white', marginBottom: '24px', fontSize: '1.1rem' }}>Company</h4>
                        <ul style={{ listStyle: 'none', display: 'flex', flexDirection: 'column', gap: '12px', padding: 0 }}>
                            <li><Link to="#" style={{ color: '#94A3B8', textDecoration: 'none' }}>About</Link></li>
                            <li><a href="#contact" style={{ color: '#94A3B8', textDecoration: 'none' }}>Contact</a></li>
                        </ul>
                    </div>
                </div>
            </div>

            <div style={{ borderTop: '1px solid #1E293B', paddingTop: '32px', display: 'flex', justifyContent: 'space-between', alignItems: 'center', color: '#64748B', fontSize: '0.9rem' }}>
                <p>© 2026 InsureAI – AI Powered Insurance Intelligence.</p>
                <p>Built for the future of insurtech.</p>
            </div>
        </footer>
    );
}
